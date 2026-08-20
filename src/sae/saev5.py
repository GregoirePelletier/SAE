"""
saev5.py — Pipeline double SAE (Gemma-3 + F2LLM Embedding-SAE).

Pipeline 1 : Gemma-3 → SAE GemmaScope-2 préentraîné (core) + extension
résiduelle entraînée sur le domaine (`FrozenCoreResidualSAE`).
Pipeline 2 : F2LLM → `PhraseLevelSAE` entraîné from-scratch sur des
embeddings de phrase.
"""

import os
import sys
import time
import urllib3
import requests
import glob
import pickle
import json
import math
import random
import re
from contextlib import contextmanager
from requests.sessions import Session
from sae_lens.registry import SAE_CLASS_REGISTRY
import gc


@contextmanager
def stage_timer(name: str):
    """Chronomètre une étape de haut niveau du pipeline (corpus, P1, P2, ...)
    et affiche sa durée -- pour repérer un temps anormal d'un run à l'autre."""
    t0 = time.perf_counter()
    print(f"  [timing] {name}...")
    try:
        yield
    finally:
        print(f"  [timing] {name} terminé en {time.perf_counter() - t0:.1f}s")

try:
    from src.analysis.activations import valid_token_mask, norm_outlier_mask
except ImportError:
    from activations import valid_token_mask, norm_outlier_mask

from src.config import (
    MODEL_SIZE, MODEL_ID, RELEASE_ID, SAE_ID, LAYER, D_MODEL, HOOK_TYPE,
    LOCAL_SAE_ROOT, SAE_SNAPSHOT, DTYPE, CLUSTER_OFFLINE_MODE,
)

# Compatibilité Gemma Scope 2
if "jump_relu" not in SAE_CLASS_REGISTRY and "jumprelu" in SAE_CLASS_REGISTRY:
    SAE_CLASS_REGISTRY["jump_relu"] = SAE_CLASS_REGISTRY["jumprelu"]

# ======================================================================
# CONFIGURATION ET PATCHS SÉCURITÉ RESEAU (CLUSTER & FRONT DGX)
# ======================================================================
# Le cluster SLURM (pas d'accès internet direct, proxy à certificat auto-signé)
# nécessite de désactiver la vérif SSL et de forcer le mode offline HF une fois
# les modèles/SAE mis en cache. En local (CLUSTER_OFFLINE_MODE=0, défaut), ces
# patchs sont sautés pour permettre les premiers téléchargements.

if CLUSTER_OFFLINE_MODE:
    os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
    os.environ["CURL_CA_BUNDLE"] = ""

    _old_merge_environment_settings = Session.merge_environment_settings

    def patched_merge_environment_settings(self, url, proxies, stream, verify, cert):
        settings = _old_merge_environment_settings(self, url, proxies, stream, verify, cert)
        settings['verify'] = False
        return settings

    Session.merge_environment_settings = patched_merge_environment_settings

    import huggingface_hub.utils
    import huggingface_hub.file_download

    _old_get_session = huggingface_hub.utils.get_session

    def patched_get_session():
        session = _old_get_session()
        session.verify = False
        return session

    huggingface_hub.utils.get_session = patched_get_session
    huggingface_hub.file_download.get_session = patched_get_session

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

import sae_lens.loading.pretrained_sae_loaders as sae_loaders

def mocked_get_safetensors_tensor_shapes(url, headers=None, timeout=10):
    """Lit les configurations de formes directement en mémoire locale."""
    from pathlib import Path
    cache_dir = Path(os.path.expanduser(f"~/.cache/huggingface/hub/models--google--{RELEASE_ID}"))
    d_model = D_MODEL
    d_sae = 16384

    if cache_dir.exists():
        snapshots = list(cache_dir.iterdir())
        if snapshots:
            config_local = snapshots[0] / f"resid_post/{SAE_ID}/config.json"
            if config_local.exists():
                try:
                    with open(config_local, "r") as f:
                        cfg = json.load(f)
                    d_sae = cfg.get("dict_size", d_sae)
                    d_model = cfg.get("act_size", d_model)
                except Exception:
                    pass

    return {
        "w_enc": [d_model, d_sae], "b_enc": [d_sae], "w_dec": [d_sae, d_model], "b_dec": [d_model],
        "W_enc": [d_model, d_sae], "B_enc": [d_sae], "W_dec": [d_sae, d_model], "B_dec": [d_model],
    }

sae_loaders.get_safetensors_tensor_shapes = mocked_get_safetensors_tensor_shapes

# ======================================================================
# IMPORTS APPLICATIFS STANDARDS
# ======================================================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

# fp16 par défaut en local (GPU Turing 6 Go sans bf16 natif) ; bf16 dispo via DTYPE=bf16 (cluster).
TORCH_DTYPE = torch.bfloat16 if DTYPE == "bf16" else torch.float16


def open_mmap_reservoir(path, n_rows, d_in, dtype):
    """Tenseur 2D adossé à un fichier disque (mmap), jamais matérialisé en RAM
    anonyme — pages lisibles/inscriptibles par bloc, récupérables par l'OS
    sous pression mémoire, contrairement à une allocation anonyme."""
    return torch.from_file(path, shared=True, size=n_rows * d_in, dtype=dtype).view(n_rows, d_in)


# Reprise après coupure (R1, AUDIT_SAE_2026-08.md §2.3/§4.3) -- brique partagée
# avec phrase_sae.py (Pipeline 2), cf. docstring de src/storage/checkpoint.py.
try:
    from src.storage.checkpoint import (
        checkpoint_path as _checkpoint_path,
        read_checkpoint as _read_checkpoint,
        write_checkpoint as _write_checkpoint,
        clear_checkpoint as _clear_checkpoint,
        GracefulShutdown as _GracefulShutdown,
    )
except ImportError:
    from checkpoint import (
        checkpoint_path as _checkpoint_path,
        read_checkpoint as _read_checkpoint,
        write_checkpoint as _write_checkpoint,
        clear_checkpoint as _clear_checkpoint,
        GracefulShutdown as _GracefulShutdown,
    )


def _extraction_progress_path(cache_dir: str) -> str:
    return _checkpoint_path(cache_dir, "p1_extraction")


def _read_extraction_progress(cache_dir: str) -> dict | None:
    return _read_checkpoint(_extraction_progress_path(cache_dir))


def _write_extraction_progress(cache_dir: str, next_doc_idx: int, n_residuals_seen: int,
                                n_residuals_collected: int) -> None:
    _write_checkpoint(
        _extraction_progress_path(cache_dir),
        next_doc_idx=next_doc_idx, n_residuals_seen=n_residuals_seen,
        n_residuals_collected=n_residuals_collected,
    )


from sae_shared import (
    ENERGY_KEYWORDS, SPORTS_KEYWORDS, SUPPORT_KEYWORDS,
    ENERGY_URL_PATTERNS, SPORTS_URL_PATTERNS, SUPPORT_URL_PATTERNS,
    prepare_domain_dataset, sample_fineweb2_chunks, split_into_phrases, group_indices_by_doc,
    build_reencode_targets, is_filler_document,
    compute_metrics, compute_rho_sae,
    downstream_classification,
    steer_activations, steer_and_decode,
    build_email_train_test_corpus,
    FrozenCoreResidualSAE, SAEBoostResidualSAE, FrozenDecoderExtendedSAE,
    PhraseLevelSAE, extract_f2llm_embeddings,
    encode_documents_with_phrase_sae, load_or_train_sae,
    compute_sae_metrics,
    pool_embeddings_by_document,
    load_or_train_extended_sae,
)

from src.sae.judge import (
    extract_causal_context, build_feature_examples_with_control,
    feature_selection_by_magnitude, odd_one_out_judge, _apply_chat_and_extract,
    local_gemma_judge,
)

try:
    from src.analysis.cooccurrence import (
        compute_npmi, corpus_diff_stats, cooccurrence_graph, find_interesting_pairs,
    )
except ImportError:
    from cooccurrence import (
        compute_npmi, corpus_diff_stats, cooccurrence_graph, find_interesting_pairs,
    )

import ctypes

def _trim_host_memory():
    """Rend les arènes glibc libérées à l'OS après teardown d'un gros modèle.
    Sans cela, le RSS croît de façon monotone à chaque cycle load/unload
    (fragmentation malloc) → OOM SLURM même si Python a bien libéré."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if os.name != "nt":
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
try:
    from src.storage.fragment_store import (
        save_fragment, load_fragment, fragment_exists, list_fragment_ids,
        feature_column, doc_maxpool, decode_core_sparse, merge_extra, AsyncFragmentWriter,
    )
except ImportError:
    from fragment_store import (
        save_fragment, load_fragment, fragment_exists, list_fragment_ids,
        feature_column, doc_maxpool, decode_core_sparse, merge_extra, AsyncFragmentWriter,
    )

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU  : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

SEED = int(os.environ.get("SEED", "42"))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

from src.config import (
    HF_TOKEN, SAVE_DIR, LOCAL_DATASET_PATH, LOCAL_MAILS_PATH,
    LOCAL_AUGMENTED_MAILS_PATH, NEURONPEDIA_LABELS_PATH, CORPUS_SPLIT_SEED,
)

os.makedirs(SAVE_DIR, exist_ok=True)
CACHE_DIR = os.path.join(SAVE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

USE_FINEWEB2   = True
# Corpus de diffing cross-domaine (energy/sports/support) : SEUL usage restant,
# volontairement réduit (n'entraîne plus le SAE, cf. section "CORPUS PRINCIPAL"
# dans le bloc MAIN plus bas -- emails+augmentés dominent désormais l'entraînement).
N_TOTAL_ENERGY = int(os.environ.get("N_TOTAL_ENERGY", "300"))
N_TOTAL_SPORTS = int(os.environ.get("N_TOTAL_SPORTS", "300"))
N_TOTAL_SUPPORT = int(os.environ.get("N_TOTAL_SUPPORT", "300"))
# Proportion d'emails+augmentés réservée au test (le reste va en train -- objectif :
# maximiser la part réellement utilisée en entraînement, cf. décision utilisateur).
EMAIL_TEST_SPLIT = float(os.environ.get("EMAIL_TEST_SPLIT", "0.05"))
# Nombre max de variantes augmentées par mail original conservées (limite le
# déséquilibre train si un mail génère beaucoup plus de variantes qu'un autre,
# et borne le volume total si besoin de contrôler le temps de calcul).
MAX_AUGMENTED_PER_MAIL = int(os.environ.get("MAX_AUGMENTED_PER_MAIL", "13"))

from src.config import (
    EMB_MODEL, MATRYOSHKA_DIM, D_SAE, K_SPARSE, EPOCHS, LR, BATCH_TRAIN, MAX_PHRASES_DOC,
    D_EXTRA, K_EXTRA, EPOCHS_EXTRA, LR_EXTRA, USE_FROZEN_CORE, N_TOKENS_EXTRA_TRAIN,
    N_FEATURES_TO_LABEL, SANITY_CHECK_FROZEN_DECODER, EXTRACTION_BATCH_SIZE,
    EXTRACTION_CHECKPOINT_INTERVAL,
)
# MODEL_SIZE, MODEL_ID, RELEASE_ID, SAE_ID, LAYER, HOOK_TYPE, LOCAL_SAE_ROOT, SAE_SNAPSHOT
# sont déjà importés depuis src.config plus haut dans ce fichier — source unique de vérité,
# partagée avec download_sae.py et src/sae/gemma_scope_loader.py.

# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DU SAE PRÉENTRAÎNÉ
# ══════════════════════════════════════════════════════════════════════════════

from src.sae import load_gemma_scope_sae


def load_pretrained_sae() -> SAE:
    """
    Charge le SAE GemmaScope local (snapshot complet attendu), sinon
    fallback Hub via load_gemma_scope_sae.
    """
    sae_dir = os.path.join(
        LOCAL_SAE_ROOT, "snapshots", SAE_SNAPSHOT, HOOK_TYPE, SAE_ID
    )
    print(f"  [SAE] Tentative locale : {sae_dir}")
    return load_gemma_scope_sae(
        sae_dir=sae_dir,
        device=DEVICE,
        release_id=RELEASE_ID,
        sae_id=f"{HOOK_TYPE}/{SAE_ID}",
    )

# ══════════════════════════════════════════════════════════════════════════════
# PHRASE-LEVEL SAE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_silhouette(doc_acts: torch.Tensor, labels: list, n_max: int = 2000) -> float:
    try:
        from sklearn.metrics import silhouette_score
        X, lbl = doc_acts.float().detach().cpu().numpy(), np.array(labels)
        if len(set(lbl)) < 2:
            return float("nan")
        if X.shape[0] > n_max:
            idx = np.random.choice(X.shape[0], n_max, replace=False)
            X, lbl = X[idx], lbl[idx]
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        return float(silhouette_score(X_norm, lbl, metric="cosine"))
    except Exception as e:
        print(f"  Silhouette failed: {e}")
        return float("nan")


def get_activating_tokens_for_doc(
    token_strings: list,
    token_residuals: torch.Tensor,
    sae,
    top_feature_indices: list,
    top_k_tokens: int = 2,
) -> dict:
    W_enc = sae.W_enc
    if W_enc.shape[0] == sae.d_sae or (hasattr(sae, "d_extra") and W_enc.shape[0] == sae.d_sae + sae.d_extra):
        W_enc = W_enc.T
    W_enc_sub = W_enc[:, top_feature_indices].float()
    b_enc_sub = sae.b_enc[top_feature_indices].float()
    
    with torch.no_grad():
        pre = token_residuals.float() @ W_enc_sub + b_enc_sub
    result = {}
    for col, f_idx in enumerate(top_feature_indices):
        scores = pre[:, col]
        k = min(top_k_tokens, len(token_strings))
        top_idx = scores.topk(k).indices.tolist()
        result[f_idx] = [(token_strings[j], round(scores[j].item(), 3)) for j in top_idx]
    return result

# ══════════════════════════════════════════════════════════════════════════════
# TÂCHE 1 : DATASET DIFFING + HYPOTHÈSE LLM
# ══════════════════════════════════════════════════════════════════════════════

def generate_llm_diff_hypothesis(
    model, tokenizer,
    diff_df: pd.DataFrame,
    label_a: str, label_b: str,
) -> str:
    model.eval()
    top_diffs = diff_df[diff_df.get("significant", True) == True].head(8) \
        if "significant" in diff_df.columns else diff_df.head(8)
    features_desc = []
    for _, row in top_diffs.iterrows():
        features_desc.append(
            f"- Feature #{int(row['feature_id'])} ({row['label']}) : "
            f"{label_a}={row['freq_A']:.3f} vs {label_b}={row['freq_B']:.3f} "
            f"(log-odds={row['log_odds_ratio']:+.2f}, q={row['q']:.1e})"
        )
    prompt = (
        f"Chercheur en interprétabilité SAE EDF R&D. Corpus '{label_a}' vs '{label_b}'.\n"
        f"Features SAE les plus discriminantes :\n{chr(10).join(features_desc)}\n\n"
        "Hypothèse globale scientifique (français, 2–3 phrases) sur la divergence sémantique."
    )
    inputs = _apply_chat_and_extract(
    tokenizer, 
    [{"role": "user", "content": prompt}], 
    device=model.device, 
    add_generation_prompt=True, 
    return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model.generate(input_ids=inputs, max_new_tokens=256, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    return response.strip()

# ══════════════════════════════════════════════════════════════════════════════
# SÉLECTION DE LATENTS PAR SIMILARITÉ D'EMBEDDING (Tâches 3 & 4)
# ══════════════════════════════════════════════════════════════════════════════
# interp_embed (Jiang, Sun et al. 2025) sélectionne les
# latents pertinents pour une requête par SIMILARITÉ D'EMBEDDING DENSE entre le label
# du latent et la requête (§4.4, Appendix F.1 : "top k latents whose labels' dense
# embeddings are the most similar to that of a provided keyphrase"), pas par matching
# de sous-chaîne. `targeted_clustering_by_axis`/`property_based_retrieval` utilisaient
# un matching littéral (`word in label`), qui rate tout label sémantiquement lié mais
# formulé différemment (ex. requête "urgence" ne matche pas le label "demande pressante").
#
# Modèle d'embedding : bge-m3 (LATENT_LABEL_EMB_MODEL, pooling CLS), PAS F2LLM
# (utilisé partout ailleurs dans le projet, Pipeline 2) -- comparaison empirique
# documentée dans src/config.py : F2LLM (pooling dernier-token, optimisé pour des
# phrases complètes) donne de bons résultats sur certaines requêtes courtes mais des
# résultats sans rapport sur d'autres (ex. "facturation résiliation panne" ->
# "fact statement", "unlock loss cash"...) ; bge-m3 (multilingue, conçu pour la
# similarité sémantique de textes courts) est fiable sur les deux cas testés.

def _embed_bge_m3(texts: list[str], batch_size: int = 64) -> torch.Tensor:
    """Pooling [CLS] normalisé -- convention du model card bge-m3."""
    from src.config import LATENT_LABEL_EMB_MODEL
    tok = AutoTokenizer.from_pretrained(LATENT_LABEL_EMB_MODEL, local_files_only=True)
    mdl = AutoModel.from_pretrained(LATENT_LABEL_EMB_MODEL, local_files_only=True).to(DEVICE).eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], padding=True, truncation=True,
                      max_length=64, return_tensors="pt").to(DEVICE)
            cls = mdl(**enc).last_hidden_state[:, 0]
            embs.append(F.normalize(cls, p=2, dim=-1).cpu())
    del mdl
    return torch.cat(embs, dim=0)


def select_latents_by_similarity(
    query: str,
    feature_labels: dict,
    top_k: int = 100,
) -> list[int]:
    """Retourne les feature_id triés par similarité cosinus décroissante entre leur
    label et la requête (embeddings bge-m3, cf. note ci-dessus). Filtre les labels
    bruts non informatifs (F{idx}, [EXT] F{idx}) qui n'apportent aucun signal
    sémantique."""
    items = [
        (f_idx, lbl) for f_idx, lbl in feature_labels.items()
        if lbl and not re.fullmatch(r"(\[EXT\]\s*)?F\d+", lbl.strip())
    ]
    if not items:
        return []
    f_ids, labels = zip(*items)
    embeddings = _embed_bge_m3([query] + list(labels))
    query_emb, label_embs = embeddings[0], embeddings[1:]
    sims = (label_embs @ query_emb).numpy()  # embeddings déjà normalisés -> produit scalaire = cosinus
    order = np.argsort(sims)[::-1][:top_k]
    return [f_ids[i] for i in order if sims[i] > 0]


# ══════════════════════════════════════════════════════════════════════════════
# TÂCHE 3 : TARGETED CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

def targeted_clustering_by_axis(
    texts: list,
    sae_acts: torch.Tensor,
    labels: list,
    feature_labels: dict,
    axis_query: str,
    top_k_features: int = 150,
    n_clusters: int = 3,
) -> dict:
    from sklearn.cluster import SpectralClustering
    print(f"\n  [Task 3] Targeted Clustering axe : '{axis_query}'")
    matched_indices = select_latents_by_similarity(axis_query, feature_labels, top_k=top_k_features)
    if len(matched_indices) < 5:
        print("  [Task 3] Fallback : latents les plus actifs.")
        matched_indices = sae_acts.float().mean(dim=0).topk(
            min(top_k_features, sae_acts.shape[1])
        ).indices.tolist()

    sub_binarized = (sae_acts[:, matched_indices].float().detach().cpu().numpy() > 1e-6).astype(np.float32)
    if sub_binarized.sum() == 0:
        sub_binarized = (sae_acts.float().detach().cpu().numpy() > 1e-6).astype(np.float32)

    spectral = SpectralClustering(
        n_clusters=n_clusters, affinity="cosine",
        assign_labels="kmeans", random_state=SEED
    )
    cluster_labels = spectral.fit_predict(sub_binarized)
    cluster_texts = {c: [] for c in range(n_clusters)}
    for i, c in enumerate(cluster_labels):
        cluster_texts[c].append(texts[i])
    print(f"  [Task 3] Éléments par cluster: {[len(cluster_texts[c]) for c in range(n_clusters)]}")
    return {"labels": cluster_labels, "cluster_texts": cluster_texts}

# ══════════════════════════════════════════════════════════════════════════════
# TÂCHE 4 : PROPERTY-BASED RETRIEVAL (Rank-Weighted)
# ══════════════════════════════════════════════════════════════════════════════

def property_based_retrieval(
    query_string: str,
    doc_acts: torch.Tensor,
    texts: list,
    feature_labels: dict,
    temperature: float = 0.2,
    top_n_results: int = 5,
    top_k_latents: int = 100,
) -> list:
    """cf. interp_embed §4.4 : (1) latents candidats par similarité d'embedding
    label<->requête (select_latents_by_similarity, pas un matching de sous-chaîne),
    (2) score documentaire = somme pondérée à température des activations de ces
    latents, décroissante avec le rang de pertinence réel de `matched_latents`
    (pas l'ordre d'itération du dict)."""
    print(f"\n  [Task 4] Recherche implicite : '{query_string}'")
    matched_latents = select_latents_by_similarity(query_string, feature_labels, top_k=top_k_latents)
    if not matched_latents:
        print("  [Task 4] Aucun latent matché.")
        return []
    k = len(matched_latents)
    weights = torch.tensor(
        [math.exp(-(rank / k) / temperature) for rank in range(k)],
        dtype=doc_acts.dtype
    )
    scores = (doc_acts[:, matched_latents].float() * weights).sum(dim=-1).detach().cpu().numpy()
    top_idx = np.argsort(scores)[::-1][:top_n_results]
    return [(texts[i], float(scores[i])) for i in top_idx if scores[i] > 1e-6]

# ══════════════════════════════════════════════════════════════════════════════
# UMAP INTERACTIF (HDBSCAN + Plotly)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_with_umap(
    texts: list,
    sae_acts: torch.Tensor,
    labels: list,
    filename: str,
    title: str,
    token_fragments_dir: str = None,
    offset: int = 0,
    activating_tokens_map: dict = None,
    feature_labels: dict = None,
) -> dict:
    import umap
    import plotly.express as px
    from sklearn.cluster import HDBSCAN
    import textwrap

    N_DOCS = len(texts)
    sae_np = sae_acts.float().detach().cpu().numpy()
    n_inf = np.isinf(sae_np).sum()
    if n_inf:
        print(f"  [WARN] {n_inf} valeurs infinies détectées — clip appliqué.")
        sae_np = np.nan_to_num(sae_np, posinf=np.finfo(np.float32).max, neginf=0.0)
    active_mask = sae_np.max(axis=0) > 0
    sae_active = sae_np[:, active_mask]
    n_active = int(active_mask.sum())
    active_indices = np.where(active_mask)[0].tolist()
    print(f"  Features actives (UMAP) : {n_active} / {sae_acts.shape[1]}")

    if n_active == 0:
        # Corpus trop petit (smoke test) ou features trop sparses pour ce sous-ensemble :
        # aucune activation positive -> UMAP ne peut pas fitter (0 colonnes). Dégrade
        # proprement plutôt que de crasher.
        print("  [WARN] Aucune feature active — UMAP/HDBSCAN sautés pour ce sous-ensemble.")
        del sae_np, sae_active
        return {
            "coords": None, "clusters": None, "n_clusters": 0,
            "n_active": 0, "active_indices": [], "df": None,
        }

    # Sur très petits corpus (ex. fallback emails synthétiques, n<15), l'initialisation
    # spectrale par défaut de UMAP appelle scipy.sparse.linalg.eigsh avec k >= N et lève
    # un TypeError — bascule sur une init aléatoire (moins de structure globale mais
    # robuste) plutôt que de planter.
    init = "spectral" if N_DOCS >= 15 else "random"

    def _fit_umap(n_components: int) -> np.ndarray:
        reducer = umap.UMAP(
            n_components=n_components,
            metric="cosine",
            n_neighbors=min(30, max(2, N_DOCS - 1)),
            min_dist=0.1,
            random_state=SEED,
            n_jobs=1,   # random_state force déjà n_jobs=1 ; explicite → supprime le UserWarning
            init=init,
        )
        try:
            return reducer.fit_transform(sae_active)
        except TypeError as e:
            print(f"  [WARN] UMAP spectral init a échoué ({e}) — retry avec init='random'.")
            reducer = umap.UMAP(
                n_components=n_components, metric="cosine",
                n_neighbors=min(30, max(2, N_DOCS - 1)),
                min_dist=0.1, random_state=SEED, n_jobs=1, init="random",
            )
            return reducer.fit_transform(sae_active)

    coords = _fit_umap(2)  # réservé à la visualisation Plotly (x/y)
    # HDBSCAN tourne sur un embedding UMAP 10D dédié, PAS sur les coordonnées 2D de
    # visualisation : UMAP-10D domine UMAP-2D sur la stabilité inter-seed du
    # clustering (ARI 1,0 vs 0,64-1,0 à DBCV quasi identique, 0,851 vs 0,829) ;
    # PCA et l'espace cosine brut sont nettement dominés par UMAP (DBCV <= 0,275).
    # Aucune config ne récupère de structure sémantique alignée sur des labels
    # connus (AMI ~0,01-0,03 partout) — ce choix améliore la reproductibilité des
    # clusters affichés d'un run à l'autre, pas leur pertinence sémantique.
    cluster_embedding = coords if N_DOCS <= 12 else _fit_umap(min(10, N_DOCS - 2))

    # Libération immédiate des copies denses (N_DOCS × n_active en fp32, potentiellement
    # plusieurs Go à width 262k) : UMAP a fini, seuls `coords`/`cluster_embedding` et
    # `sae_acts` (torch, partagé avec l'appelant) restent nécessaires.
    del sae_np, sae_active
    _trim_host_memory()

    min_cs = max(2, N_DOCS // 15)
    clusterer = HDBSCAN(min_cluster_size=min_cs, min_samples=max(1, min_cs // 2), copy=True)
    clusters = clusterer.fit_predict(cluster_embedding)

    df = pd.DataFrame({
        "x": coords[:, 0], "y": coords[:, 1],
        "cluster_raw": clusters,
        "label": labels if (labels and len(labels) == N_DOCS) else ["Unknown"] * N_DOCS,
        "doc_idx": np.arange(N_DOCS),
    }).sort_values("cluster_raw").reset_index(drop=True)
    df["cluster_id"] = df["cluster_raw"].apply(lambda c: f"Cluster {c}" if c != -1 else "Bruit (-1)")

    cluster_signatures = {}
    for c in df["cluster_raw"].unique():
        if c == -1:
            cluster_signatures[c] = "Bruit sémantique"
            continue
        orig_indices = df.loc[df["cluster_raw"] == c, "doc_idx"].values
        mean_acts = sae_acts[torch.from_numpy(orig_indices)].float().mean(dim=0)
        top_vals, top_ids = mean_acts.topk(min(3, mean_acts.shape[0]))
        sig = " | ".join(
            f"{feature_labels.get(f_idx, f'F{f_idx}')} (µ={v:.1f})"
            for v, f_idx in zip(top_vals.tolist(), top_ids.tolist())
            if v > 1e-6
        )
        cluster_signatures[c] = sig or "Aucune signature"

    # Une couleur par rang de feature (top-1, top-2, top-3, ...) pour distinguer
    # visuellement plusieurs mots-déclencheurs différents dans le même hover.
    FEATURE_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]

    custom_hover, top_feats_html, sig_col = [], [], []
    for _, row in df.iterrows():
        i = int(row["doc_idx"])
        c_raw = int(row["cluster_raw"])
        sig_col.append(cluster_signatures.get(c_raw, ""))

        r_acts = sae_acts[i]
        top_vals, top_ids = r_acts.topk(min(3, r_acts.shape[0]))
        feats_html = []

        td = None
        if token_fragments_dir:
            if fragment_exists(token_fragments_dir, int(i + offset)):
                td = load_fragment(token_fragments_dir, int(i + offset))

        for j in range(len(top_ids)):
            v = top_vals[j].item()
            if v <= 1e-6: 
                break
            f_idx = top_ids[j].item()
            f_label = feature_labels.get(f_idx, f"F{f_idx}") if feature_labels else f"F{f_idx}"
            tok_str = ""
            
            if td:
                acts_arr = feature_column(td, f_idx)
                high = np.where(acts_arr > acts_arr.max() * 0.65)[0]
                detected = list(dict.fromkeys([
                    td["token_strings"][t].replace("Ġ", " ").replace("▁", " ").strip()
                    for t in high if len(td["token_strings"][t].strip()) > 1
                ]))[:3]
                if detected: 
                    tok_str = f" <i>«{', '.join(detected)}»</i>"
            elif activating_tokens_map and i in activating_tokens_map:
                toks = activating_tokens_map[i].get(f_idx, [])
                if toks: 
                    tok_str = f" <i>«{toks[0][0].strip()}»</i>"
                    
            intensity = min(5, max(1, int((v / 15.0) * 5)))
            bar = f"<span style='color:#00cc96;'>{'█'*intensity}{'▒'*(5-intensity)}</span>"
            feats_html.append(f"<b>{f_label}</b> ({v:.1f}) {bar}{tok_str}")
            
        top_feats_html.append("<br>".join(feats_html) or "<i>Aucune feature active</i>")

        # Contexte : un mot différent par feature top-3, chacun surligné dans sa
        # propre couleur, et dédupliqué sur le mot cible (évite de toujours
        # retomber sur le même mot — ex. la salutation en tête de mail — quand
        # plusieurs features partagent leur token le plus activant).
        context_lines = []
        if td:
            seen_words = set()
            for j in range(len(top_ids)):
                v = top_vals[j].item()
                if v <= 1e-6:
                    break
                f_idx = top_ids[j].item()
                acts_arr = feature_column(td, f_idx)
                tgt_idx = int(acts_arr.argmax())
                if acts_arr[tgt_idx] <= 1e-6:
                    continue
                ctx = extract_causal_context(td["token_strings"], tgt_idx)
                m = re.search(r"<<(.+?)>>", ctx)
                word_key = m.group(1).strip().lower() if m else ctx.strip().lower()
                if word_key in seen_words:
                    continue
                seen_words.add(word_key)

                color = FEATURE_COLORS[j % len(FEATURE_COLORS)]
                f_label = feature_labels.get(f_idx, f"F{f_idx}") if feature_labels else f"F{f_idx}"
                colored = ctx.replace("<<", f"<b style='color:{color};background:#ffeeba'>").replace(">>", "</b>")
                wrapped = "<br>".join(textwrap.wrap(colored, width=80))
                context_lines.append(
                    f"<span style='font-size:11px;color:{color}'>[{f_label}]</span> {wrapped}"
                )

        if context_lines:
            final_html = "<br><br>".join(context_lines)
        else:
            wrapped = "<br>".join(textwrap.wrap(texts[i][:400], width=80))
            final_html = wrapped.replace("<<", "<b style='color:#d62728;background:#ffcccc'>").replace(">>", "</b>")
        custom_hover.append(final_html)

    df["text_preview"] = custom_hover
    df["top_features"] = top_feats_html
    df["cluster_signature"] = sig_col

    out_html = os.path.join(SAVE_DIR, filename)
    try:
        df.to_parquet(out_html.replace(".html", "_coords.parquet"), index=False)
    except Exception:
        df.to_csv(out_html.replace(".html", "_coords.csv"), index=False)

    # IMPORTANT : custom_data doit être passé ICI, dans px.scatter, et non via
    # fig.update_traces(customdata=...) après coup. px.scatter découpe déjà x/y
    # par cluster en plusieurs traces distinctes (une par couleur) ; si on
    # rattache le customdata *après*, via update_traces sans sélecteur de
    # trace, Plotly rediffuse le même array complet (toutes lignes de df) sur
    # CHAQUE trace, alors que chaque trace n'a qu'un sous-ensemble de points.
    # Le hover affichait alors le texte/label/signature d'un document au
    # hasard (mauvais index) et non celui du point réellement survolé.
    fig = px.scatter(
        df, x="x", y="y", color="cluster_id",
        category_orders={"cluster_id": sorted(df["cluster_id"].unique())},
        custom_data=["text_preview", "top_features", "cluster_id", "label", "cluster_signature"],
    )
    fig.update_traces(
        marker=dict(size=7, opacity=0.8),
        hovertemplate=(
            "<b>%{customdata[2]}</b> (label: %{customdata[3]})<br>"
            "<span style='color:#1f77b4'><b>Signature cluster :</b> %{customdata[4]}</span><br><br>"
            "<b>Top Features :</b><br>%{customdata[1]}<br><br>"
            "<b>Contexte :</b><br>%{customdata[0]}<extra></extra>"
        ),
    )
    fig.update_layout(
        title=f"{title}<br><sub>{N_DOCS} docs | {n_active} features actives</sub>",
        width=1400, height=900,
        hoverlabel=dict(bgcolor="white", font_size=12, align="left"),
        margin=dict(l=50, r=50, t=60, b=50),
    )
    fig.write_html(out_html)
    print(f"  [+] UMAP HTML : {out_html}")
    return {
        "coords": coords, "clusters": df["cluster_raw"].values,
        "n_clusters": len([c for c in df["cluster_raw"].unique() if c != -1]),
        "n_active": n_active, "active_indices": active_indices, "df": df,
    }

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE 1 — GEMMA-3 + FROZEN CORE SAE
# ══════════════════════════════════════════════════════════════════════════════

def run_llm_max_pool_pipeline(
    train_texts: list,
    train_labels: list,
    test_texts: list,
    test_labels: list,
    diff_texts: list = None,
    diff_labels: list = None,
    volume_filler_texts: list = None,
    train_groups: list = None,
) -> dict:
    """train_texts/test_texts : corpus principal (emails+augmentés), utilisé pour
    l'entraînement du SAE et les métriques. diff_texts/diff_labels : corpus
    secondaire (energy/sports/support), encodé post-hoc UNIQUEMENT pour la
    démonstration de diffing cross-domaine -- jamais utilisé pour entraîner.
    volume_filler_texts (optionnel, défaut None -> comportement 100% inchangé) :
    corpus supplémentaire (FineWeb-2 FR, cf. `sample_fineweb2_chunks`) ajouté
    UNIQUEMENT au réservoir de tokens résiduels (ablation de volume, SAE Boost,
    arXiv:2507.12990, §18). Volontairement PAS ajouté à `train_texts` lui-même :
    la sélection des features à labelliser (`feature_selection_by_magnitude`,
    `range(n_train)`) et la sonde de classification email restent calculées sur
    les emails+augmentés seuls -- sinon `volume_filler_texts` ferait partie du
    corpus "principal" et diluerait la sélection de features vers du contenu
    générique. train_groups (optionnel, défaut None -> comportement inchangé) :
    parent_id (mail d'origine) de chaque `train_texts[i]`, même longueur/ordre --
    permet une CV group-aware pour `clf_acc_email_axes` (`RESULTS_TESTS.md` §57)."""
    print("\n" + "=" * 70)
    print(" PIPELINE 1 : GEMMA-3 → MAX-POOL SAE ACTS")
    print("=" * 70)

    diff_texts = diff_texts or []
    diff_labels = diff_labels or []
    volume_filler_texts = volume_filler_texts or []
    all_texts = train_texts + volume_filler_texts + test_texts + diff_texts

    pretrained_sae = load_pretrained_sae()
    pretrained_sae = pretrained_sae.to(DEVICE).to(TORCH_DTYPE).eval()
    pretrained_sae.requires_grad_(False)
    d_core = pretrained_sae.cfg.d_sae

    d_total_expected = d_core + D_EXTRA if USE_FROZEN_CORE else d_core

    cache_acts_path      = os.path.join(CACHE_DIR, "p1_all_doc_acts.pt")
    # Réservoir memmap disque (pas un .pt chargé intégralement en RAM, cf.
    # open_mmap_reservoir ci-dessus) : le fichier lui-même EST le cache, plus
    # de copie séparée via torch.save. Le nombre de lignes réellement remplies
    # (≤ N_TOKENS_EXTRA_TRAIN si le corpus est plus petit) vit dans le sidecar
    # JSON, car le fichier mmap est toujours dimensionné à N_TOKENS_EXTRA_TRAIN.
    cache_residuals_path      = os.path.join(CACHE_DIR, "p1_raw_residuals.memmap")
    cache_residuals_meta_path = cache_residuals_path + ".meta.json"
    token_fragments_dir  = os.path.join(CACHE_DIR, "p1_token_fragments")

    n_train = len(train_texts)
    n_filler = len(volume_filler_texts)
    n_test  = len(test_texts)
    # Filler jamais fragmenté (allègement extraction, AUDIT_SAE_2026-08.md §2.2/§2.5) :
    # seul son résidu brut compte, pour le réservoir -- ni encodage core ni fragment
    # disque. n_fragmented_expected exclut donc la plage filler, contrairement à
    # len(all_texts).
    n_fragmented_expected = n_train + n_test + len(diff_texts)

    _need_extraction = True
    _need_residuals = USE_FROZEN_CORE and not os.path.exists(cache_residuals_meta_path)

    if os.path.exists(cache_acts_path) and os.path.exists(token_fragments_dir):
        fragment_ids = list_fragment_ids(token_fragments_dir)
        if len(fragment_ids) == n_fragmented_expected:
            print("  [P1] Restauration du cache (activations documents et fragments disques)...")
            all_doc_sae_acts = torch.load(cache_acts_path, map_location="cpu", weights_only=True)
            _need_extraction = False
            
            if _need_residuals:
                # Le filler n'a plus de fragment (allègement extraction ci-dessus) :
                # cette reconstruction ne peut plus porter que sur le train -- un
                # réservoir reconstruit ainsi NE contient PAS la contribution du
                # filler qu'un run frais aurait collectée. Cas déjà marginal avant ce
                # correctif (fragments complets mais résidus seuls manquants,
                # typiquement une suppression manuelle) ; --retrain (supprimer
                # token_fragments_dir) reste la seule voie pour un réservoir
                # filler-inclusif à l'identique d'un run frais.
                if n_filler > 0:
                    print(f"  [P1] ATTENTION : reconstruction de raw_residuals depuis les fragments "
                          f"locaux limitée au train ({n_train} docs) -- le filler ({n_filler} docs) "
                          "n'a plus de fragment (allègement extraction), sa contribution au "
                          "réservoir est absente de cette reconstruction.")
                else:
                    print("  [P1] Reconstruction de raw_residuals depuis les fragments de tokens locaux...")
                # Buffer memmap disque (cf. open_mmap_reservoir) : les écritures
                # vont directement sur disque par page, jamais de RAM anonyme à
                # N_TOKENS_EXTRA_TRAIN*d_in*2 octets.
                _residuals_buf = open_mmap_reservoir(
                    cache_residuals_path, N_TOKENS_EXTRA_TRAIN, pretrained_sae.cfg.d_in, TORCH_DTYPE)
                n_collected = 0
                for _fid in fragment_ids[:n_train]:
                    frag = load_fragment(token_fragments_dir, _fid)
                    if "raw_acts" in frag:
                        chunk = frag["raw_acts"]
                        take = min(N_TOKENS_EXTRA_TRAIN - n_collected, chunk.shape[0])
                        _residuals_buf[n_collected:n_collected + take] = chunk[:take]
                        n_collected += take
                    if n_collected >= N_TOKENS_EXTRA_TRAIN:
                        break
                if n_collected > 0:
                    raw_residuals = _residuals_buf[:n_collected]
                    with open(cache_residuals_meta_path, "w") as _f:
                        json.dump({"n_rows": n_collected, "d_in": pretrained_sae.cfg.d_in}, _f)
                    print(f"  [P1] Résidus bruts d'apprentissage réinitialisés : {raw_residuals.shape}")
                    _need_residuals = False
                del _residuals_buf
        else:
            print("  [P1] Fragments disques incomplets.")
            _need_extraction = True

    # Reprise (R1, AUDIT_SAE_2026-08.md §2.3/§4.3) : le critère est "quel est le
    # prochain document non traité", jamais "le run est-il complet". _resume_from>0
    # signifie qu'un checkpoint valide existe et couvre déjà les documents
    # [0, _resume_from) -- fragments ET compteurs de réservoir cohérents entre eux
    # (le checkpoint n'avance qu'après que les deux soient écrits pour un lot
    # entier, cf. écriture du checkpoint dans la boucle ci-dessous). Les fragments
    # au-delà de _resume_from, s'il en existe (run précédent tué en plein lot),
    # sont volontairement ignorés/réécrits -- on ne leur fait pas confiance.
    _resume_from, _resume_n_seen, _resume_n_collected = 0, 0, 0
    if _need_extraction:
        _progress = _read_extraction_progress(CACHE_DIR)
        if _progress is not None and 0 < _progress["next_doc_idx"] < len(all_texts):
            _resume_from = _progress["next_doc_idx"]
            _resume_n_seen = _progress["n_residuals_seen"]
            _resume_n_collected = _progress["n_residuals_collected"]
            print(f"  [P1] Reprise au document {_resume_from}/{len(all_texts)} "
                  f"(checkpoint précédent, {_resume_n_collected} résidus déjà collectés).")
        elif _progress is not None:
            print("  [P1] Checkpoint présent mais incohérent avec l'état des fragments "
                  "-- re-extraction complète forcée plutôt que de faire confiance à un "
                  "état ambigu.")

    if _need_extraction:
        print(f"  [P1] Extraction activations Gemma-3 ({MODEL_ID}, layer {LAYER}, hook {HOOK_TYPE})...")
        os.makedirs(token_fragments_dir, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, token=HF_TOKEN, trust_remote_code=True, local_files_only=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        llm = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=TORCH_DTYPE, device_map=DEVICE,
            token=HF_TOKEN, trust_remote_code=True, local_files_only=True
        ).eval()

        # G1 (AUDIT_SAE_2026-08.md §2.2) : troncature des blocs décodeur
        # au-delà de LAYER, VÉRIFIÉE sur GPU avant déploiement
        # (scripts/audit_2026_08_layer_truncation_equivalence_and_speedup.py) :
        # - V1 (troncature `layers[:LAYER]` + `output_hidden_states=True` +
        #   `hidden_states[LAYER]`, job 44536) a ÉCHOUÉ : torch.equal=False,
        #   écart max ~6e5. Cause confirmée en lisant
        #   transformers/utils/output_capturing.py : `output_hidden_states=True`
        #   passe par un mécanisme générique (`_can_record_outputs`) qui
        #   remplace INCONDITIONNELLEMENT la DERNIÈRE entrée de `hidden_states`
        #   par `last_hidden_state` (post-RMSNorm final,
        #   `tie_last_hidden_states=True`) -- invisible sur le modèle complet
        #   (LAYER n'est jamais la dernière entrée sur 49), silencieusement
        #   faux une fois LAYER devenu la dernière entrée par troncature.
        # - V2 (ce code, job 44540) : hook DIRECT sur `layers[LAYER-1]`, jamais
        #   `output_hidden_states` -- ne passe jamais par ce mécanisme.
        #   Vérifié : torch.equal=True, écart max = 0.0 bit-à-bit, 1,98× sur
        #   le débit, 13 Go de VRAM en moins (28,0 -> 15,0 Go).
        # attn_out/mlp_out utilisaient déjà un hook direct (jamais affectés par
        # ce piège) mais laissaient tourner les blocs après LAYER pour rien --
        # même troncature appliquée ici, mécanisme de hook inchangé pour eux.
        # Points de hook vérifiés empiriquement sur les config.json
        # GemmaScope-2 réels :
        #   resid_post : sortie de layers[LAYER-1] (= hidden_states[LAYER] du
        #                modèle complet, non tronqué)
        #   attn_out   : entrée de self_attn.o_proj (pré-projection de sortie)
        #   mlp_out    : sortie de post_feedforward_layernorm (après le MLP, avant l'add résiduel)
        _hook_capture = {}
        _hook_handle = None
        if HOOK_TYPE == "resid_post":
            def _capture_resid_post(module, args, output):
                _hook_capture["acts"] = output
            _n_layers_needed = LAYER
            _hook_handle = llm.model.language_model.layers[LAYER - 1].register_forward_hook(
                _capture_resid_post)
        elif HOOK_TYPE == "attn_out":
            def _capture_attn_in(module, args, kwargs):
                _hook_capture["acts"] = args[0] if args else kwargs["input"]
            _n_layers_needed = LAYER + 1
            _hook_handle = llm.model.language_model.layers[LAYER].self_attn.o_proj.register_forward_pre_hook(
                _capture_attn_in, with_kwargs=True)
        elif HOOK_TYPE == "mlp_out":
            def _capture_mlp_out(module, args, output):
                _hook_capture["acts"] = output
            _n_layers_needed = LAYER + 1
            _hook_handle = llm.model.language_model.layers[LAYER].post_feedforward_layernorm.register_forward_hook(
                _capture_mlp_out)
        else:
            raise ValueError(f"HOOK_TYPE={HOOK_TYPE!r} non supporté (resid_post/attn_out/mlp_out).")
        llm.model.language_model.layers = llm.model.language_model.layers[:_n_layers_needed]

        # Reprise : reconstruit les vecteurs déjà traités [0, _resume_from) depuis
        # leurs fragments (doc_maxpool, CPU, O(nnz) -- pas de GPU, quasi-instantané
        # même pour des dizaines de milliers de documents) plutôt que de les
        # recalculer sur GPU. Compteurs de réservoir repris à leur valeur persistée.
        # Filler (allègement extraction, §2.2) : aucun fragment -> placeholder,
        # jamais lu en aval (mêmes garanties que build_reencode_targets).
        if _resume_from > 0:
            print(f"  [P1] Reconstruction de {_resume_from} vecteurs déjà extraits depuis les fragments...")
            all_doc_sae_acts = []
            for _di in tqdm(range(_resume_from), desc="Reprise (fragments->vecteurs)"):
                if is_filler_document(_di, n_train, n_filler):
                    all_doc_sae_acts.append(torch.zeros(d_total_expected, dtype=TORCH_DTYPE))
                else:
                    all_doc_sae_acts.append(doc_maxpool(load_fragment(token_fragments_dir, _di)))
        else:
            all_doc_sae_acts = []
        n_residuals_collected = _resume_n_collected
        n_residuals_seen = _resume_n_seen      # total de tokens train vus (dénominateur réservoir)
        # Buffer préalloué UNE SEULE FOIS à la taille finale (pas de liste de
        # chunks + torch.cat, qui doublerait transitoirement le pic mémoire).
        # Rempli directement par tranches (phase 1) puis par écriture indexée
        # (phase 2, réservoir de Vitter) -- jamais plus d'une copie en RAM.
        # d_in du SAE préentraîné, pas D_MODEL : la dimension du residual stream
        # ne vaut que pour resid_post/mlp_out -- attn_out capte l'entrée de
        # o_proj, en amont de la projection multi-head vers hidden_size, donc
        # une dimension différente (4096 vs D_MODEL=3840 pour gemma-3-12b-it).
        reservoir = (open_mmap_reservoir(cache_residuals_path, N_TOKENS_EXTRA_TRAIN,
                                          pretrained_sae.cfg.d_in, TORCH_DTYPE)
                     if USE_FROZEN_CORE else None)

        _GracefulShutdown.install()
        _last_checkpoint_doc = _resume_from
        _early_exit = False
        # Écriture de fragments en arrière-plan (audit perf G2, AUDIT_SAE_2026-08.md
        # §2.2) : le GPU n'attend plus le disque pour continuer -- flush() OBLIGATOIRE
        # avant tout checkpoint de reprise qui avance next_doc_idx (cf. docstring
        # AsyncFragmentWriter), sinon un crash pourrait laisser un checkpoint plus
        # avancé que les fragments réellement sur disque.
        _fragment_writer = AsyncFragmentWriter()
        with torch.no_grad():
            for i in tqdm(range(_resume_from, len(all_texts), EXTRACTION_BATCH_SIZE), desc="Extraction P1"):
                batch = all_texts[i: i + EXTRACTION_BATCH_SIZE]
                inputs = tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=512,
                ).to(DEVICE)
                # logits_to_keep=1 : seul le hook nous intéresse ici ; sans ça, le
                # forward calcule par défaut les logits sur TOUTE la séquence et le
                # vocabulaire Gemma-3 (~262k), un gaspillage mémoire GPU inutile qui
                # peut mener à l'OOM CUDA à grande échelle. output_hidden_states
                # jamais utilisé (les trois HOOK_TYPE passent par _hook_capture) --
                # cf. commentaire de mise en place des hooks ci-dessus pour la
                # raison (piège tie_last_hidden_states de output_hidden_states).
                llm(**inputs, logits_to_keep=1)
                acts_raw = _hook_capture["acts"].detach().to(TORCH_DTYPE)
                assert acts_raw.shape[-1] == pretrained_sae.cfg.d_in, (
                    f"HOOK_TYPE={HOOK_TYPE} : shape captée {acts_raw.shape} != "
                    f"d_in SAE préentraîné={pretrained_sae.cfg.d_in} "
                    "-- mauvais point de hook, à corriger avant de faire confiance au run."
                )

                acts = acts_raw
                # Masquage (special tokens + skip-first + σ-clip) : implémentation
                # unique dans src/analysis/activations. σ-clip intra-batch (stats
                # sur B docs) plutôt qu'intra-doc, cohérent avec l'unimodalité des
                # normes observée empiriquement.
                keep_bt = valid_token_mask(
                    inputs["input_ids"], inputs["attention_mask"],
                    tokenizer, skip_first_content_token=True,
                )
                keep_bt = norm_outlier_mask(acts, keep_bt, sigma_clip=4.0)
                for b in range(acts.shape[0]):
                    doc_global_idx = i + b
                    keep = keep_bt[b]
                    if keep.sum() == 0:   # garde-fou doc vidé par le masquage
                        keep = inputs["attention_mask"][b].bool()
                    filtered = acts[b, keep]

                    # Allègement filler (AUDIT_SAE_2026-08.md §2.2/§2.5) : le filler ne
                    # sert qu'à nourrir le réservoir ci-dessous (volume de tokens pour
                    # entraîner SAEBoostResidualSAE) -- son rôle en aval s'arrête là,
                    # aucun consommateur ne relit jamais son fragment ni sa case
                    # d'all_doc_sae_acts (build_reencode_targets l'exclut déjà du
                    # ré-encodage, §2.5). Ni encodage core (pretrained_sae.encode,
                    # coût GPU dominant par document) ni écriture de fragment (I/O
                    # disque) pour ces documents -- seul `filtered` (résidu brut)
                    # compte, déjà disponible pour la mise à jour du réservoir.
                    is_filler_doc = is_filler_document(doc_global_idx, n_train, n_filler)
                    if not is_filler_doc:
                        filtered_ids = inputs["input_ids"][b, keep]
                        token_sae_acts = pretrained_sae.encode(filtered)

                        # Stockage SPARSE (CSR) : ~250 Ko/doc au lieu de ~400 Mo dense a width 262k.
                        d_total_frag = d_core + D_EXTRA if USE_FROZEN_CORE else d_core
                        if USE_FROZEN_CORE:
                            doc_sae_vec = torch.cat([
                                token_sae_acts.max(dim=0).values,
                                torch.zeros(D_EXTRA, dtype=token_sae_acts.dtype, device=token_sae_acts.device),
                            ])
                        else:
                            doc_sae_vec = token_sae_acts.max(dim=0).values

                        save_fragment(
                            token_fragments_dir, doc_global_idx,
                            token_strings=tokenizer.convert_ids_to_tokens(filtered_ids.tolist()),
                            acts_dense=token_sae_acts,   # nnz core uniquement, shape logique d_total_frag
                            d_total=d_total_frag,
                            raw_acts=filtered,
                            writer=_fragment_writer,
                        )
                        all_doc_sae_acts.append(doc_sae_vec.cpu())
                    else:
                        # Placeholder bon marché : garde all_doc_sae_acts aligné sur
                        # doc_global_idx (liste construite par append, dans l'ordre) --
                        # jamais lu en aval (cf. slicing train/test/diff plus bas).
                        all_doc_sae_acts.append(torch.zeros(d_total_expected, dtype=TORCH_DTYPE))

                    if USE_FROZEN_CORE and doc_global_idx < n_train + n_filler:
                        # Réservoir (Vitter, Algorithm R) : échantillon uniforme
                        # sur TOUS les tokens du split train, au lieu des seuls
                        # premiers documents. Phase 1 : remplissage séquentiel du
                        # buffer préalloué ; phase 2 : le m-ième token vu remplace
                        # reservoir[j], j ~ U[0, m), ssi j < N. Vectorisé par chunk
                        # (les collisions intra-chunk sont résolues last-write-wins,
                        # biais négligeable pour chunk << N).
                        x_new = filtered.cpu()
                        if n_residuals_collected < N_TOKENS_EXTRA_TRAIN:
                            take = min(N_TOKENS_EXTRA_TRAIN - n_residuals_collected, x_new.shape[0])
                            reservoir[n_residuals_collected:n_residuals_collected + take] = x_new[:take]
                            n_residuals_collected += take
                            n_residuals_seen += take
                            x_new = x_new[take:]
                        if x_new.shape[0] > 0:
                            m = n_residuals_seen + torch.arange(1, x_new.shape[0] + 1)
                            j = (torch.rand(x_new.shape[0]) * m).long()
                            hit = j < N_TOKENS_EXTRA_TRAIN
                            reservoir[j[hit]] = x_new[hit]
                            n_residuals_seen += x_new.shape[0]

                # Checkpoint (R1) : tous les documents de ce lot sont désormais
                # traités (fragment + réservoir) -- point cohérent où avancer
                # next_doc_idx. Périodique (EXTRACTION_CHECKPOINT_INTERVAL) pour
                # borner le travail reperdu sur un SIGKILL sans handler, ET à
                # chaque lot si un signal de coupure a été reçu (arrêt imminent :
                # mieux vaut un léger surcoût d'écriture qu'un lot de travail
                # perdu juste avant le SIGKILL).
                _doc_done_through = i + acts.shape[0]
                if (_doc_done_through - _last_checkpoint_doc >= EXTRACTION_CHECKPOINT_INTERVAL
                        or _GracefulShutdown.requested):
                    # flush() AVANT d'avancer le checkpoint -- sinon celui-ci pourrait
                    # prétendre "traité" un document dont le fragment n'a pas encore
                    # atteint le disque (cf. docstring AsyncFragmentWriter).
                    _fragment_writer.flush()
                    _write_extraction_progress(CACHE_DIR, _doc_done_through,
                                                n_residuals_seen, n_residuals_collected)
                    _last_checkpoint_doc = _doc_done_through
                if _GracefulShutdown.requested:
                    print(f"  [P1] Signal de coupure reçu -- checkpoint écrit au document "
                          f"{_doc_done_through}/{len(all_texts)}, arrêt propre "
                          "(reprise au prochain sbatch).")
                    _early_exit = True
                    break

        if _early_exit:
            _fragment_writer.close()
            if _hook_handle is not None:
                _hook_handle.remove()
            sys.exit(0)
        _fragment_writer.close()

        # Extraction complète : le checkpoint n'a plus lieu d'être, la prochaine
        # invocation doit se fier au test fragments-complets standard (ligne
        # ~845) plutôt qu'à un checkpoint devenu obsolète s'il reste sur disque.
        _clear_checkpoint(_extraction_progress_path(CACHE_DIR))

        all_doc_sae_acts = torch.stack(all_doc_sae_acts)
        torch.save(all_doc_sae_acts, cache_acts_path)

        if USE_FROZEN_CORE and reservoir is not None:
            # Corpus plus petit que N_TOKENS_EXTRA_TRAIN : le buffer préalloué n'a
            # été rempli que partiellement, tronquer au nombre réel de tokens vus.
            # Le fichier memmap EST déjà le cache (écriture directe pendant la
            # boucle d'extraction) : il ne reste qu'à persister le nombre réel
            # de lignes valides dans le sidecar JSON.
            n_rows_final = min(n_residuals_collected, N_TOKENS_EXTRA_TRAIN)
            raw_residuals = reservoir[:n_rows_final]
            with open(cache_residuals_meta_path, "w") as _f:
                json.dump({"n_rows": n_rows_final, "d_in": pretrained_sae.cfg.d_in}, _f)
            print(f"  [P1] Résidus bruts d'apprentissage enregistrés : {raw_residuals.shape}")
            _need_residuals = False
            
        if _hook_handle is not None:
            _hook_handle.remove()
        del llm, tokenizer
        _trim_host_memory()

    d_total = d_core
    active_sae = pretrained_sae

    if USE_FROZEN_CORE:
        frozen_core_path = os.path.join(SAVE_DIR, f"p1_frozen_core_d{D_EXTRA}_k{K_EXTRA}.pt")
        if os.path.exists(frozen_core_path):
            print(f"  [P1] Chargement FrozenCoreResidualSAE : {frozen_core_path}")
            # Pas de .to(TORCH_DTYPE) ici : core_sae (pretrained_sae) est déjà
            # casté ; la branche "extra" (W_dec_extra/W_enc_extra/b_enc_extra/
            # threshold/input_scale) doit rester fp32 -- frozen_core.py la traite
            # explicitement en fp32 partout (.float() systématique). Un cast
            # module-wide ici la bascule en bf16/fp16 et casse le backward.
            if SANITY_CHECK_FROZEN_DECODER:
                ext_sae = FrozenDecoderExtendedSAE(pretrained_sae, d_extra=D_EXTRA, k_extra=K_EXTRA).to(DEVICE)
            else:
                ext_sae = SAEBoostResidualSAE(pretrained_sae, d_extra=D_EXTRA, k_extra=K_EXTRA).to(DEVICE)
            ckpt = torch.load(frozen_core_path, map_location=DEVICE, weights_only=False)
            # L'encodeur extra lit désormais x, pas le résidu e (SAE Boost §3.1,
            # correctif AUDIT_SAE_2026-08.md §1.3) -- un checkpoint entraîné avant
            # ce correctif a des poids W_enc_extra/b_enc_extra/encoder_input_scale
            # ajustés pour un INPUT DIFFÉRENT (e, dont l'échelle et la distribution
            # n'ont rien à voir avec x). `load_state_dict(strict=False)` les
            # chargerait quand même silencieusement (mêmes noms/formes de
            # paramètres, même piège de clé de cache que CLAUDE.md documente déjà
            # pour ce loader) -- refus explicite plutôt qu'un résultat
            # silencieusement faux.
            ckpt_encoder_input = ckpt.get("config", {}).get("encoder_input")
            if ckpt_encoder_input != "x":
                raise RuntimeError(
                    f"{frozen_core_path} a été entraîné avant le correctif SAE Boost "
                    f"(encoder_input={ckpt_encoder_input!r}, attendu 'x') -- l'encodeur "
                    "extra lisait alors le résidu e, pas x : les poids ne sont pas "
                    "compatibles avec le forward actuel. Supprimer ce fichier et "
                    "réentraîner (--retrain), pas de chargement partiel possible ici."
                )
            missing, unexpected = ext_sae.load_state_dict(ckpt["state_dict"], strict=False)
            if missing or unexpected:
                print(f"  [P1] Checkpoint sans certains buffers ({missing}) — "
                      f"input_scale retombe à 1.0 par défaut, probablement entraîné "
                      f"sans AuxK. Préférez --retrain (supprimer {frozen_core_path}) "
                      f"plutôt que de le recharger tel quel.")
        else:
            if os.path.exists(cache_residuals_meta_path):
                with open(cache_residuals_meta_path) as _f:
                    _meta = json.load(_f)
                # Réouverture mmap (pas de torch.load : la lecture reste
                # paginée à la demande, jamais le tenseur entier en RAM).
                raw_residuals = open_mmap_reservoir(
                    cache_residuals_path, N_TOKENS_EXTRA_TRAIN, _meta["d_in"], TORCH_DTYPE
                )[:_meta["n_rows"]]
            else:
                print("  [P1] WARN : résidus introuvables, FrozenCore désactivé.")
                raw_residuals = None

            if raw_residuals is not None:
                print(f"  [P1] Entraînement SAEBoostResidualSAE sur {len(raw_residuals)} tokens résidus...")
                with torch.no_grad():
                    sample = raw_residuals[:min(8192, len(raw_residuals))].to(DEVICE).to(TORCH_DTYPE)
                    core_acts = pretrained_sae.encode(sample)
                    core_out  = pretrained_sae.decode(core_acts)
                    domain_residuals_cpu = (sample - core_out).cpu().float()
                    # x lui-même (appairé aux mêmes tokens que domain_residuals_cpu) : requis
                    # pour calibrer encoder_input_scale, l'encodeur extra lisant x et non plus
                    # le résidu (SAE Boost §3.1, cf. frozen_core.py::SAEBoostResidualSAE).
                    domain_inputs_cpu = sample.cpu().float()
                    del sample, core_acts, core_out
                    gc.collect(); torch.cuda.empty_cache()

                # Idem : pas de .to(TORCH_DTYPE) module-wide (cf. commentaire ci-dessus) —
                # la branche "extra" reste fp32, seul core_sae (déjà casté) est en TORCH_DTYPE.
                # SANITY_CHECK_FROZEN_DECODER (Korznikov et al. 2026) : décodeur ALÉATOIRE
                # figé, pas d'init PCA sur le résidu (affaiblirait le test, cf. frozen_core.py) —
                # domain_residuals délibérément PAS transmis dans cette branche. domain_inputs
                # (x) l'est : ça ne calibre qu'un scalaire d'échelle pour l'encodeur (toujours
                # entraîné normalement dans cette baseline), pas une direction data-informée —
                # ne contredit pas le principe du sanity-check.
                if SANITY_CHECK_FROZEN_DECODER:
                    ext_sae = FrozenDecoderExtendedSAE(
                        pretrained_sae, d_extra=D_EXTRA, k_extra=K_EXTRA,
                        domain_inputs=domain_inputs_cpu,
                    ).to(DEVICE)
                else:
                    ext_sae = SAEBoostResidualSAE(
                        pretrained_sae, d_extra=D_EXTRA, k_extra=K_EXTRA,
                        domain_residuals=domain_residuals_cpu, domain_inputs=domain_inputs_cpu,
                    ).to(DEVICE)

                from sae_shared import load_or_train_extended_sae as load_or_train
                ext_sae, history_ext = load_or_train(
                    model=ext_sae, model_name="p1_extended_sae",
                    acts_train=raw_residuals,
                    epochs=EPOCHS_EXTRA, lr=LR_EXTRA,
                    save_dir=SAVE_DIR, device=DEVICE,
                )
                ckpt = {"state_dict": {k: v.cpu() for k, v in ext_sae.state_dict().items()},
                        "config": {"d_extra": D_EXTRA, "k_extra": K_EXTRA, "layer": LAYER,
                                   "encoder_input": "x"}}
                torch.save(ckpt, frozen_core_path)
                print(f"  [P1] SAEBoostResidualSAE sauvegardé : {frozen_core_path}")
                del raw_residuals, domain_residuals_cpu, domain_inputs_cpu
                gc.collect(); torch.cuda.empty_cache()
            else:
                ext_sae = None

        if ext_sae is not None:
            cache_acts_ext = os.path.join(CACHE_DIR, f"p1_all_doc_acts_ext_d{D_EXTRA}.pt")
            if os.path.exists(cache_acts_ext) and not _need_extraction:
                all_doc_sae_acts = torch.load(cache_acts_ext, map_location="cpu", weights_only=True)
                if all_doc_sae_acts.shape[0] != len(all_texts):   # garde-fou explicite
                    all_doc_sae_acts = None
            else:
                all_doc_sae_acts = None
            if all_doc_sae_acts is None:
                print("  [P1] Re-encodage SAEBoostResidualSAE (fragments sparse, O(nnz))...")
                ext_sae.eval()
                pretrained_sae._W_dec_fp32 = pretrained_sae.W_dec.to(torch.float32)
                pretrained_sae._b_dec_fp32 = pretrained_sae.b_dec.to(torch.float32)
                all_doc_sae_acts = torch.empty(
                    (len(all_texts), d_core + D_EXTRA),
                    dtype=torch.float32,
                )

                # Filler jamais ré-encodé (audit perf, section filler) : train_doc_acts/
                # test_doc_acts/diff_doc_acts (plus bas) n'indexent JAMAIS la plage
                # [n_train, n_train+n_filler) -- aucun consommateur (sélection de
                # features, juge, sondes) ne lit cette tranche. Le filler ne sert qu'à
                # nourrir le réservoir PENDANT L'EXTRACTION (rôle déjà terminé à ce
                # stade) ; le ré-encoder coûterait jusqu'à ~13x le travail réel utile
                # sur un run avec un filler dominant (540k/584k documents sur le run de
                # référence). Les lignes filler d'all_doc_sae_acts restent non
                # initialisées (jamais lues en aval, cf. slicing ci-dessous) -- pas
                # d'écriture inutile.
                re_encode_targets = build_reencode_targets(n_train, n_filler, len(all_texts))

                # Reprise (R1, AUDIT_SAE_2026-08.md §2.3/§4.3) : cette passe est une
                # SECONDE boucle de plusieurs heures sur tout le corpus (aussi longue
                # que l'extraction elle-même, §2.5) et n'avait aucune reprise avant ce
                # correctif -- même mécanisme que l'extraction P1 (checkpoint périodique
                # + reconstruction des documents déjà traités depuis les fragments).
                # `next_idx` indexe désormais une POSITION dans `re_encode_targets`,
                # pas un indice de document brut (le filler crée un trou dans la
                # numérotation des documents traités).
                # Particularité : un fragment déjà réencodé n'a PLUS `raw_acts` (purgé
                # intentionnellement, cf. save_fragment ci-dessous) -- doc_maxpool sur
                # le CSR déjà fusionné (core+extra) suffit à reconstruire le vecteur
                # document sans avoir besoin de raw_acts pour les documents déjà faits.
                _reencode_progress_path = _checkpoint_path(CACHE_DIR, "p1_reencode")
                _eval_raw_path = os.path.join(CACHE_DIR, "p1_eval_raw_tokens.pt")
                _reencode_progress = _read_checkpoint(_reencode_progress_path)
                _reencode_resume_from = 0
                _eval_raw = []
                if _reencode_progress is not None and 0 < _reencode_progress["next_idx"] < len(re_encode_targets):
                    _reencode_resume_from = _reencode_progress["next_idx"]
                    print(f"  [P1] Reprise du ré-encodage à la position {_reencode_resume_from}"
                          f"/{len(re_encode_targets)} (checkpoint précédent, filler exclu).")
                    for _pos in tqdm(range(_reencode_resume_from),
                                     desc="Reprise (fragments->vecteurs, ré-encodage)"):
                        _di = re_encode_targets[_pos]
                        all_doc_sae_acts[_di].copy_(doc_maxpool(load_fragment(token_fragments_dir, _di)))
                    # p1_eval_raw_tokens.pt (raw_acts du split test, capturés AVANT purge) :
                    # si le run précédent a été coupé pendant ou après la fenêtre
                    # d'évaluation, ce fichier peut déjà contenir une capture partielle --
                    # la reprendre comme base plutôt que de repartir de zéro (les
                    # raw_acts des documents déjà réencodés sont, eux, irrécupérables :
                    # purgés de leur fragment).
                    if os.path.exists(_eval_raw_path):
                        _eval_raw = [torch.load(_eval_raw_path, map_location="cpu", weights_only=True)]
                    # En position dans re_encode_targets (filler exclu), le test
                    # commence juste après le train, à la position n_train (pas
                    # n_train+n_filler comme en indice de document brut).
                    if _reencode_resume_from > n_train and not _eval_raw:
                        print("  [P1] ATTENTION : reprise après la fenêtre d'évaluation "
                              "(test) sans p1_eval_raw_tokens.pt préexistant -- ces "
                              "raw_acts ont été purgés par le run précédent, l'échantillon "
                              "d'évaluation FVE/rho_SAE sera vide ou incomplet pour ce run.")

                _EVAL_CAP = 4096   # capture x_t brut du split test avant purge (fix B1)
                _GracefulShutdown.install()
                _last_checkpoint_reencode = _reencode_resume_from
                _reencode_early_exit = False
                _fragment_writer = AsyncFragmentWriter()  # audit perf G2 -- cf. flush() avant checkpoint plus bas
                with torch.no_grad():
                    for _pos in tqdm(range(_reencode_resume_from, len(re_encode_targets)),
                                     desc="Re-encodage SAEBoostResidualSAE (sparse, filler exclu)"):
                        i = re_encode_targets[_pos]
                        frag = load_fragment(token_fragments_dir, i)
                        raw_acts = frag["raw_acts"].to(DEVICE).float()
                        # L'encodeur extra lit x directement (SAE Boost §3.1, correctif
                        # frozen_core.py de cette session) -- PAS le résidu e = x - x̂_core.
                        # decode_core_sparse (reconstruction du core, O(nnz*d_in)) est donc
                        # devenu inutile ici : c'était uniquement pour calculer ce résidu,
                        # que l'encodeur ne consomme plus. Avant ce correctif, cet appel
                        # passait encore le résidu à _encode_extra_acts alors que
                        # frozen_core.py avait déjà changé de convention -- régression
                        # silencieuse détectée en relisant ce site d'appel après coup
                        # (accès direct à la méthode privée, donc pas couvert par les tests
                        # de encode()/forward() qui, eux, avaient été mis à jour).
                        token_extra_acts = ext_sae._encode_extra_acts(raw_acts)
                        if n_train + n_filler <= i < n_train + n_filler + n_test and \
                           sum(t.shape[0] for t in _eval_raw) < _EVAL_CAP:
                            _eval_raw.append(raw_acts.float().cpu())

                        csr = merge_extra(frag, token_extra_acts.float().cpu(), d_core)
                        del token_extra_acts
                        save_fragment(token_fragments_dir, i,
                                      token_strings=frag["token_strings"],
                                      csr=csr, d_total=d_core + D_EXTRA,  # raw_acts non repassé -> purgé
                                      writer=_fragment_writer)
                        all_doc_sae_acts[i].copy_(
                            doc_maxpool({
                                "rowptr": csr[0],
                                "cols": csr[1],
                                "vals": csr[2],
                                "shape": csr[3],
                            })
                        )

                        if (_pos + 1 - _last_checkpoint_reencode >= EXTRACTION_CHECKPOINT_INTERVAL
                                or _GracefulShutdown.requested):
                            # flush() AVANT le checkpoint -- même raison qu'en extraction
                            # (cf. docstring AsyncFragmentWriter) : un fragment réencodé
                            # ENCORE en file d'attente à ce moment est un fragment purgé
                            # de son raw_acts sans que le core ait été récupéré nulle
                            # part ailleurs -- perte définitive si le checkpoint le
                            # marquait "fait" avant que l'écriture n'ait réellement eu lieu.
                            _fragment_writer.flush()
                            _write_checkpoint(_reencode_progress_path, next_idx=_pos + 1)
                            if _eval_raw:   # sauvegarde incrémentale : irrécupérable après purge sinon
                                torch.save(torch.cat(_eval_raw)[:_EVAL_CAP], _eval_raw_path)
                            _last_checkpoint_reencode = _pos + 1
                        if _GracefulShutdown.requested:
                            print(f"  [P1] Signal de coupure reçu -- checkpoint écrit à la position "
                                  f"{_pos+1}/{len(re_encode_targets)} (filler exclu), arrêt propre "
                                  "(reprise au prochain sbatch).")
                            _reencode_early_exit = True
                            break

                if _reencode_early_exit:
                    _fragment_writer.close()
                    sys.exit(0)
                _fragment_writer.close()

                _clear_checkpoint(_reencode_progress_path)
                torch.save(all_doc_sae_acts, cache_acts_ext)
                if _eval_raw:
                    torch.save(torch.cat(_eval_raw)[:_EVAL_CAP], _eval_raw_path)
                    
            d_total = d_core + D_EXTRA
            active_sae = ext_sae
            print(f"  [P1] Dimension SAE étendue : {d_core} core + {D_EXTRA} extra = {d_total}")

    train_doc_acts = all_doc_sae_acts[:n_train]
    test_doc_acts  = all_doc_sae_acts[n_train + n_filler: n_train + n_filler + n_test]
    diff_doc_acts  = all_doc_sae_acts[n_train + n_filler + n_test:]  # corpus energy/sports/support, post-hoc

    # ─── LABELS DÉCOUPLÉS : GemmaScope (core) ⊕ Juge LLM (extension) ──────
    #
    # 1. CORE (idx < d_core) : labels officiels GemmaScope récupérés via
    #    Neuronpedia (cache JSON produit hors-cluster par
    #    src/sae/neuronpedia_labels.fetch_neuronpedia_labels). PAS de juge :
    #    ces features sont déjà auto-interprétées côté DeepMind.
    # 2. EXTENSION (idx >= d_core) : juge LLM odd-one-out, seul moyen de
    #    labelliser des features qui n'existent nulle part ailleurs.
    # 3. Sélection top-N SÉPARÉE par plage : les magnitudes JumpReLU du core
    #    (non bornées, outliers ~1e5) écraseraient systématiquement celles de
    #    l'extension TopK dans un classement global — un top-N joint ne
    #    sélectionnerait que du core. Deux appels à
    #    feature_selection_by_magnitude(lo, hi) rendent les deux parties
    #    comparables indépendamment.

    # -- Labels core : Neuronpedia (cache local partagé, jamais d'appel réseau) --
    # NEURONPEDIA_LABELS_PATH (src/config.py) pointe vers un emplacement UNIQUE
    # partagé par tous les runs (local_data/neuronpedia_labels/), pas un cache par
    # SAVE_DIR -- évite de re-télécharger/dupliquer le même fichier par run.
    np_labels_path = os.environ.get("NEURONPEDIA_LABELS", NEURONPEDIA_LABELS_PATH)
    labels_core: dict[int, str] = {}
    if os.path.exists(np_labels_path):
        with open(np_labels_path, "r", encoding="utf-8") as f:
            labels_core = {int(k): v for k, v in json.load(f).items()}
        print(f"  [P1 Labels] {len(labels_core)} labels GemmaScope (Neuronpedia) chargés depuis {np_labels_path}.")
    else:
        print(f"  [P1 Labels] WARN : {np_labels_path} absent — features core affichées F{{idx}}. "
              "Générer le cache hors-cluster via fetch_neuronpedia_labels().")

    # -- Sélection top-N par plage ------------------------------------------
    print("  [P1 Labels] Sélection par magnitude token-level, plages core / extension séparées...")
    top_core_indices = feature_selection_by_magnitude(
        token_fragments_dir, list(range(n_train)), d_total, N_FEATURES_TO_LABEL,
        lo=0, hi=d_core,
    )
    top_ext_indices = []
    if USE_FROZEN_CORE and d_total > d_core:
        top_ext_indices = feature_selection_by_magnitude(
            token_fragments_dir, list(range(n_train)), d_total, N_FEATURES_TO_LABEL,
            lo=d_core, hi=d_total,
        )

    with open(os.path.join(SAVE_DIR, "p1_top_core_features.json"), "w", encoding="utf-8") as f:
        json.dump({int(i): labels_core.get(int(i), f"F{i}") for i in top_core_indices},
                  f, indent=2, ensure_ascii=False)

    # -- Juge LLM : extension UNIQUEMENT ------------------------------------
    judge_cache = os.path.join(CACHE_DIR, "p1_judge_labels_extended.json")
    judge_ext_data = {}
    if top_ext_indices:
        if os.path.exists(judge_cache):
            print(f"  [P1 Judge] Restauration labels extension : {judge_cache}")
            with open(judge_cache, "r", encoding="utf-8") as f:
                judge_ext_data = json.load(f)
        else:
            print(f"  [P1 Judge] Chargement Gemma-3 — labellisation des "
                  f"{len(top_ext_indices)} features EXTENSION uniquement...")
            expert_tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID, token=HF_TOKEN, trust_remote_code=True, local_files_only=True
            )
            expert_model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, torch_dtype=TORCH_DTYPE, device_map=DEVICE,
                low_cpu_mem_usage=True,
                token=HF_TOKEN, trust_remote_code=True, local_files_only=True
            ).eval()
            judge_ext_data = odd_one_out_judge(
                model=expert_model, tokenizer=expert_tokenizer,
                feature_indices=top_ext_indices,
                token_fragments_dir=token_fragments_dir,
                acts=train_doc_acts, offset=0,
            )
            print("  [P1 Judge] Libération VRAM + malloc_trim...")
            del expert_model, expert_tokenizer
            _trim_host_memory()
            with open(judge_cache, "w", encoding="utf-8") as f:
                json.dump(judge_ext_data, f, indent=2, ensure_ascii=False)

    # -- Fusion : core (Neuronpedia) ∪ extension (juge), préfixées [EXT] ----
    label_map_p1 = dict(labels_core)
    for k, v in judge_ext_data.items():
        label_map_p1[int(k)] = "[EXT] " + v.get("label", f"F{k}")

    with open(os.path.join(SAVE_DIR, "p1_top_extended_features.json"), "w", encoding="utf-8") as f:
        json.dump(judge_ext_data, f, indent=2, ensure_ascii=False)

    # -- Comparaison core vs extension, côte à côte -------------------------
    print("\n  [P1] TOP FEATURES — CORE GemmaScope (labels Neuronpedia) :")
    for i in top_core_indices:
        print(f"    F{i:<7d} {label_map_p1.get(int(i), f'F{i}')[:80]}")
    if top_ext_indices:
        print("  [P1] TOP FEATURES — EXTENSION FrozenCore (labels juge LLM) :")
        for i in top_ext_indices:
            print(f"    F{i:<7d} {label_map_p1.get(int(i), f'F{i}')[:80]}")

    umap_res_test = analyze_with_umap(
        texts=test_texts, sae_acts=test_doc_acts, labels=test_labels,
        filename="umap_pipeline1_emails.html",
        title=f"Pipeline 1: Gemma-3 L{LAYER} → Max-Pool SAE Acts (Emails EDF, test)",
        token_fragments_dir=token_fragments_dir, offset=n_train + n_filler, feature_labels=label_map_p1,
    )
    if diff_texts:
        analyze_with_umap(
            texts=diff_texts, sae_acts=diff_doc_acts, labels=diff_labels,
            filename="umap_pipeline1_diffcorpus.html",
            title=f"Pipeline 1: Gemma-3 L{LAYER} → Max-Pool SAE Acts (energy/sports/support, post-hoc)",
            token_fragments_dir=token_fragments_dir, offset=n_train + n_filler + n_test, feature_labels=label_map_p1,
        )

    # Diffing cross-domaine (démonstration, corpus secondaire post-hoc -- cf.
    # docstring de la fonction) : energy vs sports, jamais le corpus d'entraînement.
    energy_mask = np.array([l == "energy" for l in diff_labels])
    sports_mask  = np.array([l == "sports"  for l in diff_labels])
    diff_hypothesis = "Aucun écart mesurable."
    if energy_mask.sum() > 0 and sports_mask.sum() > 0:
        # corpus_diff_stats : test exact de Fisher par feature + correction BH
        # (remplace diff_features, écarts de fréquences sans contrôle du FDR).
        pair_mask = energy_mask | sports_mask
        diff_df = corpus_diff_stats(
            diff_doc_acts[torch.from_numpy(pair_mask)].float(),
            group_mask=energy_mask[pair_mask],       # True = Énergie (corpus A)
            feature_labels=label_map_p1,
        )
        diff_df.to_csv(os.path.join(SAVE_DIR, "p1_diff_energy_sports.csv"), index=False)

        if os.environ.get("RUN_DIFF_HYPOTHESIS", "1") == "1":
            j_tok = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True, local_files_only=True)
            j_llm = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, torch_dtype=TORCH_DTYPE, device_map=DEVICE,
                low_cpu_mem_usage=True,
                token=HF_TOKEN, trust_remote_code=True, local_files_only=True
            ).eval()
            with torch.no_grad():
                diff_hypothesis = generate_llm_diff_hypothesis(j_llm, j_tok, diff_df, "Énergie", "Sports")
            print(f"  [Task 1] Hypothèse LLM :\n  {diff_hypothesis}\n")
            del j_llm, j_tok
            _trim_host_memory()
        else:
            print("  [Task 1] RUN_DIFF_HYPOTHESIS=0 — hypothèse LLM sautée (3e chargement 12B évité).")

    freq = (test_doc_acts > 1e-6).float().mean(0)
    keep_npmi = ((freq >= 0.01) & (freq <= 0.5)).nonzero(as_tuple=True)[0][:4000]
    npmi_mat = compute_npmi(test_doc_acts[:, keep_npmi])
    torch.save({"npmi": npmi_mat, "feature_ids": keep_npmi}, os.path.join(CACHE_DIR, "p1_npmi.pt"))

    # ─── Corrélations "intéressantes" (interp_embed §4.2, cf. docs/references.md) ──
    # NPMI seul (ci-dessus) mélange corrélations triviales (labels quasi-synonymes,
    # ex. "facturation"/"montant dû") et surprenantes -- on isole les secondes en
    # filtrant par dissimilarité sémantique des labels (embeddings bge-m3, cf. note
    # de select_latents_by_similarity -- plus fiable que F2LLM sur des labels courts).
    print("  [Corrélations] Recherche de paires NPMI intéressantes (NPMI élevé, labels dissimilaires)...")
    G_cooc = cooccurrence_graph(test_doc_acts, feature_labels=label_map_p1)
    if G_cooc.number_of_edges() > 0:
        node_ids = list(G_cooc.nodes)
        label_embs = _embed_bge_m3([G_cooc.nodes[n]["label"] for n in node_ids])
        label_embeddings = {n: label_embs[i].numpy() for i, n in enumerate(node_ids)}
        interesting_pairs = find_interesting_pairs(G_cooc, label_embeddings)
        with open(os.path.join(SAVE_DIR, "p1_interesting_correlations.json"), "w", encoding="utf-8") as f:
            json.dump(interesting_pairs[:50], f, indent=2, ensure_ascii=False)
        print(f"  [Corrélations] {len(interesting_pairs)} paires intéressantes "
              f"(NPMI>0.6, similarité label<0.2) sur {G_cooc.number_of_edges()} arêtes.")
    else:
        print("  [Corrélations] Graphe vide (aucune arête au-dessus du seuil NPMI) -- rien à filtrer.")

    # Requêtes retrieval/clustering alignées sur le domaine réel (test_texts =
    # emails+augmentés, plus des chunks energy/FineWeb-2).
    targeted_clustering_by_axis(
        texts=test_texts, sae_acts=test_doc_acts, labels=test_labels,
        feature_labels=label_map_p1, axis_query="urgence réclamation client"
    )

    results_retrieval = property_based_retrieval(
        "facturation résiliation panne", test_doc_acts, test_texts, label_map_p1
    )
    for rank, (doc, score) in enumerate(results_retrieval):
        print(f"    Rang {rank+1} (Boltzmann={score:.4f}) : {doc[:100]}...")

    silhouette = compute_silhouette(test_doc_acts, test_labels)
    l0_mean    = (test_doc_acts > 1e-6).float().sum(dim=-1).mean().item()
    dead_pct   = (test_doc_acts.sum(dim=0) == 0).float().mean().item() * 100
    
    print("  [Metrics] Chargement des tokens bruts d'évaluation (cache dédié, non purgé)...")
    eval_raw_path = os.path.join(CACHE_DIR, "p1_eval_raw_tokens.pt")

    if os.path.exists(eval_raw_path):
        eval_raw_tokens = torch.load(eval_raw_path, weights_only=True)  # x_t bruts, fp32, [n, d_in]
        raw_tokens_tensor = eval_raw_tokens[:1000].to(DEVICE).to(TORCH_DTYPE)
        with torch.no_grad():
            rho_sae = compute_rho_sae(active_sae, raw_tokens_tensor,
                                      n_sample=500, is_saelens=not USE_FROZEN_CORE, device=DEVICE)
    else:
        print("  [Metrics] WARN: p1_eval_raw_tokens.pt absent (run antérieur au fix B1 — "
              "fragments déjà purgés, x_t irrécupérable). Purger le cache et relancer P1.")
        eval_raw_tokens = None
        rho_sae = float("nan")

    print("\n  [FR/EN] Comparaison FVE baseline sur un échantillon de tokens...")
    if eval_raw_tokens is not None:
        token_sample = eval_raw_tokens[:4096]
        with torch.no_grad():
            metrics_pretrained = compute_metrics(
                pretrained_sae, token_sample,
                is_saelens=True, device=DEVICE
            )
        print(f"  FVE (pretrained, tokens FR) = {metrics_pretrained['FVE']:.4f} | "
              f"NMSE = {metrics_pretrained['NMSE']:.4f}")
              
        if USE_FROZEN_CORE and active_sae is not pretrained_sae:
            with torch.no_grad():
                metrics_ext = compute_metrics(active_sae, token_sample,
                                              is_saelens=False, device=DEVICE)
            print(f"  FVE (SAEBoostResidualSAE, tokens FR) = {metrics_ext['FVE']:.4f} | "
                  f"NMSE = {metrics_ext['NMSE']:.4f} | "
                  f"ΔFVE = {metrics_ext['FVE'] - metrics_pretrained['FVE']:+.4f}")
    else:
        metrics_pretrained = {"FVE": float("nan")}
        print("  [Metrics] Échantillon de tokens indisponible pour la FVE.")

    print("\n  [Downstream P1] Sonde logistique sur SAE activations (energy vs sports, corpus diffing)...")
    en_mask = torch.from_numpy(energy_mask)
    sp_mask = torch.from_numpy(sports_mask)
    if en_mask.sum() > 0 and sp_mask.sum() > 0:
        try:
            clf_results = downstream_classification(
                acts_by_label={
                    "energy": diff_doc_acts[en_mask],
                    "sports": diff_doc_acts[sp_mask],
                }
            )
        except Exception as e:
            print(f"  [Downstream P1] WARN: Classification failed: {e}")
            clf_results = {}
    else:
        print(f"  [Downstream P1] Échantillons insuffisants pour entraîner la sonde logistique.")
        clf_results = {}

    # Sonde logistique sur le corpus principal (emails+augmentés) : le SAE
    # sépare-t-il linéairement les axes de perturbation (émotion, urgence,
    # registre...) et l'original ? Question bien plus pertinente pour le cas
    # d'usage EDF que le probe energy/sports (générique, corpus secondaire).
    print("  [Downstream P1] Sonde logistique sur SAE activations (axes email, corpus principal)...")
    train_labels_arr = np.array(train_labels)
    train_groups_arr = np.array(train_groups) if train_groups is not None else None
    label_counts = pd.Series(train_labels_arr).value_counts()
    usable_labels = label_counts[label_counts >= 10].index.tolist()  # StratifiedKFold(5) minimum
    clf_results_email = {}
    if len(usable_labels) >= 2:
        try:
            acts_by_label_email = {
                lbl: train_doc_acts[torch.from_numpy(train_labels_arr == lbl)]
                for lbl in usable_labels
            }
            groups_by_label_email = (
                {lbl: train_groups_arr[train_labels_arr == lbl] for lbl in usable_labels}
                if train_groups_arr is not None else None
            )
            clf_results_email = downstream_classification(
                acts_by_label=acts_by_label_email, groups_by_label=groups_by_label_email)
            print(f"  [Downstream P1] acc_SAE (axes email, {len(usable_labels)} classes) = "
                  f"{clf_results_email.get('acc_sae', float('nan')):.4f}")
        except Exception as e:
            print(f"  [Downstream P1] WARN: Classification (axes email) failed: {e}")
    else:
        print("  [Downstream P1] Pas assez de classes email avec ≥10 échantillons pour la sonde.")

    # ─── HYGIÈNE MÉMOIRE HÔTE avant Pipeline 2 ───────────────────────────
    # test_doc_acts est une VUE de all_doc_sae_acts : la retourner telle quelle
    # maintiendrait vivant le storage complet [n_docs, 263168] fp32 (~5 Go).
    # .clone() détache le slice ; on libère ensuite le tenseur global, les SAEs
    # GPU (262k×4096 bf16 ≈ 4 Go VRAM + copies hôte) et on rend les arènes glibc.
    test_doc_acts_out = test_doc_acts.clone()
    results = {
        "L0": l0_mean, "dead_pct": dead_pct, "silhouette": silhouette,
        "rho_sae": rho_sae,
        "n_clusters": umap_res_test["n_clusters"],
        "active_features": umap_res_test["n_active"],
        "diff_hypothesis": diff_hypothesis,
        "clf_acc_sae": clf_results.get("acc_sae", float("nan")),
        "clf_acc_email_axes": clf_results_email.get("acc_sae", float("nan")),
        "clf_n_email_classes": len(usable_labels),
        "fve_pretrained": metrics_pretrained.get("FVE", float("nan")),
        "_test_doc_acts": test_doc_acts_out,
        "_label_map": label_map_p1,
        "_top_core": top_core_indices,
        "_top_ext": top_ext_indices,
    }
    del all_doc_sae_acts, train_doc_acts, test_doc_acts, diff_doc_acts
    del pretrained_sae, active_sae
    if USE_FROZEN_CORE and "ext_sae" in dir():
        try:
            del ext_sae
        except NameError:
            pass
    del umap_res_test
    _trim_host_memory()
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE 2 — F2LLM PHRASE-LEVEL SAE
# ══════════════════════════════════════════════════════════════════════════════

def run_f2llm_pipeline(
    train_texts: list,
    train_labels: list,
    test_texts: list,
    test_labels: list,
    diff_texts: list = None,
    diff_labels: list = None,
    test_groups: list = None,
) -> dict:
    """train_texts/test_texts : corpus principal (emails+augmentés) -- entraîne
    directement le PhraseLevelSAE. diff_texts/diff_labels : corpus secondaire
    (energy/sports/support), encodé post-hoc pour la démo de diffing uniquement.
    test_groups (optionnel, défaut None -> comportement inchangé) : parent_id
    (mail d'origine) de chaque `test_texts[i]` -- CV group-aware pour la sonde
    "axes email" (`RESULTS_TESTS.md` §57), même logique que Pipeline 1."""
    print("\n" + "=" * 70)
    print(" PIPELINE 2 : F2LLM-v2 PHRASE-LEVEL SAE → MAX-POOL DOCUMENT")
    print("=" * 70)

    diff_texts  = diff_texts or []
    diff_labels = diff_labels or []

    train_phrases, train_p2d = split_into_phrases(train_texts, max_phrases_per_doc=MAX_PHRASES_DOC)
    print(f"  Train : {len(train_texts)} docs → {len(train_phrases)} phrases")

    train_phrase_emb, d_in = extract_f2llm_embeddings(
        train_phrases, max_length=128,
        cache_path=os.path.join(CACHE_DIR, f"train_phrase_emb_dim{MATRYOSHKA_DIM}_n{len(train_phrases)}"),
    )

    idx = torch.randperm(len(train_phrase_emb), generator=torch.Generator().manual_seed(SEED))
    split = int(len(idx) * 0.85)
    emb_train_split = train_phrase_emb[idx[:split]]
    emb_eval_split  = train_phrase_emb[idx[split:]]

    sae_path = os.path.join(SAVE_DIR, f"p2_sae_dim{d_in}_d{D_SAE}_k{K_SPARSE}.pt")
    sae, history = load_or_train_sae(d_in=d_in, d_sae=D_SAE, k=K_SPARSE,
                                    embeddings=emb_train_split, save_path=sae_path,
                                    epochs=EPOCHS, lr=LR)
    m_eval = compute_sae_metrics(sae, emb_eval_split)
    rho_sae_p2 = compute_rho_sae(sae, emb_eval_split, n_sample=500, device=DEVICE)
    del emb_train_split, emb_eval_split; gc.collect(); torch.cuda.empty_cache()

    test_phrases, test_p2d_list = split_into_phrases(test_texts, max_phrases_per_doc=MAX_PHRASES_DOC)
    print(f"  Test  : {len(test_texts)} docs → {len(test_phrases)} phrases")
    test_phrase_emb, _ = extract_f2llm_embeddings(
        test_phrases, max_length=128,
        cache_path=os.path.join(CACHE_DIR, f"test_phrase_emb_dim{MATRYOSHKA_DIM}_n{len(test_phrases)}"),
    )
    test_p2d_arr = np.array(test_p2d_list)
    doc_acts = encode_documents_with_phrase_sae(
        n_docs=len(test_texts), sae=sae,
        phrase_embeddings=test_phrase_emb, phrase_to_doc=test_p2d_arr,
    )

    diff_doc_acts = None
    if diff_texts:
        diff_phrases, diff_p2d_list = split_into_phrases(diff_texts, max_phrases_per_doc=MAX_PHRASES_DOC)
        diff_phrase_emb, _ = extract_f2llm_embeddings(
            diff_phrases, max_length=128,
            cache_path=os.path.join(CACHE_DIR, f"diffcorpus_phrase_emb_dim{MATRYOSHKA_DIM}_n{len(diff_phrases)}"),
        )
        diff_p2d_arr = np.array(diff_p2d_list)
        diff_doc_acts = encode_documents_with_phrase_sae(
            n_docs=len(diff_texts), sae=sae,
            phrase_embeddings=diff_phrase_emb, phrase_to_doc=diff_p2d_arr,
        )

    # ─── LLM Judge P2 ────────────────────────────────────────────────────────
    top_feat_indices = doc_acts.float().mean(dim=0).topk(N_FEATURES_TO_LABEL).indices.tolist()
    judge_cache = os.path.join(CACHE_DIR, "p2_feature_labels.json")
    if os.path.exists(judge_cache):
        with open(judge_cache, "r", encoding="utf-8") as f:
            feature_labels_p2 = json.load(f)
    else:
        j_tok = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, local_files_only=True)
        j_llm = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=TORCH_DTYPE, device_map=DEVICE,
            low_cpu_mem_usage=True, local_files_only=True
        ).eval()
        # local_gemma_judge attend les activations et textes au niveau PHRASE
        # (pas au niveau document max-poolé) — ce sont les phrases individuelles
        # (test_phrases / test_phrase_emb) qui servent d'exemples au juge LLM.
        sae.eval()
        with torch.no_grad():
            test_phrase_acts = sae.encode(test_phrase_emb.to(DEVICE)).cpu()
        feature_labels_p2 = local_gemma_judge(
            model=j_llm, tokenizer=j_tok, feature_indices=top_feat_indices,
            phrase_texts=test_phrases, phrase_acts=test_phrase_acts,
            phrase_to_doc=test_p2d_arr,
        )
        del j_llm, j_tok
        _trim_host_memory()
        with open(judge_cache, "w", encoding="utf-8") as f:
            json.dump(feature_labels_p2, f, indent=2, ensure_ascii=False)

    label_map_p2 = {int(idx): entry.get("label", f"F{idx}") for idx, entry in feature_labels_p2.items()}

    activating_phrases_map = {}
    # Regroupement O(n) une seule fois -- remplace le np.where(test_p2d_arr ==
    # doc_idx) par document, O(n_docs * n_phrases) sur le nombre total de
    # comparaisons (audit perf, item 10).
    doc_to_phrase_indices = group_indices_by_doc(test_p2d_arr.tolist())
    sae.eval()
    with torch.no_grad():
        for doc_idx in range(len(test_texts)):
            phrase_indices = doc_to_phrase_indices.get(doc_idx, [])
            if not phrase_indices: continue
            row_acts = doc_acts[doc_idx]
            top_vals, top_f_ids = row_acts.topk(min(3, row_acts.shape[0]))
            active_feats = [f for f, v in zip(top_f_ids.tolist(), top_vals.tolist()) if v > 1e-6]
            if not active_feats: continue
            phrase_emb_doc = test_phrase_emb[phrase_indices].to(DEVICE)
            activating_phrases_map[doc_idx] = get_activating_tokens_for_doc(
                token_strings=[test_phrases[j] for j in phrase_indices],
                token_residuals=phrase_emb_doc, sae=sae,
                top_feature_indices=active_feats, top_k_tokens=1,
            )

    umap_res_test = analyze_with_umap(
        texts=test_texts, sae_acts=doc_acts, labels=test_labels,
        filename="umap_pipeline2_emails.html",
        title="Pipeline 2 : F2LLM-v2 Phrase SAE → Max-Pool Document (Emails EDF, test)",
        activating_tokens_map=activating_phrases_map, feature_labels=label_map_p2,
    )
    if diff_texts and diff_doc_acts is not None:
        analyze_with_umap(
            texts=diff_texts, sae_acts=diff_doc_acts, labels=diff_labels,
            filename="umap_pipeline2_diffcorpus.html",
            title="Pipeline 2 : F2LLM-v2 Phrase SAE → Max-Pool Document (energy/sports/support, post-hoc)",
            feature_labels=label_map_p2,
        )

    print("\n  [Downstream P2] Sonde logistique sur SAE activations (energy vs sports, corpus diffing)...")
    energy_mask_diff = np.array([l == "energy" for l in diff_labels])
    sports_mask_diff  = np.array([l == "sports"  for l in diff_labels])
    if energy_mask_diff.sum() > 0 and sports_mask_diff.sum() > 0 and diff_doc_acts is not None:
        try:
            diff_phrase_emb_pooled = pool_embeddings_by_document(
                diff_phrase_emb, diff_p2d_arr, n_docs=len(diff_texts)
            )
            clf_results_p2 = downstream_classification(
                acts_by_label={
                    "energy": diff_doc_acts[torch.from_numpy(energy_mask_diff)],
                    "sports": diff_doc_acts[torch.from_numpy(sports_mask_diff)],
                },
                raw_emb_by_label={
                    "energy": diff_phrase_emb_pooled[torch.from_numpy(energy_mask_diff)],
                    "sports": diff_phrase_emb_pooled[torch.from_numpy(sports_mask_diff)],
                }
            )
        except Exception as e:
            print(f"  [Downstream P2] WARN: Classification failed: {e}")
            clf_results_p2 = {}
    else:
        print(f"  [Downstream P2] Insufficient samples: energy={energy_mask_diff.sum()}, sports={sports_mask_diff.sum()}")
        clf_results_p2 = {}

    # Sonde sur les axes email (test split, corpus principal) : cf. P1 pour la
    # même logique/justification (probe plus pertinent que energy/sports ici).
    print("  [Downstream P2] Sonde logistique sur SAE activations (axes email, corpus principal)...")
    test_labels_arr = np.array(test_labels)
    test_groups_arr = np.array(test_groups) if test_groups is not None else None
    label_counts_p2 = pd.Series(test_labels_arr).value_counts()
    usable_labels_p2 = label_counts_p2[label_counts_p2 >= 10].index.tolist()
    clf_results_p2_email = {}
    if len(usable_labels_p2) >= 2:
        try:
            acts_by_label_email_p2 = {
                lbl: doc_acts[torch.from_numpy(test_labels_arr == lbl)]
                for lbl in usable_labels_p2
            }
            groups_by_label_email_p2 = (
                {lbl: test_groups_arr[test_labels_arr == lbl] for lbl in usable_labels_p2}
                if test_groups_arr is not None else None
            )
            clf_results_p2_email = downstream_classification(
                acts_by_label=acts_by_label_email_p2, groups_by_label=groups_by_label_email_p2)
            print(f"  [Downstream P2] acc_SAE (axes email, {len(usable_labels_p2)} classes) = "
                  f"{clf_results_p2_email.get('acc_sae', float('nan')):.4f}")
        except Exception as e:
            print(f"  [Downstream P2] WARN: Classification (axes email) failed: {e}")
    else:
        print("  [Downstream P2] Pas assez de classes email avec ≥10 échantillons pour la sonde.")

    silhouette_p2 = compute_silhouette(doc_acts, test_labels)
    del sae, doc_acts, test_phrase_emb, train_phrase_emb
    if diff_doc_acts is not None: del diff_doc_acts
    _trim_host_memory()

    return {
        **m_eval,
        "rho_sae": rho_sae_p2,
        "silhouette": silhouette_p2,
        "n_clusters": umap_res_test["n_clusters"],
        "active_features": umap_res_test["n_active"],
        "clf_acc_sae": clf_results_p2.get("acc_sae", float("nan")),
        "clf_acc_raw": clf_results_p2.get("acc_raw", float("nan")),
        "clf_delta":   clf_results_p2.get("delta_acc", float("nan")),
        "clf_acc_email_axes": clf_results_p2_email.get("acc_sae", float("nan")),
        "clf_n_email_classes": len(usable_labels_p2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEERING DEMO
# ══════════════════════════════════════════════════════════════════════════════

def run_steering_demo(p1_results: dict):
    doc_acts  = p1_results.get("_test_doc_acts")
    label_map = p1_results.get("_label_map", {})
    if doc_acts is None:
        return
    print("\n" + "=" * 70)
    print(" STEERING DEMO (P1 SAE)")
    print("=" * 70)
    mean_acts = doc_acts.float().mean(dim=0)
    top_f = int(mean_acts.argmax())
    top_label = label_map.get(top_f, f"F{top_f}")
    print(f"  Concept ciblé : Feature #{top_f} ({top_label}) | µ={mean_acts[top_f]:.4f}")
    suppressed = steer_activations(doc_acts, {top_f: 0.0})
    amplified  = steer_activations(doc_acts, {top_f: 3.0})
    orig_norm = F.normalize(doc_acts.float(), dim=-1)
    cos_sup = (orig_norm * F.normalize(suppressed.float(), dim=-1)).sum(dim=-1).mean().item()
    cos_amp = (orig_norm * F.normalize(amplified.float(),  dim=-1)).sum(dim=-1).mean().item()
    print(f"  cos_sim suppression  : {cos_sup:.4f}")
    print(f"  cos_sim amplification: {cos_amp:.4f}")
    with open(os.path.join(SAVE_DIR, "p1_steering_demo.json"), "w") as f:
        json.dump({"target_feature": top_f, "target_label": top_label,
                   "cos_sim_suppressed": cos_sup, "cos_sim_amplified": cos_amp}, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" CHARGEMENT DU CORPUS")
    print("=" * 70)

    # ─── CORPUS PRINCIPAL : emails originaux + variantes augmentées ────────────────
    # Domine l'entraînement du SAE (reservoir de résidus + PhraseLevelSAE) ; le
    # corpus generic energy/sports/support ne sert plus qu'au diffing post-hoc.
    # Split GROUP-AWARE par mail d'origine (aucune fuite d'une variante augmentée
    # d'un mail de test vers le train).
    # seed=CORPUS_SPLIT_SEED (PAS SEED) : le split train/test doit rester identique
    # entre deux runs qui ne diffèrent que par SEED (ablation de variance
    # d'entraînement) -- sinon la comparaison mélangerait variance d'entraînement
    # et variance de corpus.
    # CONFIRMATORY_DOMAIN_BASELINE (défaut désactivé, comportement 100% inchangé
    # sinon) : réplique à n=150 (test apparié) le protocole du corpus generic
    # (energy/sports/support, même logique de split), pour comparer domaine-vs-
    # volume à effectif comparable plutôt qu'à effectifs confondus.
    CONFIRMATORY_DOMAIN_BASELINE = os.environ.get("CONFIRMATORY_DOMAIN_BASELINE", "0") == "1"

    if CONFIRMATORY_DOMAIN_BASELINE:
        print("  [CONFIRMATORY_DOMAIN_BASELINE=1] Corpus principal = generic "
              "energy/sports/support (réplique n=150 du baseline pré-correctif).")
        energy_texts = prepare_domain_dataset(
            ENERGY_KEYWORDS, "energy", N_TOTAL_ENERGY,
            chunk_length=1024, max_chunks=20, url_patterns=ENERGY_URL_PATTERNS,
            local_dataset_path=LOCAL_DATASET_PATH, use_fineweb2=USE_FINEWEB2,
        )
        sports_texts = prepare_domain_dataset(
            SPORTS_KEYWORDS, "sports", N_TOTAL_SPORTS,
            chunk_length=1024, max_chunks=20, url_patterns=SPORTS_URL_PATTERNS,
            local_dataset_path=LOCAL_DATASET_PATH, use_fineweb2=USE_FINEWEB2,
        )
        support_texts = prepare_domain_dataset(
            SUPPORT_KEYWORDS, "support", N_TOTAL_SUPPORT,
            chunk_length=1024, max_chunks=20, url_patterns=SUPPORT_URL_PATTERNS,
            local_dataset_path=LOCAL_DATASET_PATH, use_fineweb2=USE_FINEWEB2,
        )
        _rng = np.random.default_rng(CORPUS_SPLIT_SEED)
        _test_split = float(os.environ.get("CONFIRMATORY_TEST_SPLIT", "0.1"))

        def _split_generic(texts, label):
            n = len(texts)
            idx = _rng.permutation(n)
            n_test = max(1, int(n * _test_split))
            test_idx, train_idx = idx[:n_test], idx[n_test:]
            return (
                [texts[i] for i in train_idx], [label] * (n - n_test),
                [texts[i] for i in test_idx], [label] * n_test,
            )

        en_tr, en_tr_lbl, en_te, en_te_lbl = _split_generic(energy_texts, "energy")
        sp_tr, sp_tr_lbl, sp_te, sp_te_lbl = _split_generic(sports_texts, "sports")
        su_tr, su_tr_lbl, su_te, su_te_lbl = _split_generic(support_texts, "support")
        train_texts = en_tr + sp_tr + su_tr
        train_labels = en_tr_lbl + sp_tr_lbl + su_tr_lbl
        test_texts = en_te + sp_te + su_te
        test_labels = en_te_lbl + sp_te_lbl + su_te_lbl
        print(f"Train (generic, confirmatoire) : {len(train_texts)} chunks | "
              f"Test : {len(test_texts)} chunks")
        diff_texts, diff_labels = [], []
        train_groups = None  # pas de notion de mail d'origine dans ce corpus generique
        test_groups = None
    else:
        with stage_timer("Chargement corpus principal (emails+augmentés)"):
            train_texts, train_labels, test_texts, test_labels, train_groups, test_groups = (
                build_email_train_test_corpus(
                    LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH,
                    test_split=EMAIL_TEST_SPLIT, max_augmented_per_mail=MAX_AUGMENTED_PER_MAIL,
                    seed=CORPUS_SPLIT_SEED, return_groups=True,
                )
            )
        if not train_texts:
            print("  Fallback emails synthétiques (Mails.tsv introuvable).")
            train_texts = [
                "Bonjour, je conteste ma facture d'électricité Linky, hausse injustifiée.",
                "Merci de planifier l'installation de mon compteur de raccordement électrique.",
                "Coupure réseau dans notre rue depuis 2 heures. Envoyez un technicien.",
            ]
            train_labels = ["Reclamation_Facturation", "Mise_En_Service", "Urgence_Technique"]
            test_texts, test_labels = train_texts, train_labels
        print(f"Train (emails+augmentés) : {len(train_texts)} docs | Test : {len(test_texts)} docs")

        # ─── CORPUS SECONDAIRE : diffing cross-domaine (energy vs sports) ──────────
        # Petit corpus generic, gardé UNIQUEMENT pour la démonstration existante de
        # diffing cross-domaine (p1_diff_energy_sports.csv) : encodé post-hoc par le
        # SAE déjà entraîné sur les emails (comme les emails l'étaient avant cette
        # bascule), jamais utilisé pour l'entraînement lui-même.
        with stage_timer("Préparation corpus diffing (energy/sports/support)"):
            energy_texts = prepare_domain_dataset(
                ENERGY_KEYWORDS, "energy", N_TOTAL_ENERGY,
                chunk_length=1024, max_chunks=20, url_patterns=ENERGY_URL_PATTERNS,
                local_dataset_path=LOCAL_DATASET_PATH, use_fineweb2=USE_FINEWEB2,
            )
            sports_texts = prepare_domain_dataset(
                SPORTS_KEYWORDS, "sports", N_TOTAL_SPORTS,
                chunk_length=1024, max_chunks=20, url_patterns=SPORTS_URL_PATTERNS,
                local_dataset_path=LOCAL_DATASET_PATH, use_fineweb2=USE_FINEWEB2,
            )
            support_texts = prepare_domain_dataset(
                SUPPORT_KEYWORDS, "support", N_TOTAL_SUPPORT,
                chunk_length=1024, max_chunks=20, url_patterns=SUPPORT_URL_PATTERNS,
                local_dataset_path=LOCAL_DATASET_PATH, use_fineweb2=USE_FINEWEB2,
            )
        diff_texts  = energy_texts + sports_texts + support_texts
        diff_labels = ["energy"] * len(energy_texts) + ["sports"] * len(sports_texts) + ["support"] * len(support_texts)
        print(f"Corpus diffing (energy/sports/support, post-hoc uniquement) : {len(diff_texts)} chunks")

    # ── Filler de volume (ablation SAE Boost, arXiv:2507.12990) : ajouté
    # UNIQUEMENT au réservoir résiduel via volume_filler_texts
    # (run_llm_max_pool_pipeline), jamais à train_texts lui-même -- sinon la
    # sélection de features/la sonde de classification email réintroduirait le
    # biais de domaine (cf. docstring de la fonction). Désactivé par défaut
    # (N_VOLUME_FILLER_TARGET_CHUNKS=0). Sans filtre thématique
    # (sample_fineweb2_chunks) : le filler isole un effet de VOLUME de tokens,
    # pas de pertinence thématique.
    N_VOLUME_FILLER_TARGET_CHUNKS = int(os.environ.get("N_VOLUME_FILLER_TARGET_CHUNKS", "0"))
    volume_filler_texts = []
    if N_VOLUME_FILLER_TARGET_CHUNKS > 0:
        filler_shards = sorted(glob.glob(os.environ.get(
            "VOLUME_FILLER_DATASET_GLOB", LOCAL_DATASET_PATH)))
        if not filler_shards:
            filler_shards = [LOCAL_DATASET_PATH]
        n_per_shard = max(1, N_VOLUME_FILLER_TARGET_CHUNKS // len(filler_shards))
        print(f"  [filler] Construction du corpus de volume (~{N_VOLUME_FILLER_TARGET_CHUNKS} "
              f"chunks cible sur {len(filler_shards)} shard(s) FineWeb2-fr, sans filtre thématique)...")
        for shard_path in filler_shards:
            shard_texts = sample_fineweb2_chunks(
                n_per_shard, chunk_length=1024, max_chunks=20,
                local_dataset_path=shard_path,
            )
            volume_filler_texts.extend(shard_texts)
            if len(volume_filler_texts) >= N_VOLUME_FILLER_TARGET_CHUNKS:
                break
        print(f"  [filler] {len(volume_filler_texts)} chunks de filler construits "
              f"(hors train_texts, réservoir résiduel uniquement).")

    RUN = set(os.environ.get("PIPELINES", "p1,p2").split(","))
    results_p1 = {}
    if "p1" in RUN:
        with stage_timer("Pipeline 1 (Gemma-3 + GemmaScope-2)"):
            results_p1 = run_llm_max_pool_pipeline(
                train_texts, train_labels, test_texts, test_labels, diff_texts, diff_labels,
                volume_filler_texts=volume_filler_texts, train_groups=train_groups,
            )
            run_steering_demo(results_p1)
    # Le steering n'a plus besoin des doc_acts : libération avant P2 (pic RSS).
    results_p1.pop("_test_doc_acts", None)
    _trim_host_memory()

    results_p2 = {}
    if "p2" in RUN:
        with stage_timer("Pipeline 2 (F2LLM + PhraseLevelSAE)"):
            results_p2 = run_f2llm_pipeline(
                train_texts, train_labels, test_texts, test_labels, diff_texts, diff_labels,
                test_groups=test_groups,
            )

    print("\n" + "=" * 70)
    print(" BILAN COMPARATIF")
    print("=" * 70)
    rows = [
        {
            "Pipeline":    "P1 Gemma-3 SAE (Max-Pool tokens)",
            "NMSE":        "n/a",
            "L0":          f"{results_p1.get('L0', float('nan')):.1f}",
            "dead%":       f"{results_p1.get('dead_pct', float('nan')):.1f}",
            "ρ_SAE":       f"{results_p1.get('rho_sae', float('nan')):.4f}",
            "silhouette":  f"{results_p1.get('silhouette', float('nan')):.4f}",
            "acc_SAE":     f"{results_p1.get('clf_acc_sae', float('nan')):.4f}",
            "acc_axes_email": f"{results_p1.get('clf_acc_email_axes', float('nan')):.4f} "
                               f"({results_p1.get('clf_n_email_classes', 0)} classes)",
            "FVE_base":    f"{results_p1.get('fve_pretrained', float('nan')):.4f}",
            "clusters":    results_p1.get("n_clusters", "—"),
        },
        {
            "Pipeline":    "P2 F2LLM Phrase-SAE (Max-Pool phrases)",
            "NMSE":        f"{results_p2.get('NMSE', float('nan')):.4f}",
            "L0":          f"{results_p2.get('L0', float('nan')):.1f}",
            "dead%":       f"{results_p2.get('dead_pct', float('nan')):.1f}",
            "ρ_SAE":       f"{results_p2.get('rho_sae', float('nan')):.4f}",
            "silhouette":  f"{results_p2.get('silhouette', float('nan')):.4f}",
            "acc_SAE":     f"{results_p2.get('clf_acc_sae', float('nan')):.4f}",
            "acc_axes_email": f"{results_p2.get('clf_acc_email_axes', float('nan')):.4f} "
                               f"({results_p2.get('clf_n_email_classes', 0)} classes)",
            "FVE_base":    "—",
            "clusters":    results_p2.get("n_clusters", "—"),
        },
    ]
    print(pd.DataFrame(rows).to_string(index=False))

    with open(os.path.join(SAVE_DIR, "results.json"), "w") as f:
        json.dump(
            {
                "P1_Gemma3_SAE":    {k: v for k, v in results_p1.items() if not k.startswith("_")},
                "P2_F2LLM_PhSAE":  {k: v for k, v in results_p2.items()},
            },
            f, indent=2,
        )

    print("\n" + "=" * 70)
    print(f" Terminé. Répertoire de sortie : {SAVE_DIR}")
    print("=" * 70)