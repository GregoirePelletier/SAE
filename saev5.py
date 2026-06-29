"""
saev5.py — Dual Pipeline SAE (Gemma-3 + F2LLM Embedding-SAE) — v7
===================================================================
Pipeline 1 : Gemma-3 → résiduel stream (layer LAYER)
             → FrozenCoreResidualSAE ou ExtendedSAE (cœur gelé + 1024 features FR)
             → SAE.encode per token  [T, d_sae]
             → max-pool SAE acts     [d_sae]   ← dans l'espace SAE, PAS résiduel
             Visualisation UMAP, NPMI, Steering, Diffing, Tasks 1–4.

Pipeline 2 : F2LLM-v2 → embedding par phrase → SAE BatchTopK phrase-level
             → max-pool sur phrases → vecteur document  [d_sae]
             Visualisation UMAP, classification downstream.

Corrections v7 vs v6 :
  FIX 1 — SAE.from_pretrained (SAELens API correcte)
  FIX 2 — apply_chat_template retourne un tenseur ; generate(input_ids=...) 
  FIX 3 — Pipeline 2 : split_into_phrases appliqué à l'entraînement ET l'inférence
  FIX 4 — FrozenCoreResidualSAE / ExtendedSAE intégrés (slide 8)
  FIX 5 — compute_metrics expose NMSE (= 1 - FVE)
  NEW   — ρ_SAE (préservation ranking cosinus)
  NEW   — downstream_classification (sonde logistique SAE vs raw)
  NEW   — comparaison FR/EN (FVE baseline vs adapté)
  CLEAN — sae_domain_adaptation_v2 absorbé, duplication supprimée
"""

import os
import urllib3
import requests
from requests.sessions import Session

# ======================================================================
# CONFIGURATION ET PATCHS SÉCURITÉ RESEAU (CLUSTER & FRONT DGX)
# ======================================================================

# 1. Variables d'environnement de sécurité de base
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""

# 2. Patch global au niveau de l'adaptateur de Session (requests)
_old_merge_environment_settings = Session.merge_environment_settings

def patched_merge_environment_settings(self, url, proxies, stream, verify, cert):
    settings = _old_merge_environment_settings(self, url, proxies, stream, verify, cert)
    settings['verify'] = False
    return settings

Session.merge_environment_settings = patched_merge_environment_settings

# 3. Import explicite des sous-modules pour briser le lazy loading de HF
import huggingface_hub.utils
import huggingface_hub.file_download

_old_get_session = huggingface_hub.utils.get_session

def patched_get_session():
    session = _old_get_session()
    session.verify = False
    return session

# Application du patch sur les points d'entrée internes majeurs
huggingface_hub.utils.get_session = patched_get_session
huggingface_hub.file_download.get_session = patched_get_session

# 4. Suppression des alertes de sécurité redondantes (InsecureRequestWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from pathlib import Path

# 1. Forcer le mode offline pour les composants standards
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# 2. Court-circuiter l'appel HTTP obligatoire de sae_lens pour les safetensors
import sae_lens.loading.pretrained_sae_loaders as sae_loaders

def mocked_get_safetensors_tensor_shapes(url, headers=None, timeout=10):
    """Lit les shapes directement depuis le fichier config local pour éviter requests.get"""
    import os, json
    from pathlib import Path
    
    cache_dir = Path(os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-scope-2-4b-it/snapshots"))
    
    # Valeurs par défaut pour Gemma-3-4B (Residual Stream Layer 17 -> d_model=2304, d_sae=16384)
    d_model = 2560
    d_sae = 16384
    
    if cache_dir.exists():
        snapshots = list(cache_dir.iterdir())
        if snapshots:
            config_local = snapshots[0] / "resid_post/layer_17_width_16k_l0_medium/config.json"
            if config_local.exists():
                try:
                    with open(config_local, "r") as f:
                        cfg = json.load(f)
                    d_sae = cfg.get("dict_size", d_sae)
                    d_model = cfg.get("act_size", d_model)
                except Exception:
                    pass
                
    # Retourne les deux casses (minuscules ET majuscules) pour être totalement blindé
    return {
        "w_enc": [d_model, d_sae],
        "b_enc": [d_sae],
        "w_dec": [d_sae, d_model],
        "b_dec": [d_model],
        "W_enc": [d_model, d_sae],
        "B_enc": [d_sae],
        "W_dec": [d_sae, d_model],
        "B_dec": [d_model],
    }

# Injection du mock dans la librairie
sae_loaders.get_safetensors_tensor_shapes = mocked_get_safetensors_tensor_shapes

# ======================================================================
# IMPORTS APPLICATIFS STANDARDS
# ======================================================================
import gc
import json
import math
import random
import re
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix as sp_csr
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

from sae_shared import (
    ENERGY_KEYWORDS, SPORTS_KEYWORDS, SUPPORT_KEYWORDS,
    ENERGY_URL_PATTERNS, SPORTS_URL_PATTERNS, SUPPORT_URL_PATTERNS,
    prepare_domain_dataset, split_into_phrases,
    compute_metrics, compute_rho_sae,
    downstream_classification,
    diff_features, compute_npmi,
    steer_activations, steer_and_decode,
    load_and_clean_emails,
    FrozenCoreResidualSAE, ExtendedSAE,
    PhraseLevelSAE, extract_f2llm_embeddings, get_top_activating_examples,
    encode_documents_with_phrase_sae, load_or_train_sae, load_or_train,
    compute_sae_metrics,
    pool_embeddings_by_document
)

# 2. Importe la fonction Top-K depuis src/sae/batch.py
from src.sae.batch import batch_topk_encode

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

HF_TOKEN           = os.environ.get("HF_TOKEN")
SAVE_DIR           = os.environ.get("SAVE_DIR", "./results_v7/")
LOCAL_DATASET_PATH = os.environ.get(
    "LOCAL_DATASET_PATH",
    "/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
)
LOCAL_MAILS_PATH   = os.environ.get("LOCAL_MAILS_PATH", "/home/h21486/SAE/Mails.tsv")

os.makedirs(SAVE_DIR, exist_ok=True)
CACHE_DIR = os.path.join(SAVE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

USE_FINEWEB2   = True
N_TOTAL_ENERGY = int(os.environ.get("N_TOTAL_ENERGY", "2000"))
N_TOTAL_SPORTS = int(os.environ.get("N_TOTAL_SPORTS", "2000"))
N_TOTAL_SUPPORT = int(os.environ.get("N_TOTAL_SUPPORT", "2000"))
TEST_SPLIT     = float(os.environ.get("TEST_SPLIT", "0.1"))

# Pipeline 2
EMB_MODEL       = os.environ.get("EMB_MODEL", "/home/h21486/SAE/models/F2LLM-v2-80M")
MATRYOSHKA_DIM  = int(os.environ.get("MATRYOSHKA_DIM", "320"))
D_SAE           = int(os.environ.get("D_SAE", "8192"))
K_SPARSE        = int(os.environ.get("K_SPARSE", "16"))
EPOCHS          = int(os.environ.get("EPOCHS", "30"))
LR              = float(os.environ.get("LR", "5e-4"))
BATCH_TRAIN     = int(os.environ.get("BATCH_TRAIN", "256"))
MAX_PHRASES_DOC = int(os.environ.get("MAX_PHRASES_DOC", "20"))

# Pipeline 1 — FrozenCore (slide 8 : cœur gelé + 1024 features FR)
D_EXTRA         = int(os.environ.get("D_EXTRA", "1024"))
K_EXTRA         = int(os.environ.get("K_EXTRA", "32"))
EPOCHS_EXTRA    = int(os.environ.get("EPOCHS_EXTRA", "10"))
LR_EXTRA        = float(os.environ.get("LR_EXTRA", "3e-4"))
# USE_FROZEN_CORE : si True, entraîne FrozenCoreResidualSAE sur le corpus FR
# sinon utilise le SAE préentraîné brut (mode sans adaptation)
USE_FROZEN_CORE = os.environ.get("USE_FROZEN_CORE", "1").strip() in ("1", "true", "True")
# Nombre max de tokens résiduel pour entraîner l'extension (budget mémoire)
N_TOKENS_EXTRA_TRAIN = int(os.environ.get("N_TOKENS_EXTRA_TRAIN", "500000"))

# LLM Judge
N_FEATURES_TO_LABEL = int(os.environ.get("N_FEATURES_TO_LABEL", "10"))

# Modèle Gemma-3 (Pipeline 1)
MODEL_SIZE = os.environ.get("MODEL_SIZE", "4b")
if MODEL_SIZE == "4b":
    MODEL_ID   = os.environ.get("MODEL_ID", "/home/h21486/SAE/models/gemma-3-4b-it")
    RELEASE_ID = "gemma-scope-2-4b-it-res"
    SAE_ID     = "layer_17_width_16k_l0_medium"
    LAYER      = 17
elif MODEL_SIZE == "1b":
    MODEL_ID   = os.environ.get("MODEL_ID", "/home/h21486/SAE/models/gemma-3-1b-it")
    RELEASE_ID = "gemma-scope-2-1b-it-res"
    SAE_ID     = "layer_13_width_16k_l0_medium"
    LAYER      = 13
else:  # 270m
    MODEL_ID   = os.environ.get("MODEL_ID", "/home/h21486/SAE/models/gemma-3-270m")
    RELEASE_ID = "gemma-scope-2-270m-pt-res"
    SAE_ID     = "layer_12_width_16k_l0_medium"
    LAYER      = 12

# Chemin local du SAE (priorité sur le hub HuggingFace)
LOCAL_SAE_DIR  = os.environ.get("LOCAL_SAE_DIR", f"/home/h21486/SAE/saes/{RELEASE_ID}")


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DU SAE PRÉENTRAÎNÉ
# ══════════════════════════════════════════════════════════════════════════════

def load_pretrained_sae() -> SAE:
    """
    Charge le SAE préentraîné GemmaScope depuis le répertoire local si disponible,
    sinon depuis le hub HuggingFace via SAELens.
    FIX 1 : API SAELens correcte (SAE.from_pretrained, 3-tuple).
    """
    if os.path.isdir(LOCAL_SAE_DIR):
        sae_subfolder = os.path.join(LOCAL_SAE_DIR, SAE_ID)
        if os.path.isdir(sae_subfolder):
            print(f"  [SAE] Chargement local : {sae_subfolder}")
            sae = SAE.load_from_pretrained(sae_subfolder, device=DEVICE)
            return sae
    print(f"  [SAE] Hub HuggingFace : {RELEASE_ID}/{SAE_ID}")
    # SAE.from_pretrained retourne (sae, cfg_dict, log_sparsities)
    sae, _cfg, _sparsity = SAE.from_pretrained(
        release=RELEASE_ID,
        sae_id=SAE_ID,
        device=DEVICE,
    )
    return sae


# FrozenCoreResidualSAE and ExtendedSAE are implemented in src/sae/frozen_core.py
# and imported via sae_shared, so duplicate local definitions have been removed.


# ══════════════════════════════════════════════════════════════════════════════
# LLM JUDGE — ANNOTATION LOCALE (CAUSAL STREAM)
# ══════════════════════════════════════════════════════════════════════════════

def extract_causal_context(token_strings: list, target_idx: int,
                           left_window: int = 40, right_window: int = 15) -> str:
    context_tokens = []
    for idx in range(max(0, target_idx - left_window),
                     min(len(token_strings), target_idx + right_window + 1)):
        token = token_strings[idx].replace("Ġ", " ").replace("▁", " ").strip()
        context_tokens.append(f"<<{token}>>" if idx == target_idx else token)
    return " ".join(t for t in context_tokens if t)


def build_causal_highlighted_examples(f_idx: int, per_doc_token_data: list,
                                       acts: torch.Tensor, top_k: int = 6) -> list:
    f_acts = acts[:, f_idx].detach().float().numpy()
    top_doc_indices = np.argsort(f_acts)[::-1][:top_k * 2]
    examples = []
    for d_idx in top_doc_indices:
        if f_acts[d_idx] <= 1e-6 or d_idx >= len(per_doc_token_data):
            continue
        doc_data = per_doc_token_data[d_idx]
        token_acts = np.asarray(doc_data["token_sae_acts"][:, f_idx].todense()).flatten()
        if token_acts.max() <= 1e-6:
            continue
        examples.append(extract_causal_context(doc_data["token_strings"], int(token_acts.argmax())))
        if len(examples) >= top_k:
            break
    return examples


def local_gemma_judge(
    model, tokenizer,
    feature_indices: list,
    acts: torch.Tensor,
    per_doc_token_data: list = None,
    texts: list = None,
    top_k_examples: int = 6,
) -> dict:
    results = {}
    model.eval()
    for f_idx in tqdm(feature_indices, desc="LLM Judge"):
        if per_doc_token_data:
            pos_examples = build_causal_highlighted_examples(f_idx, per_doc_token_data, acts, top_k=top_k_examples)
        elif texts:
            raw = get_top_activating_examples(f_idx, acts.float(), texts, top_k=top_k_examples)
            pos_examples = [ex[:300] for ex, _ in raw]
        else:
            continue
        if not pos_examples:
            results[f_idx] = {"label": "dead_feature", "brief_description": "Aucune activation.", "score": 0}
            continue

        saved = [ex[:150].strip() + "..." for ex in pos_examples]
        formatted = "".join([f"Exemple {i+1}: {ex}\n" for i, ex in enumerate(pos_examples)])
        prompt = (
            "Tu es un chercheur expert en interprétabilité mécaniste pour EDF R&D (SEQUOIA).\n"
            "Analyse l'activation de la feature suivante au sein du flux résiduel.\n"
            "Les tokens déclencheurs sont entourés de << >>.\n"
            "ATTENTION CAUSALE : le concept se situe principalement dans le contexte EN AMONT (à gauche).\n\n"
            f"<exemples_flux_causal>\n{formatted}</exemples_flux_causal>\n\n"
            "Réponds UNIQUEMENT sous la forme d'un objet JSON valide :\n"
            "- 'label' : nom court (3 mots max, français)\n"
            "- 'brief_description' : une phrase résumant le déclencheur\n"
            "- 'score' : entier 1–5\n\n"
            '{"label": "...", "brief_description": "...", "score": 5}'
        )
        # FIX 2 : apply_chat_template retourne un tenseur, pas un dict
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs, max_new_tokens=128, do_sample=False
            )
            response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        try:
            json_str = re.search(r"\{.*\}", response, re.DOTALL).group()
            parsed = json.loads(json_str)
            parsed["saved_context_examples"] = saved
            results[f_idx] = parsed
        except Exception:
            results[f_idx] = {"label": f"Feature_{f_idx}", "brief_description": "Échec JSON.",
                               "score": -1, "saved_context_examples": saved}
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PHRASE-LEVEL SAE (Pipeline 2)
# ══════════════════════════════════════════════════════════════════════════════

# Phrase-level SAE helpers are now implemented in src/sae/phrase_sae.py
# and imported via sae_shared to avoid duplicate local definitions.


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


# ══════════════════════════════════════════════════════════════════════════════
# ACTIVATING TOKENS (Pipeline 2 — projection locale)
# ══════════════════════════════════════════════════════════════════════════════

def get_activating_tokens_for_doc(
    token_strings: list,
    token_residuals: torch.Tensor,
    sae,
    top_feature_indices: list,
    top_k_tokens: int = 2,
) -> dict:
    if hasattr(sae, "W_enc"):
        W_enc_sub = sae.W_enc[:, top_feature_indices].float()
        b_enc_sub = sae.b_enc[top_feature_indices].float()
    else:
        W_enc_sub = sae.W_enc[top_feature_indices, :].T.float()
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
    top_diffs = diff_df.head(8)
    features_desc = []
    for _, row in top_diffs.iterrows():
        features_desc.append(
            f"- Feature #{int(row['feature_id'])} ({row['feature_label']}) : "
            f"{label_a}={row['freq_A']:.3f} vs {label_b}={row['freq_B']:.3f} "
            f"(écart={row['frequency_difference']:.3f})"
        )
    prompt = (
        f"Chercheur en interprétabilité SAE EDF R&D. Corpus '{label_a}' vs '{label_b}'.\n"
        f"Features SAE les plus discriminantes :\n{chr(10).join(features_desc)}\n\n"
        "Hypothèse globale scientifique (français, 2–3 phrases) sur la divergence sémantique."
    )
    # FIX 2 : retourne un tenseur, pas un dict
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(input_ids=inputs, max_new_tokens=256, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    return response.strip()


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
    query_words = set(axis_query.lower().split())
    matched_indices = [
        f_idx for f_idx, lbl in feature_labels.items()
        if any(word in lbl.lower() for word in query_words)
    ]
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
# TÂCHE 4 : PROPERTY-BASED RETRIEVAL (Boltzmann rank-weighted)
# ══════════════════════════════════════════════════════════════════════════════

def property_based_retrieval(
    query_string: str,
    doc_acts: torch.Tensor,
    texts: list,
    feature_labels: dict,
    temperature: float = 0.2,
    top_n_results: int = 5,
) -> list:
    """
    Score Boltzmann : w_i = exp(-(rank_i / k) / T)
    Pondère les latents matchés par leur rang décroissant d'importance.
    """
    print(f"\n  [Task 4] Recherche implicite : '{query_string}'")
    query_words = set(query_string.lower().split())
    matched_latents = [
        f_idx for f_idx, lbl in feature_labels.items()
        if any(word in lbl.lower() for word in query_words)
    ]
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
    per_doc_token_data: list = None,
    activating_tokens_map: dict = None,
    feature_labels: dict = None,
) -> dict:
    import umap
    import plotly.express as px
    from sklearn.cluster import HDBSCAN
    import textwrap

    N_DOCS = len(texts)
    sae_np = sae_acts.float().detach().cpu().numpy()
    active_mask = sae_np.max(axis=0) > 0
    sae_active = sae_np[:, active_mask]
    n_active = int(active_mask.sum())
    active_indices = np.where(active_mask)[0].tolist()
    print(f"  Features actives (UMAP) : {n_active} / {sae_acts.shape[1]}")

    reducer = umap.UMAP(
        n_components=2, metric="cosine",
        n_neighbors=min(30, max(2, N_DOCS - 1)),
        min_dist=0.1, random_state=SEED,
    )
    coords = reducer.fit_transform(sae_active)

    min_cs = max(2, N_DOCS // 15)
    clusterer = HDBSCAN(min_cluster_size=min_cs, min_samples=max(1, min_cs // 2))
    clusters = clusterer.fit_predict(coords)

    df = pd.DataFrame({
        "x": coords[:, 0], "y": coords[:, 1],
        "cluster_raw": clusters,
        "label": labels if (labels and len(labels) == N_DOCS) else ["Unknown"] * N_DOCS,
        "doc_idx": np.arange(N_DOCS),
    }).sort_values("cluster_raw").reset_index(drop=True)
    df["cluster_id"] = df["cluster_raw"].apply(lambda c: f"Cluster {c}" if c != -1 else "Bruit (-1)")

    # Signatures sémantiques par cluster
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

    custom_hover, top_feats_html, sig_col = [], [], []
    for _, row in df.iterrows():
        i = int(row["doc_idx"])
        c_raw = int(row["cluster_raw"])
        sig_col.append(cluster_signatures.get(c_raw, ""))

        r_acts = sae_acts[i]
        top_vals, top_ids = r_acts.topk(min(3, r_acts.shape[0]))
        feats_html = []
        best_feat = top_ids[0].item() if top_vals[0] > 1e-6 else -1
        for j in range(len(top_ids)):
            v = top_vals[j].item()
            if v <= 1e-6: break
            f_idx = top_ids[j].item()
            f_label = feature_labels.get(f_idx, f"F{f_idx}") if feature_labels else f"F{f_idx}"
            tok_str = ""
            if per_doc_token_data and i < len(per_doc_token_data):
                td = per_doc_token_data[i]
                acts_arr = np.asarray(td["token_sae_acts"][:, f_idx].todense()).flatten()
                high = np.where(acts_arr > acts_arr.max() * 0.65)[0]
                detected = list(dict.fromkeys([
                    td["token_strings"][t].replace("Ġ", " ").replace("▁", " ").strip()
                    for t in high if len(td["token_strings"][t].strip()) > 1
                ]))[:3]
                if detected: tok_str = f" <i>«{', '.join(detected)}»</i>"
            elif activating_tokens_map and i in activating_tokens_map:
                toks = activating_tokens_map[i].get(f_idx, [])
                if toks: tok_str = f" <i>«{toks[0][0].strip()}»</i>"
            intensity = min(5, max(1, int((v / 15.0) * 5)))
            bar = f"<span style='color:#00cc96;'>{'█'*intensity}{'▒'*(5-intensity)}</span>"
            feats_html.append(f"<b>{f_label}</b> ({v:.1f}) {bar}{tok_str}")
        top_feats_html.append("<br>".join(feats_html) or "<i>Aucune feature active</i>")

        if best_feat != -1 and per_doc_token_data and i < len(per_doc_token_data):
            td = per_doc_token_data[i]
            acts_arr = np.asarray(td["token_sae_acts"][:, best_feat].todense()).flatten()
            raw_text = extract_causal_context(td["token_strings"], int(acts_arr.argmax()))
        else:
            raw_text = texts[i][:400]

        wrapped = "<br>".join(textwrap.wrap(raw_text, width=80))
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

    fig = px.scatter(df, x="x", y="y", color="cluster_id",
                     category_orders={"cluster_id": sorted(df["cluster_id"].unique())})
    fig.update_traces(
        marker=dict(size=7, opacity=0.8),
        hovertemplate=(
            "<b>%{customdata[2]}</b> (label: %{customdata[3]})<br>"
            "<span style='color:#1f77b4'><b>Signature cluster :</b> %{customdata[4]}</span><br><br>"
            "<b>Top Features :</b><br>%{customdata[1]}<br><br>"
            "<b>Contexte :</b><br>%{customdata[0]}<extra></extra>"
        ),
        customdata=np.stack((
            df["text_preview"].values, df["top_features"].values,
            df["cluster_id"].values, df["label"].values,
            df["cluster_signature"].values,
        ), axis=-1),
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
    email_texts: list = None,
    email_labels: list = None,
) -> dict:
    print("\n" + "=" * 70)
    print(" PIPELINE 1 : GEMMA-3 → MAX-POOL SAE ACTS")
    print("=" * 70)

    email_texts = email_texts or []
    email_labels = email_labels or []
    all_texts = train_texts + test_texts + email_texts

    # ─── Chargement SAE préentraîné ───────────────────────────────────────────
    pretrained_sae = load_pretrained_sae()
    pretrained_sae = pretrained_sae.to(DEVICE).to(torch.bfloat16).eval()
    pretrained_sae.requires_grad_(False)
    d_core = pretrained_sae.cfg.d_sae

    # Calcul de la dimension totale attendue en cas d'extension
    d_total_expected = d_core + D_EXTRA if USE_FROZEN_CORE else d_core

    cache_acts_path      = os.path.join(CACHE_DIR, "p1_all_doc_acts.pt")
    cache_token_path     = os.path.join(CACHE_DIR, "p1_per_doc_token_data.pkl")
    cache_residuals_path = os.path.join(CACHE_DIR, "p1_raw_residuals.pt")

    if os.path.exists(cache_acts_path) and os.path.exists(cache_token_path):
        print("  [P1] Restauration cache...")
        all_doc_sae_acts = torch.load(cache_acts_path, map_location="cpu", weights_only=True)
        with open(cache_token_path, "rb") as f:
            per_doc_token_data = pickle.load(f)
        
        # Validation de la dimension des matrices creuses restaurées du cache
        if len(per_doc_token_data) > 0:
            current_sparse_dim = per_doc_token_data[0]["token_sae_acts"].shape[1]
            if current_sparse_dim != d_total_expected:
                print(f"  [P1] Incohérence de dimension détectée dans le cache ({current_sparse_dim} vs {d_total_expected}). Re-extraction forcée.")
                _need_residuals = True
            else:
                _need_residuals = USE_FROZEN_CORE and not os.path.exists(cache_residuals_path)
        else:
            _need_residuals = True
    else:
        _need_residuals = True

    if _need_residuals or not (os.path.exists(cache_acts_path) and os.path.exists(cache_token_path)):
        print(f"  [P1] Extraction activations Gemma-3 ({MODEL_ID}, layer {LAYER})...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, token=HF_TOKEN, trust_remote_code=True, local_files_only=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        llm = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE,
            token=HF_TOKEN, trust_remote_code=True, local_files_only=True
        ).eval()

        all_doc_sae_acts = []
        per_doc_token_data = []
        raw_residuals_list = []   # pour l'entraînement FrozenCore
        n_residuals_collected = 0

        with torch.no_grad():
            for i in tqdm(range(0, len(all_texts), 4), desc="Extraction P1"):
                batch = all_texts[i: i + 4]
                inputs = tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=512,
                ).to(DEVICE)
                outputs = llm(**inputs, output_hidden_states=True)
                acts_raw = outputs.hidden_states[LAYER].detach().to(torch.bfloat16)

                # Gemma Scope s'attend à des activations non-scalées ou normalisées via RMSNorm
                # On applique la RMSNorm locale pour s'aligner sur la distribution de Gemma Scope
                epsilon = 1e-6
                rms = torch.rsqrt(acts_raw.pow(2).mean(dim=-1, keepdim=True) + epsilon)
                acts = acts_raw * rms
                mask = inputs["attention_mask"].bool()

                for b in range(acts.shape[0]):
                    valid_ids = inputs["input_ids"][b, mask[b]]
                    valid_toks = acts[b, mask[b]]
                    special_mask = torch.isin(
                        valid_ids, torch.tensor(tokenizer.all_special_ids).to(DEVICE)
                    )
                    keep = ~special_mask
                    if keep.sum() == 0:
                        keep = torch.ones_like(keep, dtype=torch.bool)
                    filtered = valid_toks[keep]
                    filtered_ids = valid_ids[keep]

                    # SAE encode per token → [T, d_core]
                    token_sae_acts = pretrained_sae.encode(filtered)
                    
                    # Alignement de la dimension de la matrice pour le stockage par token
                    if USE_FROZEN_CORE:
                        # Si le Frozen Core est actif, on pré-alloue l'espace des 1024 features supplémentaires (FR)
                        # remplies de zéros, pour éviter les futurs crashs d'indexation SciPy lors du Judge.
                        T = token_sae_acts.shape[0]
                        extra_zeros = torch.zeros((T, D_EXTRA), dtype=token_sae_acts.dtype, device=token_sae_acts.device)
                        token_sae_acts_padded = torch.cat([token_sae_acts, extra_zeros], dim=-1)
                        doc_sae_vec = token_sae_acts_padded.max(dim=0).values
                        
                        per_doc_token_data.append({
                            "token_strings": tokenizer.convert_ids_to_tokens(filtered_ids.tolist()),
                            "token_sae_acts": sp_csr(token_sae_acts_padded.float().cpu().numpy(), shape=(T, d_total_expected)),
                        })
                    else:
                        doc_sae_vec = token_sae_acts.max(dim=0).values
                        per_doc_token_data.append({
                            "token_strings": tokenizer.convert_ids_to_tokens(filtered_ids.tolist()),
                            "token_sae_acts": sp_csr(token_sae_acts.float().cpu().numpy(), shape=(token_sae_acts.shape[0], d_core)),
                        })

                    all_doc_sae_acts.append(doc_sae_vec.cpu())

                    # Collecte résidus pour FrozenCore (seulement docs train)
                    if USE_FROZEN_CORE and n_residuals_collected < N_TOKENS_EXTRA_TRAIN:
                        raw_residuals_list.append(filtered.cpu())
                        n_residuals_collected += filtered.shape[0]

        all_doc_sae_acts = torch.stack(all_doc_sae_acts)
        torch.save(all_doc_sae_acts, cache_acts_path)
        with open(cache_token_path, "wb") as f:
            pickle.dump(per_doc_token_data, f)

        if USE_FROZEN_CORE and raw_residuals_list:
            raw_residuals = torch.cat(raw_residuals_list, dim=0)[:N_TOKENS_EXTRA_TRAIN]
            torch.save(raw_residuals, cache_residuals_path)
            print(f"  [P1] Résidus bruts cachés : {raw_residuals.shape}")
            del raw_residuals_list
        del llm, tokenizer
        gc.collect(); torch.cuda.empty_cache()

    # ─── FrozenCore / ExtendedSAE (slide 8 : cœur gelé + 1024 features FR) ──
    d_total = d_core   # par défaut : SAE prétrained seul
    active_sae = pretrained_sae   # SAE utilisé pour l'inférence

    if USE_FROZEN_CORE:
        frozen_core_path = os.path.join(SAVE_DIR, f"p1_frozen_core_d{D_EXTRA}_k{K_EXTRA}.pt")
        if os.path.exists(frozen_core_path):
            print(f"  [P1] Chargement FrozenCoreResidualSAE : {frozen_core_path}")
            ext_sae = ExtendedSAE(pretrained_sae, d_extra=D_EXTRA, k_extra=K_EXTRA).to(DEVICE).to(torch.bfloat16)
            ckpt = torch.load(frozen_core_path, map_location=DEVICE, weights_only=False)
            ext_sae.load_state_dict(ckpt["state_dict"])
        else:
            if os.path.exists(cache_residuals_path):
                raw_residuals = torch.load(cache_residuals_path, weights_only=True)
            else:
                print("  [P1] WARN : résidus introuvables, FrozenCore désactivé.")
                raw_residuals = None

            if raw_residuals is not None:
                print(f"  [P1] Entraînement ExtendedSAE sur {len(raw_residuals)} tokens résidus...")
                # Calcule résidus e = x - core_out pour initialisation PCA
                with torch.no_grad():
                    sample = raw_residuals[:min(8192, len(raw_residuals))].to(DEVICE).to(torch.bfloat16)
                    core_acts = pretrained_sae.encode(sample)
                    core_out  = pretrained_sae.decode(core_acts)
                    domain_residuals_cpu = (sample - core_out).cpu().float()
                    del sample, core_acts, core_out
                    gc.collect(); torch.cuda.empty_cache()

                ext_sae = ExtendedSAE(
                    pretrained_sae, d_extra=D_EXTRA, k_extra=K_EXTRA,
                    domain_residuals=domain_residuals_cpu
                ).to(DEVICE).to(torch.bfloat16)

                # Entraînement uniquement sur les poids extra
                ext_sae, history_ext = load_or_train(
                    model=ext_sae, model_name="p1_extended_sae",
                    acts_train=raw_residuals,
                    epochs=EPOCHS_EXTRA, lr=LR_EXTRA,
                    save_dir=SAVE_DIR, device=DEVICE,
                )
                ckpt = {"state_dict": {k: v.cpu() for k, v in ext_sae.state_dict().items()},
                        "config": {"d_extra": D_EXTRA, "k_extra": K_EXTRA, "layer": LAYER}}
                torch.save(ckpt, frozen_core_path)
                print(f"  [P1] ExtendedSAE sauvegardé : {frozen_core_path}")
                del raw_residuals, domain_residuals_cpu
                gc.collect(); torch.cuda.empty_cache()
            else:
                ext_sae = None

        if ext_sae is not None:
            print("  [P1] Re-encodage et mise à jour dynamique des activations d'ExtendedSAE...")
            cache_acts_ext = os.path.join(CACHE_DIR, f"p1_all_doc_acts_ext_d{D_EXTRA}.pt")
            
            if os.path.exists(cache_acts_ext):
                all_doc_sae_acts = torch.load(cache_acts_ext, map_location="cpu", weights_only=True)
                # Recalculer ou re-charger per_doc_token_data n'est pas requis si déjà injecté,
                # mais on s'assure ici que la matrice creuse contient les vraies valeurs entraînées.
            else:
                ext_sae.eval()
                new_acts = []
                with torch.no_grad():
                    for i in tqdm(range(len(per_doc_token_data)), desc="Calcul des activations ExtendedSAE"):
                        # Extraction des features de base document-level
                        core_vec = all_doc_sae_acts[i].unsqueeze(0).to(DEVICE).to(torch.bfloat16)
                        x_approx = pretrained_sae.decode(core_vec)
                        
                        # Calcul des activations pour l'extension (FR)
                        extra_acts = ext_sae._encode_extra_acts(x_approx)
                        full_vec = torch.cat([core_vec, extra_acts], dim=-1).cpu().squeeze(0)
                        new_acts.append(full_vec)
                        
                        # Ré-injection chirurgicale des activations d'extension au niveau token (matrice creuse)
                        # pour que le LLM Judge puisse inspecter les features 16384+ sans IndexError
                        td = per_doc_token_data[i]
                        dense_token_acts = torch.from_numpy(td["token_sae_acts"].todense()).to(DEVICE).to(torch.bfloat16)
                        
                        core_out_tokens = pretrained_sae.decode(dense_token_acts[:, :d_core])
                        # À ce stade, dense_token_acts contient l'extraction originale complète (core)
                        # Si vous n'avez plus l'activation brute 'acts' dans le scope du cache, le vrai résiduel
                        # par token ne peut pas être reconstruit à partir de core_out seul. 
                        # Le moyen le plus propre est de calculer le résiduel d'après l'approximation du document :
                        residual_tokens = x_approx - core_out_tokens
                        token_extra_acts = ext_sae._encode_extra_acts(residual_tokens)
                        
                        # Concaténation et reconstruction de la structure SciPy
                        full_token_acts = torch.cat([dense_token_acts[:, :d_core], token_extra_acts], dim=-1)
                        td["token_sae_acts"] = sp_csr(full_token_acts.float().cpu().numpy(), shape=(full_token_acts.shape[0], d_total_expected))

                all_doc_sae_acts = torch.stack(new_acts)
                torch.save(all_doc_sae_acts, cache_acts_ext)
                
                # Mise à jour du cache token avec les valeurs d'extensions réelles
                with open(cache_token_path, "wb") as f:
                    pickle.dump(per_doc_token_data, f)
                    
            d_total = d_core + D_EXTRA
            active_sae = ext_sae
            print(f"  [P1] Dimension SAE étendue : {d_core} core + {D_EXTRA} extra = {d_total}")

    # ─── Splits train / test / email ─────────────────────────────────────────
    n_train = len(train_texts)
    n_test  = len(test_texts)
    train_doc_acts = all_doc_sae_acts[:n_train]
    test_doc_acts  = all_doc_sae_acts[n_train: n_train + n_test]
    email_doc_acts = all_doc_sae_acts[n_train + n_test:]
    train_token_data = per_doc_token_data[:n_train]
    test_token_data  = per_doc_token_data[n_train: n_train + n_test]
    email_token_data = per_doc_token_data[n_train + n_test:]

    # ─── LLM Judge ────────────────────────────────────────────────────────────
    top_feat_indices = train_doc_acts.float().mean(dim=0).topk(N_FEATURES_TO_LABEL).indices.tolist()
    judge_cache = os.path.join(CACHE_DIR, "p1_feature_labels.json")
    if os.path.exists(judge_cache):
        with open(judge_cache, "r", encoding="utf-8") as f:
            label_map_data = json.load(f)
    else:
        j_tok = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True, local_files_only=True)
        j_llm = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE,
            token=HF_TOKEN, trust_remote_code=True, local_files_only=True
        ).eval()
        
        with torch.no_grad():
            label_map_data = local_gemma_judge(
                model=j_llm, tokenizer=j_tok, feature_indices=top_feat_indices,
                acts=train_doc_acts, per_doc_token_data=train_token_data
            )
            
        with open(judge_cache, "w", encoding="utf-8") as f:
            json.dump(label_map_data, f, indent=2, ensure_ascii=False)
        del j_llm, j_tok; gc.collect(); torch.cuda.empty_cache()

    label_map_p1 = {int(idx): entry.get("label", f"F{idx}") for idx, entry in label_map_data.items()}

    # ─── UMAP FineWeb-2 ───────────────────────────────────────────────────────
    umap_res_test = analyze_with_umap(
        texts=test_texts, sae_acts=test_doc_acts, labels=test_labels,
        filename="umap_pipeline1_llm_per_token.html",
        title=f"Pipeline 1: Gemma-3 L{LAYER} → Max-Pool SAE Acts (FineWeb-2)",
        per_doc_token_data=test_token_data, feature_labels=label_map_p1,
    )
    if email_texts:
        analyze_with_umap(
            texts=email_texts, sae_acts=email_doc_acts, labels=email_labels,
            filename="umap_pipeline1_emails.html",
            title=f"Pipeline 1: Gemma-3 L{LAYER} → Max-Pool SAE Acts (EDF Mails)",
            per_doc_token_data=email_token_data, feature_labels=label_map_p1,
        )

    # ─── Tâche 1 : Diffing + hypothèse LLM ───────────────────────────────────
    energy_mask = np.array([l == "energy" for l in train_labels])
    sports_mask  = np.array([l == "sports"  for l in train_labels])
    diff_hypothesis = "Aucun écart mesurable."
    if energy_mask.sum() > 0 and sports_mask.sum() > 0:
        diff_df = diff_features(
            train_doc_acts[torch.from_numpy(energy_mask)].float(),
            train_doc_acts[torch.from_numpy(sports_mask)].float(),
            feature_labels=label_map_p1,
        )
        diff_df.to_csv(os.path.join(SAVE_DIR, "p1_diff_energy_sports.csv"), index=False)
        j_tok = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True, local_files_only=True)
        j_llm = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE,
            token=HF_TOKEN, trust_remote_code=True, local_files_only=True
        ).eval()
        with torch.no_grad():
            diff_hypothesis = generate_llm_diff_hypothesis(j_llm, j_tok, diff_df, "Énergie", "Sports")
        print(f"  [Task 1] Hypothèse LLM :\n  {diff_hypothesis}\n")
        del j_llm, j_tok; gc.collect(); torch.cuda.empty_cache()

    # ─── Tâche 2 : NPMI ───────────────────────────────────────────────────────
    npmi_mat = compute_npmi(test_doc_acts)
    torch.save(npmi_mat, os.path.join(CACHE_DIR, "p1_npmi.pt"))

    # ─── Tâche 3 : Targeted Clustering ───────────────────────────────────────
    targeted_clustering_by_axis(
        texts=test_texts, sae_acts=test_doc_acts, labels=test_labels,
        feature_labels=label_map_p1, axis_query="énergie électrique"
    )

    # ─── Tâche 4 : Property-Based Retrieval ──────────────────────────────────
    results_retrieval = property_based_retrieval(
        "électrique nucléaire réseau", test_doc_acts, test_texts, label_map_p1
    )
    for rank, (doc, score) in enumerate(results_retrieval):
        print(f"    Rang {rank+1} (Boltzmann={score:.4f}) : {doc[:100]}...")

    # ─── Métriques ────────────────────────────────────────────────────────────
    silhouette = compute_silhouette(test_doc_acts, test_labels)
    l0_mean    = (test_doc_acts > 1e-6).float().sum(dim=-1).mean().item()
    dead_pct   = (test_doc_acts.sum(dim=0) == 0).float().mean().item() * 100
    
    with torch.no_grad():
        rho_sae = compute_rho_sae(active_sae, test_doc_acts,
                                  n_sample=500, is_saelens=not USE_FROZEN_CORE, device=DEVICE)

    # ─── Comparaison FR/EN (perspective slide 19) ────────────────────────────
    print("\n  [FR/EN] Comparaison FVE baseline sur sous-ensemble train...")
    with torch.no_grad():
        metrics_pretrained = compute_metrics(
        pretrained_sae, all_doc_sae_acts[:n_train, :d_core].float(),
        is_saelens=True, device=DEVICE
    )
    print(f"  FVE (prétrained, train FR) = {metrics_pretrained['FVE']:.4f} | "
          f"NMSE = {metrics_pretrained['NMSE']:.4f}")
          
    if USE_FROZEN_CORE and active_sae is not pretrained_sae:
        with torch.no_grad():
            metrics_ext = compute_metrics(active_sae, all_doc_sae_acts[:n_train].float(),
                                          is_saelens=False, device=DEVICE)
        print(f"  FVE (ExtendedSAE, train FR) = {metrics_ext['FVE']:.4f} | "
              f"NMSE = {metrics_ext['NMSE']:.4f} | "
              f"ΔFVE = {metrics_ext['FVE'] - metrics_pretrained['FVE']:+.4f}")

    # ─── Downstream classification ────────────────────────────────────────────
    print("\n  [Downstream P1] Sonde logistique sur SAE activations...")
    en_mask = torch.from_numpy(energy_mask)
    sp_mask = torch.from_numpy(sports_mask)
    if en_mask.sum() > 0 and sp_mask.sum() > 0:
        try:
            clf_results = downstream_classification(
                acts_by_label={
                    "energy": train_doc_acts[en_mask],
                    "sports": train_doc_acts[sp_mask],
                }
            )
        except Exception as e:
            print(f"  [Downstream P1] WARN: Classification failed: {e}")
            clf_results = {}
    else:
        print(f"  [Downstream P1] Insufficient samples: energy={en_mask.sum()}, sports={sp_mask.sum()}")
        clf_results = {}

    return {
        "L0": l0_mean, "dead_pct": dead_pct, "silhouette": silhouette,
        "rho_sae": rho_sae,
        "n_clusters": umap_res_test["n_clusters"],
        "active_features": umap_res_test["n_active"],
        "diff_hypothesis": diff_hypothesis,
        "clf_acc_sae": clf_results.get("acc_sae", float("nan")),
        "fve_pretrained": metrics_pretrained["FVE"],
        "_test_doc_acts": test_doc_acts,
        "_test_token_data": test_token_data,
        "_label_map": label_map_p1,
        "_active_sae": active_sae,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE 2 — F2LLM PHRASE-LEVEL SAE
# ══════════════════════════════════════════════════════════════════════════════

def run_f2llm_pipeline(
    train_texts: list,
    train_labels: list,
    test_texts: list,
    test_labels: list,
    email_texts: list = None,
    email_labels: list = None,
) -> dict:
    print("\n" + "=" * 70)
    print(" PIPELINE 2 : F2LLM-v2 PHRASE-LEVEL SAE → MAX-POOL DOCUMENT")
    print("=" * 70)

    email_texts  = email_texts or []
    email_labels = email_labels or []

    # FIX 3 : split_into_phrases sur TRAIN ET TEST
    # Entraînement sur phrases (représentation sémantique fine)
    train_phrases, train_p2d = split_into_phrases(train_texts, max_phrases_per_doc=MAX_PHRASES_DOC)
    print(f"  Train : {len(train_texts)} docs → {len(train_phrases)} phrases")

    train_phrase_emb, d_in = extract_f2llm_embeddings(
    train_phrases, emb_model=EMB_MODEL, matryoshka_dim=MATRYOSHKA_DIM,
    max_length=128,
    cache_path=os.path.join(CACHE_DIR, f"train_phrase_emb_dim{MATRYOSHKA_DIM}"),
    )

    idx = torch.randperm(len(train_phrase_emb), generator=torch.Generator().manual_seed(SEED))
    split = int(len(idx) * 0.85)
    emb_train_split = train_phrase_emb[idx[:split]]
    emb_eval_split  = train_phrase_emb[idx[split:]]

    sae_path = os.path.join(SAVE_DIR, f"p2_sae_dim{d_in}_d{D_SAE}_k{K_SPARSE}.pt")
    sae, history = load_or_train_sae(
        d_in=d_in, d_sae=D_SAE, k=K_SPARSE,
        embeddings=emb_train_split, save_path=sae_path,
        epochs=EPOCHS, lr=LR,
    )
    m_eval = compute_sae_metrics(sae, emb_eval_split)
    rho_sae_p2 = compute_rho_sae(sae, emb_eval_split, n_sample=500, device=DEVICE)
    del emb_train_split, emb_eval_split; gc.collect(); torch.cuda.empty_cache()

    # Test : split en phrases → encode → max-pool par doc
    test_phrases, test_p2d_list = split_into_phrases(test_texts, max_phrases_per_doc=MAX_PHRASES_DOC)
    print(f"  Test  : {len(test_texts)} docs → {len(test_phrases)} phrases")
    
    test_phrase_emb, _ = extract_f2llm_embeddings(
    test_phrases, emb_model=EMB_MODEL, matryoshka_dim=MATRYOSHKA_DIM,
    max_length=128,
    cache_path=os.path.join(CACHE_DIR, f"test_phrase_emb_dim{MATRYOSHKA_DIM}"),
    )
    test_p2d_arr = np.array(test_p2d_list)
    doc_acts = encode_documents_with_phrase_sae(
        n_docs=len(test_texts), sae=sae,
        phrase_embeddings=test_phrase_emb, phrase_to_doc=test_p2d_arr,
    )

    # Email (si disponible)
    email_doc_acts = None
    if email_texts:
        email_phrases, email_p2d_list = split_into_phrases(email_texts, max_phrases_per_doc=MAX_PHRASES_DOC)
        email_phrase_emb, _ = extract_f2llm_embeddings(
        email_phrases, emb_model=EMB_MODEL, matryoshka_dim=MATRYOSHKA_DIM,
        max_length=128,
        cache_path=os.path.join(CACHE_DIR, f"email_phrase_emb_dim{MATRYOSHKA_DIM}"),
        )
        email_p2d_arr = np.array(email_p2d_list)
        email_doc_acts = encode_documents_with_phrase_sae(
            n_docs=len(email_texts), sae=sae,
            phrase_embeddings=email_phrase_emb, phrase_to_doc=email_p2d_arr,
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
            MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE, local_files_only=True
        ).eval()
        feature_labels_p2 = local_gemma_judge(
            model=j_llm, tokenizer=j_tok, feature_indices=top_feat_indices,
            acts=doc_acts, texts=test_texts,
        )
        del j_llm, j_tok; gc.collect(); torch.cuda.empty_cache()
        with open(judge_cache, "w", encoding="utf-8") as f:
            json.dump(feature_labels_p2, f, indent=2, ensure_ascii=False)

    label_map_p2 = {int(idx): entry.get("label", f"F{idx}") for idx, entry in feature_labels_p2.items()}

    # ─── Activating phrases par doc ───────────────────────────────────────────
    activating_phrases_map = {}
    sae.eval()
    with torch.no_grad():
        for doc_idx in range(len(test_texts)):
            phrase_indices = np.where(test_p2d_arr == doc_idx)[0].tolist()
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

    # ─── UMAP ─────────────────────────────────────────────────────────────────
    umap_res_test = analyze_with_umap(
        texts=test_texts, sae_acts=doc_acts, labels=test_labels,
        filename="umap_pipeline2_f2llm_phrases.html",
        title="Pipeline 2 : F2LLM-v2 Phrase SAE → Max-Pool Document (FineWeb-2)",
        activating_tokens_map=activating_phrases_map, feature_labels=label_map_p2,
    )
    if email_texts and email_doc_acts is not None:
        analyze_with_umap(
            texts=email_texts, sae_acts=email_doc_acts, labels=email_labels,
            filename="umap_pipeline2_emails.html",
            title="Pipeline 2 : F2LLM-v2 Phrase SAE → Max-Pool Document (EDF Mails)",
            feature_labels=label_map_p2,
        )

    # ─── Downstream classification ────────────────────────────────────────────
    print("\n  [Downstream P2] Sonde logistique sur SAE activations...")
    energy_mask_test = np.array([l == "energy" for l in test_labels])
    sports_mask_test  = np.array([l == "sports"  for l in test_labels])
    if energy_mask_test.sum() > 0 and sports_mask_test.sum() > 0:
        try:
            # Pool phrase embeddings to document level for consistency with doc_acts
            from sae_shared import pool_embeddings_by_document
            test_phrase_emb_pooled = pool_embeddings_by_document(
                test_phrase_emb, test_p2d_arr, n_docs=len(test_texts)
            )
            clf_results_p2 = downstream_classification(
                acts_by_label={
                    "energy": doc_acts[torch.from_numpy(energy_mask_test)],
                    "sports": doc_acts[torch.from_numpy(sports_mask_test)],
                },
                raw_emb_by_label={
                    "energy": test_phrase_emb_pooled[torch.from_numpy(energy_mask_test)],
                    "sports": test_phrase_emb_pooled[torch.from_numpy(sports_mask_test)],
                }
            )
        except Exception as e:
            print(f"  [Downstream P2] WARN: Classification failed: {e}")
            clf_results_p2 = {}
    else:
        print(f"  [Downstream P2] Insufficient samples: energy={energy_mask_test.sum()}, sports={sports_mask_test.sum()}")
        clf_results_p2 = {}

    silhouette_p2 = compute_silhouette(doc_acts, test_labels)
    del sae, doc_acts, test_phrase_emb, train_phrase_emb
    if email_doc_acts is not None: del email_doc_acts
    gc.collect(); torch.cuda.empty_cache()

    return {
        **m_eval,
        "rho_sae": rho_sae_p2,
        "silhouette": silhouette_p2,
        "n_clusters": umap_res_test["n_clusters"],
        "active_features": umap_res_test["n_active"],
        "clf_acc_sae": clf_results_p2.get("acc_sae", float("nan")),
        "clf_acc_raw": clf_results_p2.get("acc_raw", float("nan")),
        "clf_delta":   clf_results_p2.get("delta_acc", float("nan")),
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

    rng = np.random.default_rng(SEED)

    def _split(texts, label, frac=TEST_SPLIT):
        n = len(texts)
        idx = rng.permutation(n)
        n_test = max(1, int(n * frac))
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        return (
            [texts[i] for i in train_idx], [label] * (n - n_test),
            [texts[i] for i in test_idx],  [label] * n_test,
        )

    en_tr, en_tr_lbl, en_te, en_te_lbl = _split(energy_texts, "energy")
    sp_tr, sp_tr_lbl, sp_te, sp_te_lbl = _split(sports_texts, "sports")
    su_tr, su_tr_lbl, su_te, su_te_lbl = _split(support_texts, "support")
    train_texts  = en_tr  + sp_tr  + su_tr
    train_labels = en_tr_lbl + sp_tr_lbl + su_tr_lbl
    test_texts   = en_te  + sp_te  + su_te
    test_labels  = en_te_lbl + sp_te_lbl + su_te_lbl
    print(f"Train : {len(train_texts)} chunks | Test : {len(test_texts)} chunks")

    email_texts, email_labels = load_and_clean_emails(LOCAL_MAILS_PATH)
    if not email_texts:
        print("  Fallback emails synthétiques.")
        email_texts = [
            "Bonjour, je conteste ma facture d'électricité Linky, hausse injustifiée.",
            "Merci de planifier l'installation de mon compteur de raccordement électrique.",
            "Coupure réseau dans notre rue depuis 2 heures. Envoyez un technicien.",
        ]
        email_labels = ["Reclamation_Facturation", "Mise_En_Service", "Urgence_Technique"]

    results_p1 = run_llm_max_pool_pipeline(
        train_texts, train_labels, test_texts, test_labels, email_texts, email_labels
    )
    results_p2 = run_f2llm_pipeline(
        train_texts, train_labels, test_texts, test_labels, email_texts, email_labels
    )
    run_steering_demo(results_p1)

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
            "FVE_base":    "—",
            "clusters":    results_p2.get("n_clusters", "—"),
        },
    ]
    print(pd.DataFrame(rows).to_string(index=False))

    with open(os.path.join(SAVE_DIR, "results_v7.json"), "w") as f:
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