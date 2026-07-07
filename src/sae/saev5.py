"""
saev5.py — Dual Pipeline SAE (Gemma-3 + F2LLM Embedding-SAE) — v8
===================================================================
Corrections apportées en v8 :
  FIX 1 — Normalisation RMS supprimée : GemmaScope s'applique sur le residual stream brut.
  FIX 2 — Dette de design résolue : stockage des activations réelles (raw_acts) pour le résidu exact.
  FIX 3 — Sélection des features par fréquence d'activation pour éviter les biais.
  FIX 4 — rho_sae et FVE calculés au niveau Token avec des activations réelles.
  FIX 5 — W_enc dispatch consolidé pour tous les types d'encodeurs.
  FIX 6 — NameError sur test_token_data résolue.
  FIX 7 — load_pretrained_sae : Résolution du triplet de retour et unification des routes de chargement locales.
  FIX 8 — load_or_train_extended_sae : Résolution de la signature erronée d'import pour Pipeline 1.
  FIX 9 — local_gemma_judge : Définition de la fonction de labellisation locale manquante pour le Pipeline 2.
"""

import os
import urllib3
import requests
import glob
import pickle
import json
import math
import random
import re
from requests.sessions import Session
from sae_lens.registry import SAE_CLASS_REGISTRY
import gc

# Compatibilité Gemma Scope 2
if "jump_relu" not in SAE_CLASS_REGISTRY and "jumprelu" in SAE_CLASS_REGISTRY:
    SAE_CLASS_REGISTRY["jump_relu"] = SAE_CLASS_REGISTRY["jumprelu"]

# ======================================================================
# CONFIGURATION ET PATCHS SÉCURITÉ RESEAU (CLUSTER & FRONT DGX)
# ======================================================================

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
    d_model = 4096 if MODEL_SIZE == "12b" else 2560
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

from sae_shared import (
    ENERGY_KEYWORDS, SPORTS_KEYWORDS, SUPPORT_KEYWORDS,
    ENERGY_URL_PATTERNS, SPORTS_URL_PATTERNS, SUPPORT_URL_PATTERNS,
    prepare_domain_dataset, split_into_phrases,
    compute_metrics, compute_rho_sae,
    downstream_classification,
    steer_activations, steer_and_decode,
    load_and_clean_emails,
    FrozenCoreResidualSAE, ExtendedSAE,
    PhraseLevelSAE, extract_f2llm_embeddings,
    encode_documents_with_phrase_sae, load_or_train_sae,
    compute_sae_metrics,
    pool_embeddings_by_document,
    load_or_train_extended_sae  # Import de la fonction d'apprentissage corrigée
)

from src.sae.judge import (
    extract_causal_context, build_feature_examples_with_control,
    feature_selection_by_magnitude, odd_one_out_judge, _apply_chat_and_extract,
    local_gemma_judge,
)

from src.sae.batch import batch_topk_encode
try:
    from src.analysis.cooccurrence import compute_npmi, corpus_diff_stats
except ImportError:
    from cooccurrence import compute_npmi, corpus_diff_stats

# Labels GemmaScope officiels (Neuronpedia) — cache JSON obligatoire (cluster offline)
try:
    from src.sae.neuronpedia_labels import merge_with_judge_labels  # noqa: F401
except ImportError:
    merge_with_judge_labels = None

import ctypes

def _trim_host_memory():
    """Rend les arènes glibc libérées à l'OS après teardown d'un gros modèle.
    Sans cela, le RSS croît de façon monotone à chaque cycle load/unload
    (fragmentation malloc) → OOM SLURM même si Python a bien libéré."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
try:
    from src.storage.fragment_store import (
        save_fragment, load_fragment, fragment_exists, list_fragment_ids,
        feature_column, doc_maxpool, decode_core_sparse, merge_extra,
    )
except ImportError:
    from fragment_store import (
        save_fragment, load_fragment, fragment_exists, list_fragment_ids,
        feature_column, doc_maxpool, decode_core_sparse, merge_extra,
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

HF_TOKEN           = os.environ.get("HF_TOKEN")
SAVE_DIR           = os.environ.get("SAVE_DIR", "./results/")
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

# Pipeline 1 — FrozenCore
D_EXTRA         = int(os.environ.get("D_EXTRA", "1024"))
K_EXTRA         = int(os.environ.get("K_EXTRA", "32"))
EPOCHS_EXTRA    = int(os.environ.get("EPOCHS_EXTRA", "10"))
LR_EXTRA        = float(os.environ.get("LR_EXTRA", "3e-4"))
USE_FROZEN_CORE = os.environ.get("USE_FROZEN_CORE", "1").strip() in ("1", "true", "True")
N_TOKENS_EXTRA_TRAIN = int(os.environ.get("N_TOKENS_EXTRA_TRAIN", "500000"))

# LLM Judge
N_FEATURES_TO_LABEL = int(os.environ.get("N_FEATURES_TO_LABEL", "10"))

# Modèle Gemma-3
MODEL_SIZE = os.environ.get("MODEL_SIZE", "12b")

if MODEL_SIZE == "12b":
    MODEL_ID   = os.environ.get("MODEL_ID", "/home/h21486/SAE/models/gemma-3-12b-it")
    RELEASE_ID = "gemma-scope-2-12b-it-res"
    SAE_ID     = os.environ.get("SAE_ID", "layer_24_width_16k_l0_medium")
    LAYER      = 24
elif MODEL_SIZE == "4b":
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

HOOK_TYPE      = os.environ.get("HOOK_TYPE", "resid_post")
LOCAL_SAE_ROOT = os.environ.get("LOCAL_SAE_DIR", f"/home/h21486/SAE/saes/{RELEASE_ID}")
SAE_SNAPSHOT   = os.environ.get(
    "SAE_SNAPSHOT", "0000000000000000000000000000000000000000"
)

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
# TÂCHE 4 : PROPERTY-BASED RETRIEVAL (Rank-Weighted)
# ══════════════════════════════════════════════════════════════════════════════

def property_based_retrieval(
    query_string: str,
    doc_acts: torch.Tensor,
    texts: list,
    feature_labels: dict,
    temperature: float = 0.2,
    top_n_results: int = 5,
) -> list:
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

    reducer = umap.UMAP(
        n_components=2, 
        metric="cosine",
        n_neighbors=min(30, max(2, N_DOCS - 1)),
        min_dist=0.1, 
        random_state=SEED,
        n_jobs=1,   # random_state force déjà n_jobs=1 ; explicite → supprime le UserWarning
    )
    coords = reducer.fit_transform(sae_active)

    # Libération immédiate des copies denses (N_DOCS × n_active en fp32, potentiellement
    # plusieurs Go à width 262k) : UMAP a fini, seuls `coords` (2D) et `sae_acts`
    # (torch, partagé avec l'appelant) restent nécessaires.
    del sae_np, sae_active, reducer
    _trim_host_memory()

    min_cs = max(2, N_DOCS // 15)
    clusterer = HDBSCAN(min_cluster_size=min_cs, min_samples=max(1, min_cs // 2), copy=True)
    clusters = clusterer.fit_predict(coords)

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
    email_texts: list = None,
    email_labels: list = None,
) -> dict:
    print("\n" + "=" * 70)
    print(" PIPELINE 1 : GEMMA-3 → MAX-POOL SAE ACTS")
    print("=" * 70)

    email_texts = email_texts or []
    email_labels = email_labels or []
    all_texts = train_texts + test_texts + email_texts

    pretrained_sae = load_pretrained_sae()
    pretrained_sae = pretrained_sae.to(DEVICE).to(torch.bfloat16).eval()
    pretrained_sae.requires_grad_(False)
    d_core = pretrained_sae.cfg.d_sae

    d_total_expected = d_core + D_EXTRA if USE_FROZEN_CORE else d_core

    cache_acts_path      = os.path.join(CACHE_DIR, "p1_all_doc_acts.pt")
    cache_residuals_path = os.path.join(CACHE_DIR, "p1_raw_residuals.pt")
    token_fragments_dir  = os.path.join(CACHE_DIR, "p1_token_fragments")
    
    n_train = len(train_texts)
    n_test  = len(test_texts)
    
    _need_extraction = True
    _need_residuals = USE_FROZEN_CORE and not os.path.exists(cache_residuals_path)
    
    if os.path.exists(cache_acts_path) and os.path.exists(token_fragments_dir):
        fragment_ids = list_fragment_ids(token_fragments_dir)
        if len(fragment_ids) == len(all_texts):
            print("  [P1] Restauration du cache (activations documents et fragments disques)...")
            all_doc_sae_acts = torch.load(cache_acts_path, map_location="cpu", weights_only=True)
            _need_extraction = False
            
            if _need_residuals:
                print("  [P1] Reconstruction de raw_residuals depuis les fragments de tokens locaux...")
                raw_residuals_list = []
                n_collected = 0
                for _fid in fragment_ids[:n_train]:
                    frag = load_fragment(token_fragments_dir, _fid)
                    if "raw_acts" in frag:
                        raw_residuals_list.append(frag["raw_acts"])
                        n_collected += frag["raw_acts"].shape[0]
                    if n_collected >= N_TOKENS_EXTRA_TRAIN:
                        break
                if raw_residuals_list:
                    raw_residuals = torch.cat(raw_residuals_list, dim=0)[:N_TOKENS_EXTRA_TRAIN]
                    torch.save(raw_residuals, cache_residuals_path)
                    print(f"  [P1] Résidus bruts d'apprentissage réinitialisés : {raw_residuals.shape}")
                    _need_residuals = False
                    del raw_residuals_list
        else:
            print("  [P1] Fragments disques incomplets. Re-extraction forcée.")
            _need_extraction = True

    if _need_extraction:
        print(f"  [P1] Extraction activations Gemma-3 ({MODEL_ID}, layer {LAYER})...")
        os.makedirs(token_fragments_dir, exist_ok=True)
        
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
        raw_residuals_list = []
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

                acts = acts_raw
                mask = inputs["attention_mask"].bool()

                for b in range(acts.shape[0]):
                    doc_global_idx = i + b
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
                    )

                    all_doc_sae_acts.append(doc_sae_vec.cpu())

                    if USE_FROZEN_CORE and n_residuals_collected < N_TOKENS_EXTRA_TRAIN and doc_global_idx < len(train_texts):
                        raw_residuals_list.append(filtered.cpu())
                        n_residuals_collected += filtered.shape[0]

        all_doc_sae_acts = torch.stack(all_doc_sae_acts)
        torch.save(all_doc_sae_acts, cache_acts_path)

        if USE_FROZEN_CORE and raw_residuals_list:
            raw_residuals = torch.cat(raw_residuals_list, dim=0)[:N_TOKENS_EXTRA_TRAIN]
            torch.save(raw_residuals, cache_residuals_path)
            print(f"  [P1] Résidus bruts d'apprentissage enregistrés : {raw_residuals.shape}")
            del raw_residuals_list
            _need_residuals = False
            
        del llm, tokenizer
        _trim_host_memory()

    d_total = d_core
    active_sae = pretrained_sae

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

                # FIX : Résolution de la signature erronée d'import
                from sae_shared import load_or_train_extended_sae as load_or_train
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
            cache_acts_ext = os.path.join(CACHE_DIR, f"p1_all_doc_acts_ext_d{D_EXTRA}.pt")
            if os.path.exists(cache_acts_ext) and not _need_extraction:
                all_doc_sae_acts = torch.load(cache_acts_ext, map_location="cpu", weights_only=True)
                if all_doc_sae_acts.shape[0] != len(all_texts):   # garde-fou explicite
                    all_doc_sae_acts = None
            else:
                all_doc_sae_acts = None
            if all_doc_sae_acts is None:
                print("  [P1] Re-encodage ExtendedSAE (fragments sparse, O(nnz))...")
                ext_sae.eval()
                new_acts = []
                _eval_raw, _EVAL_CAP = [], 4096   # capture x_t brut du split test avant purge (fix B1)
                with torch.no_grad():
                    for i in tqdm(range(len(all_texts)), desc="Re-encodage ExtendedSAE (sparse)"):
                        frag = load_fragment(token_fragments_dir, i)
                        raw_acts = frag["raw_acts"].to(DEVICE).to(torch.bfloat16)
                        # x_core reconstruit sans densifier [T, d_core] :
                        core_out_tokens = decode_core_sparse(frag, pretrained_sae, d_core, device=DEVICE)
                        residual_tokens = raw_acts - core_out_tokens
                        token_extra_acts = ext_sae._encode_extra_acts(residual_tokens)

                        if n_train <= i < n_train + n_test and \
                           sum(t.shape[0] for t in _eval_raw) < _EVAL_CAP:
                            _eval_raw.append(raw_acts.float().cpu())

                        csr = merge_extra(frag, token_extra_acts.float().cpu(), d_core)
                        save_fragment(token_fragments_dir, i,
                                      token_strings=frag["token_strings"],
                                      csr=csr, d_total=d_core + D_EXTRA)  # raw_acts non repassé -> purgé
                        new_acts.append(doc_maxpool({"rowptr": csr[0], "cols": csr[1],
                                                     "vals": csr[2], "shape": csr[3]}))

                all_doc_sae_acts = torch.stack(new_acts)
                torch.save(all_doc_sae_acts, cache_acts_ext)
                if _eval_raw:
                    torch.save(torch.cat(_eval_raw)[:_EVAL_CAP],
                               os.path.join(CACHE_DIR, "p1_eval_raw_tokens.pt"))
                    
            d_total = d_core + D_EXTRA
            active_sae = ext_sae
            print(f"  [P1] Dimension SAE étendue : {d_core} core + {D_EXTRA} extra = {d_total}")

    train_doc_acts = all_doc_sae_acts[:n_train]
    test_doc_acts  = all_doc_sae_acts[n_train: n_train + n_test]
    email_doc_acts = all_doc_sae_acts[n_train + n_test:]

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

    # -- Labels core : Neuronpedia (cache offline) ---------------------------
    np_labels_path = os.environ.get(
        "NEURONPEDIA_LABELS", os.path.join(CACHE_DIR, "neuronpedia_labels_core.json"))
    labels_core: dict[int, str] = {}
    if os.path.exists(np_labels_path):
        with open(np_labels_path, "r", encoding="utf-8") as f:
            labels_core = {int(k): v for k, v in json.load(f).items()}
        print(f"  [P1 Labels] {len(labels_core)} labels GemmaScope (Neuronpedia) chargés.")
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
                MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE,
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
        filename="umap_pipeline1_llm_per_token.html",
        title=f"Pipeline 1: Gemma-3 L{LAYER} → Max-Pool SAE Acts (FineWeb-2)",
        token_fragments_dir=token_fragments_dir, offset=n_train, feature_labels=label_map_p1,
    )
    if email_texts:
        analyze_with_umap(
            texts=email_texts, sae_acts=email_doc_acts, labels=email_labels,
            filename="umap_pipeline1_emails.html",
            title=f"Pipeline 1: Gemma-3 L{LAYER} → Max-Pool SAE Acts (EDF Mails)",
            token_fragments_dir=token_fragments_dir, offset=n_train + n_test, feature_labels=label_map_p1,
        )

    energy_mask = np.array([l == "energy" for l in train_labels])
    sports_mask  = np.array([l == "sports"  for l in train_labels])
    diff_hypothesis = "Aucun écart mesurable."
    if energy_mask.sum() > 0 and sports_mask.sum() > 0:
        # corpus_diff_stats : test exact de Fisher par feature + correction BH
        # (remplace diff_features, écarts de fréquences sans contrôle du FDR).
        pair_mask = energy_mask | sports_mask
        diff_df = corpus_diff_stats(
            train_doc_acts[torch.from_numpy(pair_mask)].float(),
            group_mask=energy_mask[pair_mask],       # True = Énergie (corpus A)
            feature_labels=label_map_p1,
        )
        diff_df.to_csv(os.path.join(SAVE_DIR, "p1_diff_energy_sports.csv"), index=False)

        if os.environ.get("RUN_DIFF_HYPOTHESIS", "1") == "1":
            j_tok = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True, local_files_only=True)
            j_llm = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE,
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

    npmi_mat = compute_npmi(test_doc_acts)
    torch.save(npmi_mat, os.path.join(CACHE_DIR, "p1_npmi.pt"))

    targeted_clustering_by_axis(
        texts=test_texts, sae_acts=test_doc_acts, labels=test_labels,
        feature_labels=label_map_p1, axis_query="énergie électrique"
    )

    results_retrieval = property_based_retrieval(
        "électrique nucléaire réseau", test_doc_acts, test_texts, label_map_p1
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
        raw_tokens_tensor = eval_raw_tokens[:1000].to(DEVICE).to(torch.bfloat16)
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
            print(f"  FVE (ExtendedSAE, tokens FR) = {metrics_ext['FVE']:.4f} | "
                  f"NMSE = {metrics_ext['NMSE']:.4f} | "
                  f"ΔFVE = {metrics_ext['FVE'] - metrics_pretrained['FVE']:+.4f}")
    else:
        metrics_pretrained = {"FVE": float("nan")}
        print("  [Metrics] Échantillon de tokens indisponible pour la FVE.")

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
        print(f"  [Downstream P1] Échantillons insuffisants pour entraîner la sonde logistique.")
        clf_results = {}

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
        "fve_pretrained": metrics_pretrained.get("FVE", float("nan")),
        "_test_doc_acts": test_doc_acts_out,
        "_label_map": label_map_p1,
        "_top_core": top_core_indices,
        "_top_ext": top_ext_indices,
    }
    del all_doc_sae_acts, train_doc_acts, test_doc_acts, email_doc_acts
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
    email_texts: list = None,
    email_labels: list = None,
) -> dict:
    print("\n" + "=" * 70)
    print(" PIPELINE 2 : F2LLM-v2 PHRASE-LEVEL SAE → MAX-POOL DOCUMENT")
    print("=" * 70)

    email_texts  = email_texts or []
    email_labels = email_labels or []

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

    email_doc_acts = None
    if email_texts:
        email_phrases, email_p2d_list = split_into_phrases(email_texts, max_phrases_per_doc=MAX_PHRASES_DOC)
        email_phrase_emb, _ = extract_f2llm_embeddings(
            email_phrases, max_length=128,
            cache_path=os.path.join(CACHE_DIR, f"email_phrase_emb_dim{MATRYOSHKA_DIM}_n{len(email_phrases)}"),
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
            MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE,
            low_cpu_mem_usage=True, local_files_only=True
        ).eval()
        # FIX 9 : local_gemma_judge attend les activations et textes au niveau PHRASE
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

    print("\n  [Downstream P2] Sonde logistique sur SAE activations...")
    energy_mask_test = np.array([l == "energy" for l in test_labels])
    sports_mask_test  = np.array([l == "sports"  for l in test_labels])
    if energy_mask_test.sum() > 0 and sports_mask_test.sum() > 0:
        try:
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
    run_steering_demo(results_p1)
    # Le steering n'a plus besoin des doc_acts : libération avant P2 (pic RSS).
    results_p1.pop("_test_doc_acts", None)
    _trim_host_memory()

    results_p2 = run_f2llm_pipeline(
        train_texts, train_labels, test_texts, test_labels, email_texts, email_labels
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