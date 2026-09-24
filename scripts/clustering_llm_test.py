"""
scripts/clustering_llm_test.py — Clustering ciblé complet (Appendix F.1,
§4.3, arXiv:2512.10092v2) : `saev5.py::targeted_clustering_by_axis` ne fait
que la sélection de latents par UN SEUL `axis_query` fixe + Jaccard (corrigé
séparément, cf. docs/archive/audits/AUDIT_SAE_2026-08.md) -- il manquait la génération de
mots-clés LLM + union top-k, l'étiquetage de cluster, l'accuracy par
réassignation LLM et le z-score de conductance en espace dense
(docs/archive/audits/AUDIT_SAE_2026-08.md §1/§7). Ce script exerce la chaîne complète sur les 150
features d'extension déjà labellisées (`p1_top_extended_features.json`, reste
autonome sans reconstruire le dictionnaire complet core+extension).

Usage :
    SAVE_DIR=./results_v10_emails_main/ PYTHONPATH=. \
      .venv/bin/python scripts/clustering_llm_test.py
"""
import json
import os
import sys

import numpy as np
import torch
from src.sae.sae_shared import load_all_doc_acts
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import SpectralClustering
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import SAVE_DIR, CORPUS_SPLIT_SEED, LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, LATENT_LABEL_EMB_MODEL
from src.data.preparation import build_email_train_test_corpus
from src.analysis.clustering_llm import (
    generate_keywords, select_latents_union, generate_cluster_labels,
    compute_cluster_accuracy, conductance_zscore,
)
from src.sae.judge import load_judge_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AXIS_QUERY = os.environ.get("AXIS_QUERY", "type de réclamation client")
TOP_K_PER_KEYWORD = int(os.environ.get("TOP_K_PER_KEYWORD", "50"))
N_CLUSTERS = int(os.environ.get("N_CLUSTERS", "4"))
N_DOCS = int(os.environ.get("N_DOCS", "300"))
SEED = int(os.environ.get("SEED", "42"))

CACHE_DIR = os.path.join(SAVE_DIR, "cache")
EXT_FEATURES_PATH = os.path.join(SAVE_DIR, "p1_top_extended_features.json")
OUT_PATH = os.path.join(CACHE_DIR, "clustering_llm_verification.json")


def _embed(texts: list, batch_size: int = 64) -> torch.Tensor:
    """Même convention que `saev5.py::_embed_bge_m3` (pooling [CLS] normalisé)
    -- dupliqué ici car `saev5.py` exécute tout son pipeline au chargement du
    module (pas de garde `if __name__ == "__main__"`), donc non-importable
    comme bibliothèque (cf. docs/archive/audits/AUDIT_SAE_2026-08.md, limitation notée)."""
    tok = AutoTokenizer.from_pretrained(LATENT_LABEL_EMB_MODEL, local_files_only=True)
    mdl = AutoModel.from_pretrained(LATENT_LABEL_EMB_MODEL, local_files_only=True).to(DEVICE).eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=64,
                      return_tensors="pt").to(DEVICE)
            cls = mdl(**enc).last_hidden_state[:, 0]
            embs.append(F.normalize(cls, p=2, dim=-1).cpu())
    del mdl
    return torch.cat(embs, dim=0)


def main():
    with open(EXT_FEATURES_PATH, encoding="utf-8") as f:
        ext_data = json.load(f)
    feature_labels = {int(fid): v["label"] for fid, v in ext_data.items() if v.get("interp_score") == 1}
    print(f"[cluster-llm] {len(feature_labels)} features interprétables depuis {EXT_FEATURES_PATH}", flush=True)

    train_texts, _, _, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED,
    )
    all_doc_acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts_ext_d1024.pt")
    if not os.path.exists(all_doc_acts_path):
        all_doc_acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts.pt")
    all_doc_sae_acts = load_all_doc_acts(all_doc_acts_path)
    train_doc_acts = all_doc_sae_acts[:len(train_texts)]

    rng = np.random.default_rng(SEED)
    doc_idx = np.sort(rng.choice(len(train_texts), size=min(N_DOCS, len(train_texts)), replace=False))
    texts = [train_texts[i] for i in doc_idx]
    doc_acts = train_doc_acts[doc_idx]
    print(f"[cluster-llm] {len(texts)} documents échantillonnés pour le clustering", flush=True)

    print("[cluster-llm] Chargement du juge...", flush=True)
    model, tokenizer = load_judge_model()

    print(f"[cluster-llm] Génération de mots-clés pour : {AXIS_QUERY!r}", flush=True)
    keywords = generate_keywords(model, tokenizer, AXIS_QUERY)
    print(f"[cluster-llm] Mots-clés : {keywords}", flush=True)

    fids = sorted(feature_labels.keys())
    label_embs = _embed(list(feature_labels.values()))
    label_emb_map = {fid: label_embs[i] for i, fid in enumerate(fids)}

    def _select_fn(query, labels_dict, top_k):
        q_emb = _embed([query])[0]
        sims = torch.stack([label_emb_map[fid] for fid in labels_dict]) @ q_emb
        order = torch.argsort(sims, descending=True)[:top_k]
        return [list(labels_dict.keys())[i] for i in order.tolist()]

    matched = select_latents_union(keywords, feature_labels, select_fn=_select_fn, top_k=TOP_K_PER_KEYWORD)
    print(f"[cluster-llm] {len(matched)} latents sélectionnés (union sur {len(keywords)} mots-clés)", flush=True)
    if len(matched) < 5:
        matched = fids

    sub_binarized = (doc_acts[:, matched].float().numpy() > 1e-6).astype(np.float32)
    jaccard_sim = 1.0 - squareform(pdist(sub_binarized, metric="jaccard"))
    spectral = SpectralClustering(n_clusters=N_CLUSTERS, affinity="precomputed", assign_labels="kmeans", random_state=SEED)
    cluster_labels = spectral.fit_predict(jaccard_sim)
    print(f"[cluster-llm] Tailles de cluster : {np.bincount(cluster_labels).tolist()}", flush=True)

    # Diff simplifié pour la description de cluster (top features par fréquence
    # DANS le cluster, top exemples = 5 premiers documents du cluster) -- pas
    # un vrai test de Fisher inter-cluster comme corpus_diff_stats, suffisant
    # pour fournir un contexte au prompt de labellisation (Appendix F.1).
    clusters_info = []
    for c in range(N_CLUSTERS):
        mask = cluster_labels == c
        if mask.sum() == 0:
            clusters_info.append({"features": [], "examples": []})
            continue
        freq = sub_binarized[mask].mean(axis=0)
        top_local = np.argsort(freq)[::-1][:5]
        top_feats = [feature_labels[matched[i]] for i in top_local]
        top_examples = [texts[i][:200] for i in np.nonzero(mask)[0][:5]]
        clusters_info.append({"features": top_feats, "examples": top_examples})

    print("[cluster-llm] Génération des labels de cluster...", flush=True)
    cluster_descriptions_list = generate_cluster_labels(model, tokenizer, clusters_info)
    cluster_descriptions = dict(enumerate(cluster_descriptions_list))
    for cid, desc in cluster_descriptions.items():
        print(f"    Cluster {cid} : {desc}", flush=True)

    print("[cluster-llm] Accuracy par réassignation LLM...", flush=True)
    accuracy = compute_cluster_accuracy(model, tokenizer, texts, cluster_labels.tolist(), cluster_descriptions)

    print("[cluster-llm] Z-score de conductance (espace dense bge-m3)...", flush=True)
    dense_embeddings = _embed(texts).numpy()
    z_scores = conductance_zscore(dense_embeddings, cluster_labels, n_random=100, seed=SEED)

    results = {
        "axis_query": AXIS_QUERY, "keywords": keywords, "n_clusters": N_CLUSTERS,
        "cluster_sizes": np.bincount(cluster_labels).tolist(),
        "cluster_descriptions": cluster_descriptions,
        "accuracy": accuracy, "conductance_zscore": z_scores,
    }
    print("\n" + "=" * 60, flush=True)
    for cid in cluster_descriptions:
        print(f"  Cluster {cid} ({cluster_descriptions[cid]}) : acc={accuracy.get(cid):.3f} "
              f"z_conductance={z_scores.get(cid, float('nan')):.2f}", flush=True)

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
