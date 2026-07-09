"""
cooccurrence.py — Graphe de co-occurrence NPMI, diff de corpus, clustering en
espace sparse. S'appuie sur networkx + scipy (pas d'implémentation maison de
Louvain/tests stat). Héberge compute_npmi (implémentation unique) et corpus_diff_stats (Fisher+BH).
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd
import torch
import networkx as nx
from scipy import sparse
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


def compute_npmi(doc_acts: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """NPMI vectorisée. npmi_ij = pmi_ij / (-log p_ij) ; diag = 1 ; 0 si cooc nulle.
    (Unique implémentation — supprimée de sae_shared.)"""
    n = doc_acts.shape[0]
    b = (doc_acts > 1e-6).float()
    cooc = b.T @ b
    p_ij = cooc / n
    p_i = b.sum(0) / n
    pmi = torch.log((p_ij + eps) / (p_i.unsqueeze(1) * p_i.unsqueeze(0) + eps))
    npmi = pmi / (-torch.log(p_ij + eps))
    npmi = torch.where(cooc > 0, npmi, torch.zeros_like(npmi))
    npmi.fill_diagonal_(1.0)
    return npmi


# ─── Graphe de co-occurrence ───

def cooccurrence_graph(
    doc_acts: torch.Tensor,
    npmi_threshold: float = 0.3,
    min_freq: float = 0.01,
    max_freq: float = 0.5,
    feature_labels: Optional[dict[int, str]] = None,
) -> nx.Graph:
    """
    Nœuds = features (fréquence ∈ [min_freq, max_freq] pour écarter morts et
    quasi-denses type sink), arêtes = NPMI > seuil. Communautés Louvain
    (networkx.community.louvain_communities) stockées en attribut de nœud.
    """
    freq = (doc_acts > 1e-6).float().mean(0)
    keep = ((freq >= min_freq) & (freq <= max_freq)).nonzero(as_tuple=True)[0]
    npmi = compute_npmi(doc_acts[:, keep])

    G = nx.Graph()
    for local, f in enumerate(keep.tolist()):
        G.add_node(f, freq=float(freq[f]),
                   label=(feature_labels or {}).get(f, f"F{f}"))
    iu, ju = torch.triu_indices(len(keep), len(keep), offset=1)
    vals = npmi[iu, ju]
    sel = (vals > npmi_threshold).nonzero(as_tuple=True)[0]
    for s in sel.tolist():
        G.add_edge(int(keep[iu[s]]), int(keep[ju[s]]), npmi=float(vals[s]))

    for cid, com in enumerate(nx.community.louvain_communities(G, weight="npmi", seed=0)):
        for n in com:
            G.nodes[n]["community"] = cid
    return G


# ─── Diff statistique entre sous-corpus ───

def corpus_diff_stats(
    doc_acts: torch.Tensor,
    group_mask: np.ndarray,           # bool [n_docs] : True = corpus A
    feature_labels: Optional[dict[int, str]] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Test exact de Fisher par feature (2×2 : actif/inactif × A/B), correction
    Benjamini-Hochberg. Retourne log-odds-ratio + q-values, trié par |LOR| signif.
    """
    bin_acts = (doc_acts > 1e-6).cpu().numpy()
    A, B = bin_acts[group_mask], bin_acts[~group_mask]
    nA, nB = len(A), len(B)
    rows = []
    for f in range(bin_acts.shape[1]):
        a, b = int(A[:, f].sum()), int(B[:, f].sum())
        if a + b == 0:
            continue
        odds, p = fisher_exact([[a, nA - a], [b, nB - b]])
        lor = np.log((a + .5) * (nB - b + .5) / ((nA - a + .5) * (b + .5)))  # Haldane
        rows.append({"feature_id": f, "freq_A": a / nA, "freq_B": b / nB,
                     "log_odds_ratio": lor, "p": p})
    cols = ["feature_id", "freq_A", "freq_B", "log_odds_ratio", "p", "q", "significant", "label"]
    if not rows:
        # Corpus trop petit / features trop sparses : aucune feature active dans A ∪ B.
        # DataFrame vide mais bien formée plutôt qu'un KeyError sur colonnes absentes.
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    df["q"] = multipletests(df["p"], method="fdr_bh")[1]
    df["significant"] = df["q"] < alpha
    df["label"] = df["feature_id"].map(lambda i: (feature_labels or {}).get(i, f"F{i}"))
    return df.sort_values("log_odds_ratio", key=abs, ascending=False).reset_index(drop=True)


# ─── Clustering en espace sparse ───

def cluster_in_feature_space(
    doc_acts: torch.Tensor,
    min_cluster_size: int = 15,
    metric: str = "cosine",
) -> tuple[np.ndarray, np.ndarray]:
    """
    HDBSCAN sur profils d'activation SAE (binarisés puis TF-IDF-pondérés) —
    et non sur embeddings bruts. Retourne (labels, UMAP 2D pour visu).
    """
    import umap
    import hdbscan
    from sklearn.feature_extraction.text import TfidfTransformer

    X = sparse.csr_matrix((doc_acts > 1e-6).float().cpu().numpy())
    X = TfidfTransformer().fit_transform(X)               # downweight features denses
    emb2d = umap.UMAP(n_components=2, metric=metric, random_state=0, n_jobs=1).fit_transform(X)
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(emb2d)
    return labels, emb2d