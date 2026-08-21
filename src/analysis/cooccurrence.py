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
    max_features: int = 4000,
) -> nx.Graph:
    """
    Nœuds = features (fréquence ∈ [min_freq, max_freq] pour écarter morts et
    quasi-denses type sink), arêtes = NPMI > seuil. Communautés Louvain
    (networkx.community.louvain_communities) stockées en attribut de nœud.

    `max_features` (défaut 4000, même plafond que le chemin voisin
    `saev5.py::keep_npmi`) : `torch.triu_indices` ci-dessous est O(K²) en
    mémoire -- sans plafond, une largeur de dictionnaire de 65k/262k avec une
    bande de fréquence peu sélective ferait exploser cette allocation
    (AUDIT_SAE_2026-08.md, §2 Performance). Les features en excès sont
    tronquées après le tri implicite de `nonzero()` (ordre d'indice croissant,
    pas un tri par fréquence) -- même convention que `keep_npmi`."""
    freq = (doc_acts > 1e-6).float().mean(0)
    keep = ((freq >= min_freq) & (freq <= max_freq)).nonzero(as_tuple=True)[0][:max_features]
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


def find_interesting_pairs(
    G: nx.Graph,
    label_embeddings: dict[int, np.ndarray],
    npmi_threshold: float = 0.6,
    sim_threshold: float = 0.2,
) -> list[dict]:
    """
    Isole, parmi les arêtes du graphe, les paires "intéressantes" au sens d'interp_embed
    (Jiang, Sun et al. 2025, §4.2/Appendix E.1 — cf. docs/references.md) : NPMI élevé
    (fortement corrélées) MAIS labels sémantiquement DISSIMILAIRES. Une corrélation entre
    deux concepts a priori non reliés (ex. "urgence" et "facturation") révèle plus
    probablement un biais/artefact réel qu'une corrélation entre labels quasi-synonymes
    (ex. "facturation" et "montant dû", déjà attendus comme corrélés) — `cooccurrence_graph`
    seul mélange les deux, ce filtre les sépare. `label_embeddings` : {feature_id: vecteur
    normalisé L2} précalculé par l'appelant (ce module reste agnostique du modèle
    d'embedding utilisé, cf. src/sae/saev5.py::select_latents_by_similarity pour la
    réutilisation de F2LLM déjà en place dans le projet).
    """
    results = []
    for u, v, data in G.edges(data=True):
        npmi = data.get("npmi", 0.0)
        if npmi < npmi_threshold or u not in label_embeddings or v not in label_embeddings:
            continue
        sim = float(np.dot(label_embeddings[u], label_embeddings[v]))
        if sim < sim_threshold:
            results.append({
                "feature_a": u, "label_a": G.nodes[u].get("label", f"F{u}"),
                "feature_b": v, "label_b": G.nodes[v].get("label", f"F{v}"),
                "npmi": npmi, "label_similarity": sim,
            })
    return sorted(results, key=lambda r: r["npmi"], reverse=True)


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
    # Comptage vectorisé une seule fois sur toute la largeur (A.sum(0)) plutôt
    # qu'un A[:, f].sum() par feature dans la boucle -- même résultat, mais un
    # seul passage optimisé au lieu de d_sae accès colonne strided sur un
    # tableau C-contigu (AUDIT_SAE_2026-08.md, §2 Performance). fisher_exact
    # lui-même reste par feature (scipy ne le vectorise pas) : seule la partie
    # comptage change, la méthode statistique est identique.
    a_counts, b_counts = A.sum(0), B.sum(0)
    rows = []
    for f in range(bin_acts.shape[1]):
        a, b = int(a_counts[f]), int(b_counts[f])
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

    HDBSCAN tourne sur un embedding UMAP 10D dédié, PAS sur `emb2d` (réservé
    à la visualisation) : UMAP-10D domine UMAP-2D sur la stabilité inter-seed
    du clustering à DBCV quasi identique ; PCA et l'espace cosine brut sont
    nettement dominés par UMAP sur ce corpus.
    """
    import umap
    import hdbscan
    from sklearn.feature_extraction.text import TfidfTransformer

    X = sparse.csr_matrix((doc_acts > 1e-6).float().cpu().numpy())
    X = TfidfTransformer().fit_transform(X)               # downweight features denses
    n_docs = X.shape[0]
    emb2d = umap.UMAP(n_components=2, metric=metric, random_state=0, n_jobs=1).fit_transform(X)
    if n_docs <= 12:
        cluster_embedding = emb2d
    else:
        n_components = min(10, n_docs - 2)
        cluster_embedding = umap.UMAP(
            n_components=n_components, metric=metric, random_state=0, n_jobs=1
        ).fit_transform(X)
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(cluster_embedding)
    return labels, emb2d