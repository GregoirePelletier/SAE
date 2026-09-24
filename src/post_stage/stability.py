"""Primitives numpy de la stabilite inter-graines des features EXTRA (E05,
docs/post_stage/PLAN_E00-E09.md §10). Pures (aucun modele, aucun
I/O) pour rester testables en CPU rapide ; le script scripts/post_stage/
e05_stability.py assemble les checkpoints et les activations reels."""
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


def normalize_rows(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=np.float64)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    return W / np.maximum(norms, 1e-12)


def nn_match(Wa: np.ndarray, Wb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Plus proche voisin (cosinus SIGNE -- les coefficients SAE sont non
    negatifs, la valeur absolue n'a pas de sens, plan §10.3), plusieurs-a-un
    autorise. Retourne (best_idx dans Wb, best_cos) pour chaque ligne de Wa."""
    C = normalize_rows(Wa) @ normalize_rows(Wb).T
    best_idx = C.argmax(axis=1)
    return best_idx, C[np.arange(C.shape[0]), best_idx]


def matched_column_corr(Xa: np.ndarray, Xb: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
    """Pearson entre les colonnes Xa[:, ia[k]] et Xb[:, ib[k]] (memes lignes =
    memes documents). NaN si l'une des deux colonnes est constante ("support
    insuffisant", plan §10.2 -- jamais une correlation fabriquee)."""
    A = Xa[:, ia].astype(np.float64)
    B = Xb[:, ib].astype(np.float64)
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    num = (A * B).sum(axis=0)
    den = np.sqrt((A * A).sum(axis=0) * (B * B).sum(axis=0))
    out = np.full(num.shape, np.nan)
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out


def topk_jaccard(sa: np.ndarray, sb: np.ndarray, k: int = 20) -> float:
    """Jaccard@k des k documents les plus activants (tri stable : a egalite
    l'indice de document le plus petit gagne -- deterministe)."""
    ta = set(np.argsort(-sa, kind="stable")[:k].tolist())
    tb = set(np.argsort(-sb, kind="stable")[:k].tolist())
    return len(ta & tb) / len(ta | tb)


def anisotropic_null_dictionary(W: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """Dictionnaire nul de m directions unitaires tirees d'une gaussienne de
    MEME moyenne et MEME covariance empirique que les lignes de W (geometrie
    anisotrope reelle, plan §10.5 : la reference isotrope r/d n'est qu'une
    intuition). Tirage = mu + combinaison gaussienne des lignes centrees."""
    Wn = normalize_rows(W)
    mu = Wn.mean(axis=0, keepdims=True)
    centered = (Wn - mu) / np.sqrt(max(Wn.shape[0] - 1, 1))
    Z = rng.standard_normal((m, Wn.shape[0]))
    return normalize_rows(mu + Z @ centered)


def mutual_knn_edges(W: np.ndarray, k: int = 10) -> List[Tuple[int, int, float]]:
    """Aretes (i<j, cosinus) entre voisinages MUTUELS parmi les k plus proches
    (plan §10.4)."""
    Wn = normalize_rows(W)
    C = Wn @ Wn.T
    np.fill_diagonal(C, -np.inf)
    n = C.shape[0]
    k = min(k, n - 1)
    if k < 1:
        return []
    topk = np.argpartition(-C, k - 1, axis=1)[:, :k]
    neigh = [set(row.tolist()) for row in topk]
    edges = []
    for i in range(n):
        for j in neigh[i]:
            if i < j and i in neigh[j]:
                edges.append((i, j, float(C[i, j])))
    return edges


def louvain_labels(n_nodes: int, weighted_edges: List[Tuple[int, int, float]], seed: int = 0,
                   resolution: float = 1.0) -> np.ndarray:
    """Communautes Louvain, UNE seule configuration (plan §10.4 : pas de
    balayage de resolutions). Chaque noeud sans arete reste un isolat.
    Identifiants tries par taille decroissante (0 = plus gros groupe)."""
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_weighted_edges_from(weighted_edges)
    comms = nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
    comms = sorted(comms, key=lambda c: (-len(c), min(c)))
    labels = np.full(n_nodes, -1, dtype=int)
    for gid, c in enumerate(comms):
        for node in c:
            labels[node] = gid
    return labels


def subspace_basis(D: np.ndarray, r_cap: int = 16, energy: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    """Base orthonormee (d x r) du sous-espace engendre par les lignes de D
    (SVD). Rang r = plus petit r tel que `energy` de l'energie spectrale est
    couverte, plafonne a r_cap et au rang de D (regle fixee a l'avance, pas
    choisie apres coup). Retourne (Q, spectre)."""
    U, s, Vt = np.linalg.svd(normalize_rows(D), full_matrices=False)
    e = (s ** 2) / max((s ** 2).sum(), 1e-12)
    r = int(np.searchsorted(np.cumsum(e), energy) + 1)
    r = max(1, min(r, r_cap, len(s)))
    return Vt[:r].T, s


def subspace_overlap(Qa: np.ndarray, Qb: np.ndarray) -> Tuple[float, int]:
    """overlap = ||Qa^T Qb||_F^2 / r sur r = min(rang a, rang b) premiers axes
    (plan §10.5). Retourne (overlap, r) -- le rang n'est jamais cache."""
    r = min(Qa.shape[1], Qb.shape[1])
    M = Qa[:, :r].T @ Qb[:, :r]
    return float((M ** 2).sum() / r), r


def group_partner(members_a: np.ndarray, match_idx: np.ndarray, labels_b: np.ndarray,
                  exclude_label: Optional[int] = None) -> Tuple[Optional[int], float, int]:
    """Groupe partenaire dans B d'un groupe A : etiquette de B la plus
    frequente parmi les correspondances individuelles (plus proche voisin)
    de ses membres. Retourne (partner_label|None, purity, n_matched) --
    purity = part des membres apparies tombant dans le partenaire. Un
    partenaire isolat (etiquette de taille 1) est autorise : "absence de
    correspondance" est un resultat, pas une erreur."""
    targets = labels_b[match_idx[members_a]]
    if len(targets) == 0:
        return None, float("nan"), 0
    vals, counts = np.unique(targets, return_counts=True)
    order = np.lexsort((vals, -counts))
    best = vals[order[0]]
    return int(best), float(counts[order[0]] / len(targets)), int(len(targets))


def resample_group_same_strata(members: np.ndarray, strata: np.ndarray, pool: np.ndarray,
                               rng: np.random.Generator) -> np.ndarray:
    """Groupe aleatoire de meme taille, chaque membre remplace par un
    element du pool tire dans la MEME strate (frequence), sans remise quand
    possible -- controle de taille+frequence du plan §10.6."""
    out = []
    used = set()
    for m in members:
        cand = pool[(strata[pool] == strata[m]) & ~np.isin(pool, list(used))]
        if len(cand) == 0:
            cand = pool[~np.isin(pool, list(used))]
        if len(cand) == 0:
            cand = pool
        pick = int(rng.choice(cand))
        used.add(pick)
        out.append(pick)
    return np.array(out, dtype=int)
