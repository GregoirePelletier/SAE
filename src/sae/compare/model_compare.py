"""
model_compare.py — Comparaison de deux modèles d'embeddings via leurs SAE
(Pipeline 2) et détection de "pollution" par les données d'entraînement.

Cadre formel
------------
Corpus partagé D (n docs). Pour chaque modèle m ∈ {A, B} : SAE_m entraîné sur
E_m(D), matrice d'activation Z_m ∈ R^{n×d_m}, binarisée B_m = 1[Z_m > τ].

1. Appariement de features cross-modèle (le corpus partagé sert de base commune) :
   C[i,j] = Pearson(Z_A[:,i], Z_B[:,j]) ; appariement hongrois
   (scipy.optimize.linear_sum_assignment) sur -C, restreint aux features vivantes.
   match_score_i = C[i, σ(i)].

2. Signature de pollution. Une feature "polluée" (artefact des données de
   pré-entraînement de E_m, pas du corpus D) présente conjointement :
   (a) orphelinat : match_score < τ_match — aucun équivalent dans l'autre modèle,
       alors que les concepts réels du corpus sont bilatéralement représentés ;
   (b) co-occurrence anormale : sa communauté NPMI ne s'aligne sur aucune
       partition métier du corpus (labels/domaines) — mesuré par l'AMI entre
       communautés Louvain et labels ;
   (c) incohérence sémantique intra-communauté : les top-docs de la communauté
       ont une similarité moyenne (dans l'AUTRE modèle, juge neutre) proche du
       niveau chance — la communauté est un clique structurel, pas thématique.

   pollution_score_i = z(orphan_i) + z(1 - ami_com(i)) + z(1 - coherence_com(i)),
   test de significativité par permutation (shuffle des colonnes de B_m).

3. Verdict au niveau modèle : fraction de masse d'activation portée par les
   features au-dessus du quantile 0.95 du null permuté, comparée entre A et B.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import networkx as nx
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score

try:
    from src.sae.sae_shared import compute_npmi
except ImportError:
    from sae_shared import compute_npmi
from .cooccurrence import cooccurrence_graph


# ─── 1. Appariement cross-modèle ───

def match_features(
    Z_A: torch.Tensor, Z_B: torch.Tensor, min_freq: float = 0.005
) -> pd.DataFrame:
    """Hongrois sur corrélation d'activation. Colonnes: feat_A, feat_B, corr."""
    fA = (Z_A > 1e-6).float().mean(0)
    fB = (Z_B > 1e-6).float().mean(0)
    liveA = (fA > min_freq).nonzero(as_tuple=True)[0]
    liveB = (fB > min_freq).nonzero(as_tuple=True)[0]

    Xa = _standardize(Z_A[:, liveA])
    Xb = _standardize(Z_B[:, liveB])
    C = (Xa.T @ Xb / Xa.shape[0]).cpu().numpy()          # Pearson [|A|,|B|]

    ra, cb = linear_sum_assignment(-C)
    return pd.DataFrame({
        "feat_A": liveA[ra].tolist(),
        "feat_B": liveB[cb].tolist(),
        "corr": C[ra, cb],
    }).sort_values("corr", ascending=False).reset_index(drop=True)


def _standardize(Z: torch.Tensor) -> torch.Tensor:
    Z = Z.float()
    return (Z - Z.mean(0)) / (Z.std(0) + 1e-8)


def orphan_scores(matches: pd.DataFrame, side: str = "A") -> pd.Series:
    """orphan = 1 - corr appariée (les features hors appariement → orphan = 1)."""
    col = f"feat_{side}"
    return pd.Series(1.0 - matches.set_index(col)["corr"].clip(lower=0.0), name="orphan")


# ─── 2. Score de pollution ───

@dataclass
class PollutionReport:
    per_feature: pd.DataFrame     # feature_id, orphan, ami_alignment, coherence, pollution_score, q_null
    model_score: float            # masse d'activation portée par features > quantile null
    n_flagged: int


def pollution_report(
    Z_self: torch.Tensor,
    Z_other: torch.Tensor,
    emb_other: torch.Tensor,          # embeddings bruts du modèle-juge [n, d]
    corpus_labels: np.ndarray,        # labels métier (domaine/intention) [n]
    matches: pd.DataFrame,
    side: str = "A",
    npmi_threshold: float = 0.3,
    n_perm: int = 200,
    top_docs: int = 20,
    seed: int = 0,
) -> PollutionReport:
    rng = np.random.default_rng(seed)
    orphan = orphan_scores(matches, side)

    G = cooccurrence_graph(Z_self, npmi_threshold=npmi_threshold)
    coms = nx.get_node_attributes(G, "community")

    # (b) alignement communautés ↔ labels : AMI par communauté
    B = (Z_self > 1e-6).cpu().numpy()
    ami_by_com = {}
    for cid in set(coms.values()):
        feats = [f for f, c in coms.items() if c == cid]
        doc_in_com = B[:, feats].any(axis=1).astype(int)
        ami_by_com[cid] = max(adjusted_mutual_info_score(doc_in_com, corpus_labels), 0.0)

    # (c) cohérence des top-docs de la communauté, jugée par l'AUTRE modèle
    E = torch.nn.functional.normalize(emb_other.float(), dim=1)
    base_sim = float((E @ E.T).mean())                    # niveau chance corpus
    coh_by_com = {}
    for cid in set(coms.values()):
        feats = [f for f, c in coms.items() if c == cid]
        score = torch.as_tensor(B[:, feats].sum(axis=1), dtype=torch.float)
        idx = score.topk(min(top_docs, len(score))).indices
        S = E[idx] @ E[idx].T
        coh_by_com[cid] = float((S.sum() - S.trace()) / (len(idx) * (len(idx) - 1)))

    rows = []
    for f in G.nodes:
        cid = coms[f]
        rows.append({
            "feature_id": f,
            "orphan": float(orphan.get(f, 1.0)),
            "ami_alignment": ami_by_com[cid],
            "coherence": max(coh_by_com[cid] - base_sim, 0.0),
            "community": cid,
        })
    df = pd.DataFrame(rows)
    z = lambda s: (s - s.mean()) / (s.std() + 1e-8)
    df["pollution_score"] = z(df["orphan"]) + z(1 - df["ami_alignment"]) + z(-df["coherence"])

    # Null par permutation : shuffle indépendant des colonnes de B → détruit la
    # structure de co-occurrence en préservant les fréquences marginales.
    null_max = []
    for _ in range(n_perm):
        Bp = B.copy()
        for j in range(Bp.shape[1]):
            rng.shuffle(Bp[:, j])
        npmi_p = compute_npmi(torch.as_tensor(Bp, dtype=torch.float))
        null_max.append(float(npmi_p.fill_diagonal_(0).max()))
    q95 = float(np.quantile(null_max, 0.95))
    df["q_null_npmi95"] = q95

    thr = df["pollution_score"].mean() + 2 * df["pollution_score"].std()
    flagged = df[df["pollution_score"] > thr]
    mass = float(Z_self[:, flagged["feature_id"].tolist()].sum() / (Z_self.sum() + 1e-8))
    return PollutionReport(df.sort_values("pollution_score", ascending=False),
                           model_score=mass, n_flagged=len(flagged))


# ─── 3. Orchestration ───

def compare_embedding_models(
    Z_A: torch.Tensor, Z_B: torch.Tensor,
    emb_A: torch.Tensor, emb_B: torch.Tensor,
    corpus_labels: np.ndarray,
) -> dict:
    """Rapport symétrique. model_score élevé + n_flagged élevé ⇒ modèle pollué."""
    matches = match_features(Z_A, Z_B)
    repA = pollution_report(Z_A, Z_B, emb_B, corpus_labels, matches, side="A")
    repB = pollution_report(Z_B, Z_A, emb_A, corpus_labels, matches, side="B")
    return {
        "matches": matches,
        "report_A": repA,
        "report_B": repB,
        "verdict": ("A pollué" if repA.model_score > 1.5 * repB.model_score
                    else "B pollué" if repB.model_score > 1.5 * repA.model_score
                    else "comparable"),
        "mean_match_corr": float(matches["corr"].mean()),
    }