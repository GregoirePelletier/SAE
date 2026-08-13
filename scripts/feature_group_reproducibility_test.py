"""
scripts/feature_group_reproducibility_test.py — Grouper les features
d'extension gagne-t-il en reproductibilité inter-seed par rapport à la
feature individuelle ?

Contexte : `RESULTS_TESTS.md` §21 (ablation seed-variance, SEED=42 "run
principal" `results_v10_emails_main` vs SEED=123 `results_v13_ablation_seed123`)
montre qu'au niveau INDIVIDUEL seulement 22/78 = 28,2% des labels de features
d'extension sont des chaînes de caractères IDENTIQUES entre les deux seeds.

Groupement par SIMILARITÉ SÉMANTIQUE (embeddings bge-m3,
`src/sae/saev5.py::_embed_bge_m3`) des labels plutôt que par co-activation :
`cooccurrence_graph` (filtre de fréquence document ∈ [0.01, 0.5]) et la
sélection des 150 features jugées (`feature_selection_by_magnitude`, par
magnitude, pas fréquence) ciblent des ensembles disjoints -- aucune des 150
features jugées par seed n'apparaît comme nœud du graphe NPMI, dans les deux
seeds (les features TopK-sparses à forte magnitude s'activent trop rarement
en fréquence documentaire pour passer ce filtre). Utilise uniquement
`p1_top_extended_features.json`, déjà en cache pour les deux seeds -- aucune
activation brute requise. Compare deux façons d'apparier les features
interprétables entre seed=42 et seed=123 SUR LA MÊME MÉTRIQUE (similarité
cosinus d'embedding de label, via appariement hongrois -- même primitive que
`match_features`, appliquée ici aux embeddings de label plutôt qu'aux
activations) :
  (a) FEATURE-À-FEATURE : chaque feature interprétable de seed A appariée à
      sa plus proche voisine de seed B (généralise le recouvrement EXACT du
      §21 en un score continu).
  (b) GROUPE-À-GROUPE : les features interprétables de chaque seed sont
      d'abord regroupées par similarité de label (communautés Louvain sur un
      graphe de similarité cosinus intra-seed, seuil explicite), puis les
      GROUPES (pas les features individuelles) sont appariés entre seeds par
      similarité de centroïde.

Si (b) donne une similarité de meilleur-appariement significativement plus
élevée que (a), le regroupement améliore bien la reproductibilité apparente.

Usage (CPU uniquement) :
    PYTHONPATH=. .venv/bin/python scripts/feature_group_reproducibility_test.py
"""
from __future__ import annotations

import json
import os
import sys

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "sae"))

RUN_A = "./results_v10_emails_main"       # SEED=42, "run principal" (§21)
RUN_B = "./results_v13_ablation_seed123"  # SEED=123
OUT_PATH = "./results_v10_emails_main/cache/feature_group_reproducibility_results.json"
SIM_EDGE_THRESHOLD = 0.5   # seuil arbitraire mais explicite, pour le graphe intra-seed
SEED = 42


def load_interpretable_labels(run_dir: str) -> dict[str, str]:
    with open(os.path.join(run_dir, "p1_top_extended_features.json"), encoding="utf-8") as f:
        data = json.load(f)
    return {k: v["label"] for k, v in data.items() if v.get("interp_score") == 1}


def louvain_by_label_similarity(embs: np.ndarray, keys: list[str]) -> dict[str, int]:
    """Communautés Louvain sur un graphe de similarité cosinus intra-ensemble
    (même primitive networkx que `cooccurrence_graph`, appliquée à une
    similarité de label plutôt qu'à une co-activation NPMI -- cf. docstring
    module pour la justification du pivot)."""
    sim = embs @ embs.T
    G = nx.Graph()
    G.add_nodes_from(keys)
    n = len(keys)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] > SIM_EDGE_THRESHOLD:
                G.add_edge(keys[i], keys[j], weight=float(sim[i, j]))
    communities = list(nx.community.louvain_communities(G, weight="weight", seed=SEED))
    return {k: cid for cid, com in enumerate(communities) for k in com}, communities


def best_match_similarities(embs_a: np.ndarray, embs_b: np.ndarray) -> np.ndarray:
    """Appariement hongrois (même primitive que `model_compare.match_features`,
    ici sur une matrice de similarité cosinus de labels). Retourne la
    similarité de la paire assignée à chaque élément de A (n_a <= n_b sinon
    tronqué par linear_sum_assignment -- pas de contrainte 1-à-1 stricte
    nécessaire ici, seule la distribution de similarité importe)."""
    sim = embs_a @ embs_b.T
    ra, cb = linear_sum_assignment(-sim)
    return sim[ra, cb]


def main() -> None:
    print("[repro] Chargement des labels interprétables (extension, interp_score==1)...")
    labels_a = load_interpretable_labels(RUN_A)
    labels_b = load_interpretable_labels(RUN_B)
    print(f"[repro] seed=42 : {len(labels_a)} interprétables ; seed=123 : {len(labels_b)} "
          f"(cf. §21 : 68 et 71 attendus).")

    from src.sae.saev5 import _embed_bge_m3
    keys_a, keys_b = list(labels_a), list(labels_b)
    print(f"[repro] Embedding bge-m3 de {len(keys_a) + len(keys_b)} labels...")
    all_embs = _embed_bge_m3([labels_a[k] for k in keys_a] + [labels_b[k] for k in keys_b]).numpy()
    embs_a, embs_b = all_embs[:len(keys_a)], all_embs[len(keys_a):]

    # ── Recouvrement EXACT (réplique §21, sanity check) ──────────────────
    exact_overlap = len(set(labels_a.values()) & set(labels_b.values()))
    print(f"[repro] Recouvrement EXACT de labels (réplique §21) : {exact_overlap} "
          f"labels identiques en commun.")

    # ── (a) FEATURE-À-FEATURE ────────────────────────────────────────────
    sim_feature = best_match_similarities(embs_a, embs_b)
    print(f"\n[repro] (a) Appariement FEATURE-À-FEATURE : "
          f"moyenne={sim_feature.mean():.3f}  médiane={np.median(sim_feature):.3f}")

    # ── (b) GROUPE-À-GROUPE ──────────────────────────────────────────────
    com_a, communities_a = louvain_by_label_similarity(embs_a, keys_a)
    com_b, communities_b = louvain_by_label_similarity(embs_b, keys_b)
    print(f"[repro] (b) {len(communities_a)} groupes (seed 42), {len(communities_b)} groupes (seed 123).")

    def centroid(embs, keys, keep_keys):
        idx = [keys.index(k) for k in keep_keys]
        return embs[idx].mean(axis=0)

    centroids_a = np.stack([centroid(embs_a, keys_a, list(c)) for c in communities_a])
    centroids_b = np.stack([centroid(embs_b, keys_b, list(c)) for c in communities_b])
    centroids_a /= np.linalg.norm(centroids_a, axis=1, keepdims=True)
    centroids_b /= np.linalg.norm(centroids_b, axis=1, keepdims=True)
    sim_group = best_match_similarities(centroids_a, centroids_b)
    print(f"[repro] (b) Appariement GROUPE-À-GROUPE : "
          f"moyenne={sim_group.mean():.3f}  médiane={np.median(sim_group):.3f}")

    # ── (a) vs (b), même métrique (similarité cosinus) ──────────────────
    u_stat, mw_p = mannwhitneyu(sim_group, sim_feature, alternative="greater")
    print(f"\n[repro] Mann-Whitney U (H1 : similarité groupe > similarité feature) : "
          f"U={u_stat:.0f}  p={mw_p:.4f}")

    results = {
        "run_a": RUN_A, "run_b": RUN_B, "seed_a": 42, "seed_b": 123,
        "n_interpretable_a": len(labels_a), "n_interpretable_b": len(labels_b),
        "exact_label_overlap": exact_overlap,
        "feature_level": {"mean_sim": float(sim_feature.mean()), "median_sim": float(np.median(sim_feature)),
                           "n": len(sim_feature)},
        "group_level": {"mean_sim": float(sim_group.mean()), "median_sim": float(np.median(sim_group)),
                         "n_groups_a": len(communities_a), "n_groups_b": len(communities_b)},
        "mannwhitney_group_gt_feature": {"statistic": float(u_stat), "p": float(mw_p)},
        "sim_edge_threshold": SIM_EDGE_THRESHOLD,
        "pivot_note": "Approche activation/NPMI initiale infaisable (cache purgé + 0 overlap "
                       "features jugées / nœuds graphe NPMI) -- pivot vers similarité de label, cf. docstring.",
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Écrit : {OUT_PATH}")


if __name__ == "__main__":
    main()
