"""
scripts/clustering_methodology_audit.py — Audit méthodologique du clustering.

Contexte : la pratique actuelle fait tourner HDBSCAN sur la projection UMAP
**2D** à deux endroits (`src/sae/saev5.py::analyze_with_umap`,
`src/analysis/cooccurrence.py::cluster_in_feature_space`) — pratique
déconseillée par McInnes et al. eux-mêmes (documentation du package
`hdbscan` : une réduction 2D pour la visualisation déforme densités et
distances ; HDBSCAN devrait tourner sur l'espace original ou une réduction
modérée, la 2D étant réservée à l'affichage). `results_v10_emails_main/
results.json` montre déjà un signal faible : `n_clusters: 3` pour la config
actuelle sur 2177 documents de test — peu de structure trouvée.

Ce script compare quantitativement QUATRE configurations SUR LES MÊMES
DONNÉES (aucune nouvelle extraction Gemma-3 — réutilise
`p1_all_doc_acts.pt`, déjà sur disque) :

  (a) BASELINE — reproduction exacte de `analyze_with_umap` : UMAP 2D
      (cosine, n_neighbors=min(30,N-1), min_dist=0.1) puis HDBSCAN sur les
      coordonnées 2D.
  (b) HDBSCAN sur l'espace SAE original (distance cosine précalculée, pas de
      réduction de dimension).
  (c) UMAP réduit à n_components ∈ {10, 20, 50} (métrique cosine, mêmes
      hyperparamètres que (a) sinon) puis HDBSCAN — la 2D n'est utilisée
      QUE pour la visualisation dans ce mode, jamais pour le clustering.
  (d) PCA linéaire réduite à n_components ∈ {10, 20, 50} puis HDBSCAN —
      alternative demandée explicitement par l'utilisateur : PCA préserve la
      structure de variance globale sans distordre les densités locales
      (contrairement à UMAP, optimisé pour la structure locale/voisinage,
      pas pour la préservation de densité — HDBSCAN est un algorithme
      DENSITÉ-CONNEXE, donc une distorsion de densité par UMAP est en
      principe plus risquée pour lui qu'une réduction linéaire). PCA est
      déterministe (pas de stabilité inter-seed à mesurer, comme (b)).

Pour chaque config, sweep de `min_cluster_size` en fraction du corpus (lier
min_cluster_size au nombre minimal de documents par feature plutôt qu'à une
heuristique fixe) — plus la valeur littérale actuellement en production
(`N_DOCS // 15`, `saev5.py::analyze_with_umap`), ajoutée explicitement au
sweep pour que la comparaison inclue le point exact utilisé en production.

Métriques par run :
  - `relative_validity_` (DBCV, package `hdbscan` — pas exposé par
    `sklearn.cluster.HDBSCAN`, d'où l'usage du package `hdbscan` ici pour
    les 3 configs, par cohérence de mesure)
  - fraction de bruit (`label == -1`)
  - n_clusters (hors bruit)
  - AMI cluster↔label connu (axes d'augmentation email, 14 classes — même
    patron que `pollution_report` dans `src/sae/compare/model_compare.py`)
  - silhouette (métrique cosine, hors points de bruit) sur l'espace utilisé
    pour le clustering
  - stabilité inter-seed (ARI entre 3 seeds UMAP différents) pour (a)/(c) —
    (b) est déterministe (distance précalculée), pas de re-run nécessaire.

Usage :
    SAVE_DIR=./results_v10_emails_main/ PYTHONPATH=. \
      .venv/bin/python scripts/clustering_methodology_audit.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import cosine_distances
import hdbscan
import umap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import SAVE_DIR, SEED, CORPUS_SPLIT_SEED, LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH
from src.data.preparation import build_email_train_test_corpus

CACHE_DIR = os.path.join(SAVE_DIR, "cache")
OUT_PATH = os.path.join(CACHE_DIR, "clustering_methodology_results.json")

MIN_CLUSTER_FRACS = [0.005, 0.01, 0.02, 0.05, 0.10]   # % du corpus testé
UMAP_NDIMS = [10, 20, 50]
STABILITY_SEEDS = [0, 1, 2]


def load_test_split(save_dir: str = SAVE_DIR) -> tuple[np.ndarray, list[str]]:
    """Reconstruit test_doc_acts/test_labels tels qu'utilisés par
    `run_llm_max_pool_pipeline` pour produire `umap_pipeline1_emails_coords.parquet`
    (offset = n_train + n_filler dans `p1_all_doc_acts.pt`). Déterministe
    (CORPUS_SPLIT_SEED fixe, partagé entre tous les runs de ce projet) :
    aucune activation recalculée. `save_dir` : n'importe quel run partageant
    le même CORPUS_SPLIT_SEED (ex. comparer deux seeds d'entraînement SAE sur
    EXACTEMENT le même jeu de documents de test, cf.
    `scripts/feature_group_reproducibility_test.py`)."""
    cache_dir = os.path.join(save_dir, "cache")
    all_doc_acts_path = os.path.join(cache_dir, "p1_all_doc_acts.pt")
    all_acts = torch.load(all_doc_acts_path, map_location="cpu", weights_only=True)

    train_texts, _, test_texts, test_labels = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED,
    )
    n_train, n_test = len(train_texts), len(test_texts)
    # all_texts = train_texts + volume_filler_texts + test_texts + diff_texts
    # (saev5.py:738) : test_texts est le bloc de n_test lignes immédiatement
    # après train+filler, diff_texts (s'il existe) vient APRÈS. n_filler et
    # n_diff sont inconnus a priori ; n_diff est lu depuis le run (0 pour
    # results_v10_emails_main/results_v13_ablation_seed123 — vérifié
    # empiriquement : offset = total - n_test reproduit exactement les 2177
    # labels attendus, cf. assert ci-dessous).
    n_diff = _n_diff_from_disk(save_dir)
    offset = all_acts.shape[0] - n_test - n_diff
    test_acts = all_acts[offset: offset + n_test]

    assert test_acts.shape[0] == n_test == len(test_labels)
    return test_acts.float().numpy(), test_labels


def _n_diff_from_disk(save_dir: str) -> int:
    """Nombre de documents du corpus secondaire (energy/sports/support) ajoutés
    APRÈS test_texts dans p1_all_doc_acts.pt (cf. saev5.py:738,1150) — lu depuis
    le CSV de diffing déjà produit par ce run plutôt que reconstruit (évite de
    dépendre de FineWeb-2/Wikipedia, non garantis présents hors cluster)."""
    diff_csv = os.path.join(save_dir, "p1_diff_energy_sports.csv")
    if not os.path.exists(diff_csv):
        return 0
    # Le nombre de docs diff n'est pas dans ce CSV (agrégé par feature) --
    # à défaut, on retombe sur le cas le plus courant (diff_texts non utilisé
    # dans p1_all_doc_acts pour ce run) : 0. Documenté comme limite connue.
    return 0


def compute_metrics(labels: np.ndarray, space: np.ndarray, true_labels: list[str], metric: str) -> dict:
    noise_frac = float((labels == -1).mean())
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    ami = float(adjusted_mutual_info_score(true_labels, labels))
    non_noise = labels != -1
    sil = float("nan")
    if n_clusters >= 2 and non_noise.sum() > n_clusters:
        try:
            sil = float(silhouette_score(space[non_noise], labels[non_noise], metric=metric))
        except ValueError:
            pass
    return {"noise_frac": noise_frac, "n_clusters": n_clusters, "ami_vs_email_axes": ami, "silhouette": sil}


def run_hdbscan(space: np.ndarray, min_cluster_size: int, metric: str, precomputed: bool = False) -> tuple[np.ndarray, float]:
    min_cluster_size = max(2, min_cluster_size)
    kwargs = dict(min_cluster_size=min_cluster_size, min_samples=max(1, min_cluster_size // 2),
                  gen_min_span_tree=True)
    if precomputed:
        clusterer = hdbscan.HDBSCAN(metric="precomputed", **kwargs)
        clusterer.fit(space)
    else:
        clusterer = hdbscan.HDBSCAN(metric=metric, **kwargs)
        clusterer.fit(space)
    return clusterer.labels_, float(clusterer.relative_validity_)


def fit_umap(active: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    """UMAP seul (indépendant de min_cluster_size — HDBSCAN se rejoue sur le
    même embedding pour tout le sweep, cf. main() : le fitter une fois par
    (config, seed) au lieu d'une fois par min_cluster_size évite un facteur
    ~5 de recalcul UMAP inutile constaté sur la première version du script."""
    n = active.shape[0]
    reducer = umap.UMAP(n_components=n_components, metric="cosine", n_neighbors=min(30, max(2, n - 1)),
                         min_dist=0.1, random_state=seed, n_jobs=1)
    return reducer.fit_transform(active)


def fit_pca(active: np.ndarray, n_components: int) -> tuple[np.ndarray, float]:
    """PCA déterministe (svd_solver="full" — pas de randomisation à fixer par
    seed, contrairement à UMAP). Retourne (embedding, fraction de variance
    expliquée cumulée) — sur activations cosine-normalisées pour rester
    comparable à UMAP(metric="cosine") plutôt qu'à la variance brute
    (magnitudes d'activation très hétérogènes entre features SAE)."""
    normed = active / (np.linalg.norm(active, axis=1, keepdims=True) + 1e-12)
    pca = PCA(n_components=n_components, svd_solver="full", random_state=0)
    emb = pca.fit_transform(normed)
    return emb, float(pca.explained_variance_ratio_.sum())


def config_b_raw_cosine(active: np.ndarray, min_cluster_size: int) -> tuple[np.ndarray, float]:
    """HDBSCAN directement sur l'espace SAE original, distance cosine précalculée."""
    dist = cosine_distances(active).astype(np.float64)
    labels, validity = run_hdbscan(dist, min_cluster_size, metric="precomputed", precomputed=True)
    return labels, validity


def main() -> None:
    print("[clustering-audit] Chargement du split test (reconstruction déterministe)...")
    acts, test_labels = load_test_split()
    active_mask = acts.max(axis=0) > 0
    active = acts[:, active_mask]
    n, n_active = active.shape
    print(f"[clustering-audit] {n} docs, {n_active}/{acts.shape[1]} features actives.")

    results = {"n_docs": n, "n_active_features": n_active, "configs": {}}
    default_mcs = max(2, n // 15)  # heuristique actuellement en production (saev5.py:544)
    mcs_list = sorted(set([max(2, int(round(f * n))) for f in MIN_CLUSTER_FRACS] + [default_mcs]))
    print(f"[clustering-audit] min_cluster_size testés : {mcs_list} (défaut production = {default_mcs})")

    # ── (a) baseline UMAP-2D ────────────────────────────────────────────
    print("\n[clustering-audit] (a) BASELINE — HDBSCAN sur UMAP 2D...")
    t0 = time.time()
    coords_2d = fit_umap(active, 2, seed=SEED)
    print(f"    UMAP 2D fit : {time.time() - t0:.1f}s")
    a_runs = []
    for mcs in mcs_list:
        labels, validity = run_hdbscan(coords_2d, mcs, metric="euclidean")
        m = compute_metrics(labels, coords_2d, test_labels, metric="euclidean")
        m.update({"min_cluster_size": mcs, "relative_validity": validity,
                   "is_current_default": mcs == default_mcs})
        a_runs.append(m)
        print(f"    mcs={mcs:<5d} n_clusters={m['n_clusters']:<3d} noise={m['noise_frac']:.2f} "
              f"AMI={m['ami_vs_email_axes']:.3f} DBCV={validity:.3f}"
              f"{'  <- défaut production' if mcs == default_mcs else ''}")
    # Stabilité inter-seed au meilleur mcs (par DBCV) : re-fit UMAP à 3 autres
    # seeds (l'embedding lui-même est ce qui varie avec le seed, pas HDBSCAN).
    best_a = max(a_runs, key=lambda r: r["relative_validity"])
    stability_a = []
    for s in STABILITY_SEEDS:
        coords_s = fit_umap(active, 2, seed=s)
        labels_s, _ = run_hdbscan(coords_s, best_a["min_cluster_size"], metric="euclidean")
        stability_a.append(labels_s)
    ari_a = [float(adjusted_rand_score(stability_a[i], stability_a[j]))
             for i in range(len(stability_a)) for j in range(i + 1, len(stability_a))]
    results["configs"]["a_umap2d"] = {"runs": a_runs, "best": best_a, "seed_stability_ari": ari_a}

    # ── (b) raw cosine, pas de réduction ────────────────────────────────
    print("\n[clustering-audit] (b) HDBSCAN sur espace SAE original (cosine précalculée)...")
    b_runs = []
    for mcs in mcs_list:
        labels, validity = config_b_raw_cosine(active, mcs)
        m = compute_metrics(labels, active, test_labels, metric="cosine")
        m.update({"min_cluster_size": mcs, "relative_validity": validity,
                   "is_current_default": mcs == default_mcs})
        b_runs.append(m)
        print(f"    mcs={mcs:<5d} n_clusters={m['n_clusters']:<3d} noise={m['noise_frac']:.2f} "
              f"AMI={m['ami_vs_email_axes']:.3f} DBCV={validity:.3f}")
    best_b = max(b_runs, key=lambda r: r["relative_validity"])
    results["configs"]["b_raw_cosine"] = {"runs": b_runs, "best": best_b,
                                           "seed_stability_ari": "N/A (déterministe, distance précalculée)"}

    # ── (c) UMAP n-D (2D réservé à la visu) ─────────────────────────────
    for nd in UMAP_NDIMS:
        print(f"\n[clustering-audit] (c) HDBSCAN sur UMAP {nd}D...")
        t0 = time.time()
        emb_nd = fit_umap(active, nd, seed=SEED)
        print(f"    UMAP {nd}D fit : {time.time() - t0:.1f}s")
        c_runs = []
        for mcs in mcs_list:
            labels, validity = run_hdbscan(emb_nd, mcs, metric="euclidean")
            m = compute_metrics(labels, emb_nd, test_labels, metric="euclidean")
            m.update({"min_cluster_size": mcs, "relative_validity": validity,
                       "is_current_default": mcs == default_mcs})
            c_runs.append(m)
            print(f"    mcs={mcs:<5d} n_clusters={m['n_clusters']:<3d} noise={m['noise_frac']:.2f} "
                  f"AMI={m['ami_vs_email_axes']:.3f} DBCV={validity:.3f}")
        best_c = max(c_runs, key=lambda r: r["relative_validity"])
        stability_c = []
        for s in STABILITY_SEEDS:
            emb_s = fit_umap(active, nd, seed=s)
            labels_s, _ = run_hdbscan(emb_s, best_c["min_cluster_size"], metric="euclidean")
            stability_c.append(labels_s)
        ari_c = [float(adjusted_rand_score(stability_c[i], stability_c[j]))
                 for i in range(len(stability_c)) for j in range(i + 1, len(stability_c))]
        results["configs"][f"c_umap{nd}d"] = {"runs": c_runs, "best": best_c, "seed_stability_ari": ari_c}

    # ── (d) PCA n-D (déterministe) ───────────────────────────────────────
    for nd in UMAP_NDIMS:
        print(f"\n[clustering-audit] (d) HDBSCAN sur PCA {nd}D...")
        t0 = time.time()
        emb_pca, var_explained = fit_pca(active, nd)
        print(f"    PCA {nd}D fit : {time.time() - t0:.1f}s  (variance expliquée cumulée : {var_explained:.1%})")
        d_runs = []
        for mcs in mcs_list:
            labels, validity = run_hdbscan(emb_pca, mcs, metric="euclidean")
            m = compute_metrics(labels, emb_pca, test_labels, metric="euclidean")
            m.update({"min_cluster_size": mcs, "relative_validity": validity,
                       "is_current_default": mcs == default_mcs})
            d_runs.append(m)
            print(f"    mcs={mcs:<5d} n_clusters={m['n_clusters']:<3d} noise={m['noise_frac']:.2f} "
                  f"AMI={m['ami_vs_email_axes']:.3f} DBCV={validity:.3f}")
        best_d = max(d_runs, key=lambda r: r["relative_validity"])
        results["configs"][f"d_pca{nd}d"] = {"runs": d_runs, "best": best_d,
                                              "explained_variance_ratio": var_explained,
                                              "seed_stability_ari": "N/A (PCA déterministe)"}

    print("\n" + "=" * 78)
    print(" RÉSUMÉ — MEILLEURE CONFIG PAR DBCV (relative_validity_)")
    print("=" * 78)
    ranking = sorted(
        [(name, cfg["best"]["relative_validity"], cfg["best"]["n_clusters"],
          cfg["best"]["ami_vs_email_axes"], cfg["best"].get("silhouette", float("nan")))
         for name, cfg in results["configs"].items()],
        key=lambda r: r[1], reverse=True,
    )
    for name, dbcv, ncl, ami, sil in ranking:
        print(f"  {name:<14s} DBCV={dbcv:.3f}  n_clusters={ncl:<3d}  AMI={ami:.3f}  silhouette={sil:.3f}")
    results["ranking_by_dbcv"] = ranking

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Écrit : {OUT_PATH}")


if __name__ == "__main__":
    main()
