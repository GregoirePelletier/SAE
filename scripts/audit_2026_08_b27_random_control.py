"""
scripts/audit_2026_08_b27_random_control.py — B.27 de `docs/AUDIT_2026-08.md` :
`feature_group_reproducibility_test.py` conclut que le groupement de features
par similarité de label améliore la reproductibilité inter-seed (similarité
GROUPE-À-GROUPE > FEATURE-À-FEATURE, Mann-Whitney). Mais moyenner plusieurs
vecteurs avant de comparer leur similarité cosinus AUGMENTE MÉCANIQUEMENT la
similarité par réduction de variance -- un effet géométrique pur, indépendant
de toute structure sémantique réelle. Le test original ne distingue pas "le
regroupement capture une vraie robustesse conceptuelle" de "moyenner des
vecteurs les rapproche toujours un peu plus".

Ce script ajoute le témoin manquant : mêmes features, mêmes labels, mêmes
embeddings bge-m3 (aucun recalcul), mais regroupées AU HASARD en groupes de
même distribution de tailles que les communautés Louvain réelles. Si le
groupe réel ne bat pas significativement le groupe aléatoire, l'effet observé
dans le script original n'est que l'artefact du moyennage.

Ne modifie PAS `feature_group_reproducibility_test.py` (résultat déjà publié,
cf. RESULTS_TESTS.md) : script séparé réutilisant les mêmes fonctions.

Usage : sbatch slurm/validation/run_audit_b27_random_control.slurm
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "sae"))

from scripts.feature_group_reproducibility_test import (
    RUN_A, RUN_B, load_interpretable_labels, louvain_by_label_similarity,
    best_match_similarities,
)

OUT_PATH = "./results_v10_emails_main/cache/audit_2026_08_b27_random_control_results.json"
SEED = 42
N_RANDOM_TRIALS = 200  # tirages de partitions aléatoires, pour une distribution stable


def random_partition_like(keys: list[str], community_sizes: list[int], rng: np.random.Generator) -> list[list[str]]:
    """Partitionne `keys` au hasard en groupes de tailles EXACTEMENT
    `community_sizes` (même distribution que les communautés Louvain réelles,
    somme des tailles == len(keys) par construction de l'appelant)."""
    order = rng.permutation(keys)
    groups, i = [], 0
    for size in community_sizes:
        groups.append(list(order[i:i + size]))
        i += size
    return groups


def centroid_sims_for_partition(embs: np.ndarray, keys: list[str], groups: list[list[str]]) -> np.ndarray:
    idx_map = {k: i for i, k in enumerate(keys)}
    centroids = np.stack([
        embs[[idx_map[k] for k in g]].mean(axis=0) for g in groups
    ])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    return centroids


def main() -> None:
    print("[b27-control] Chargement des labels interprétables (identique au script original)...")
    labels_a = load_interpretable_labels(RUN_A)
    labels_b = load_interpretable_labels(RUN_B)

    from src.sae.saev5 import _embed_bge_m3
    keys_a, keys_b = list(labels_a), list(labels_b)
    print(f"[b27-control] Embedding bge-m3 de {len(keys_a) + len(keys_b)} labels...")
    all_embs = _embed_bge_m3([labels_a[k] for k in keys_a] + [labels_b[k] for k in keys_b]).numpy()
    embs_a, embs_b = all_embs[:len(keys_a)], all_embs[len(keys_a):]

    sim_feature = best_match_similarities(embs_a, embs_b)
    print(f"[b27-control] (a) FEATURE-À-FEATURE : moyenne={sim_feature.mean():.3f}")

    _, communities_a = louvain_by_label_similarity(embs_a, keys_a)
    _, communities_b = louvain_by_label_similarity(embs_b, keys_b)
    sizes_a = [len(c) for c in communities_a]
    sizes_b = [len(c) for c in communities_b]

    def centroid(embs, keys, keep_keys):
        idx = [keys.index(k) for k in keep_keys]
        return embs[idx].mean(axis=0)

    centroids_a_real = np.stack([centroid(embs_a, keys_a, list(c)) for c in communities_a])
    centroids_b_real = np.stack([centroid(embs_b, keys_b, list(c)) for c in communities_b])
    centroids_a_real /= np.linalg.norm(centroids_a_real, axis=1, keepdims=True)
    centroids_b_real /= np.linalg.norm(centroids_b_real, axis=1, keepdims=True)
    sim_group_real = best_match_similarities(centroids_a_real, centroids_b_real)
    print(f"[b27-control] (b) GROUPE-À-GROUPE RÉEL (Louvain) : moyenne={sim_group_real.mean():.3f}")

    print(f"[b27-control] (c) GROUPE-À-GROUPE ALÉATOIRE ({N_RANDOM_TRIALS} tirages, "
          f"même distribution de tailles : {len(sizes_a)} groupes seed42 {sizes_a}, "
          f"{len(sizes_b)} groupes seed123 {sizes_b})...")
    rng = np.random.default_rng(SEED)
    sim_group_random_all = []
    for _ in range(N_RANDOM_TRIALS):
        groups_a_rand = random_partition_like(keys_a, sizes_a, rng)
        groups_b_rand = random_partition_like(keys_b, sizes_b, rng)
        centroids_a_rand = centroid_sims_for_partition(embs_a, keys_a, groups_a_rand)
        centroids_b_rand = centroid_sims_for_partition(embs_b, keys_b, groups_b_rand)
        sim_group_random_all.append(best_match_similarities(centroids_a_rand, centroids_b_rand))
    sim_group_random = np.concatenate(sim_group_random_all)
    print(f"[b27-control] (c) moyenne sur {N_RANDOM_TRIALS} tirages : "
          f"{sim_group_random.mean():.3f} (n={len(sim_group_random)} paires au total)")

    u_real_vs_feature, p_real_vs_feature = mannwhitneyu(
        sim_group_real, sim_feature, alternative="greater")
    u_real_vs_random, p_real_vs_random = mannwhitneyu(
        sim_group_real, sim_group_random, alternative="greater")
    u_random_vs_feature, p_random_vs_feature = mannwhitneyu(
        sim_group_random, sim_feature, alternative="greater")

    results = {
        "n_random_trials": N_RANDOM_TRIALS,
        "feature_level": {"mean_sim": float(sim_feature.mean()), "n": len(sim_feature)},
        "group_real_level": {"mean_sim": float(sim_group_real.mean()), "n": len(sim_group_real),
                              "sizes_a": sizes_a, "sizes_b": sizes_b},
        "group_random_level": {"mean_sim": float(sim_group_random.mean()), "n": len(sim_group_random)},
        "mannwhitney_group_real_gt_feature": {"statistic": float(u_real_vs_feature), "p": float(p_real_vs_feature)},
        "mannwhitney_group_real_gt_group_random": {"statistic": float(u_real_vs_random), "p": float(p_real_vs_random)},
        "mannwhitney_group_random_gt_feature": {"statistic": float(u_random_vs_feature), "p": float(p_random_vs_feature)},
    }
    print("\n" + "=" * 70)
    print(" RÉSUMÉ — B.27 : témoin aléatoire pour le test de regroupement")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("\n[b27-control] Interprétation : si groupe_réel > groupe_aléatoire n'est PAS "
          "significatif, l'écart groupe > feature du script original est un artefact de "
          "moyennage, pas une robustesse conceptuelle réelle du regroupement.")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Écrit : {OUT_PATH}")


if __name__ == "__main__":
    main()
