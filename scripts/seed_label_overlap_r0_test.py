"""
scripts/seed_label_overlap_r0_test.py — Recouvrement exact des labels de
features interprétables entre graines, sous méthodologie R0 (stratifié +
juge Qwen3.8-27B + déduplication par mail parent, K_EXTRA=5, D_EXTRA=1024,
25M tokens, layer 31, n=300).

Contexte : `RESULTS_TESTS.md` §21 mesure 22/78 = 28,2% de recouvrement EXACT
de labels entre SEED=42 et SEED=123, mais sous sélection par magnitude, juge
Gemma-3-12b-it auto-référent et n=150 — un protocole non R0
(`results_v10_emails_main` vs `results_v13_ablation_seed123`). Ce script
réplique la même métrique (réutilisée telle quelle depuis
`scripts/feature_group_reproducibility_test.py::exact_label_overlap`) sur les
trois graines déjà disponibles sous méthodologie pleinement corrigée : R0
(SEED=42, §97/§119), V1 (SEED=123, §99/§119) et V2 (SEED=7, §101/§119) —
mêmes valeurs de seed que §21 pour R0/V1, ce qui permet une comparaison
directe.

Coût : zéro calcul GPU/LLM — lit uniquement les JSON de labels déjà en cache
(`p1_top_extended_features.json`, identique à
`cache/p1_judge_labels_extended.json`, vérifié par diff) pour les trois runs.
Comparable au "smoke-test CPU-only sur cache" toléré hors sbatch (CLAUDE.md,
Cluster SLURM) : trois fichiers JSON de quelques centaines de Ko, aucune
lecture de tenseur/modèle/checkpoint.

Usage (CPU uniquement) :
    PYTHONPATH=. .venv/bin/python scripts/seed_label_overlap_r0_test.py
"""
from __future__ import annotations

import itertools
import json
import os

RUNS = {
    42: ("./results_v27_ablation_classic_setup_k5_25m_layer31", "R0"),
    123: ("./results_v38_ablation_v1_seed123_classic_setup_k5_25m_layer31", "V1"),
    7: ("./results_v39_ablation_v2_seed7_classic_setup_k5_25m_layer31", "V2"),
}
OUT_PATH = "./results_v27_ablation_classic_setup_k5_25m_layer31/cache/seed_label_overlap_r0_results.json"


def load_interpretable_labels(run_dir: str) -> dict[str, str]:
    with open(os.path.join(run_dir, "p1_top_extended_features.json"), encoding="utf-8") as f:
        data = json.load(f)
    return {k: v["label"] for k, v in data.items() if v.get("interp_score") == 1}


def main() -> None:
    labels_by_seed = {}
    for seed, (run_dir, tag) in RUNS.items():
        labels_by_seed[seed] = load_interpretable_labels(run_dir)
        print(f"[overlap] seed={seed} ({tag}, {run_dir}) : "
              f"{len(labels_by_seed[seed])} features interprétables.")

    pairwise = {}
    for (seed_a, seed_b) in itertools.combinations(RUNS.keys(), 2):
        set_a = set(labels_by_seed[seed_a].values())
        set_b = set(labels_by_seed[seed_b].values())
        inter = set_a & set_b
        union = set_a | set_b
        key = f"seed{seed_a}_vs_seed{seed_b}"
        pairwise[key] = {
            "tag_a": RUNS[seed_a][1], "tag_b": RUNS[seed_b][1],
            "n_distinct_labels_a": len(set_a), "n_distinct_labels_b": len(set_b),
            "n_intersection": len(inter), "n_union": len(union),
            "jaccard_pct": round(100 * len(inter) / len(union), 1) if union else float("nan"),
        }
        print(f"[overlap] seed{seed_a} ({RUNS[seed_a][1]}) vs seed{seed_b} ({RUNS[seed_b][1]}) : "
              f"{len(inter)}/{len(union)} = {pairwise[key]['jaccard_pct']}% (Jaccard, labels exacts)")

    results = {
        "protocol": "R0 (stratifié + Qwen3.8-27B + déduplication mail parent, K_EXTRA=5, "
                     "D_EXTRA=1024, 25M tokens, layer 31, n=300)",
        "reference_historical": "RESULTS_TESTS.md §21 : 22/78 = 28,2% sous magnitude + juge "
                                 "auto-référent, n=150, SEED=42 vs SEED=123",
        "n_interpretable_by_seed": {str(s): len(labels_by_seed[s]) for s in RUNS},
        "pairwise": pairwise,
        "metric_note": "Jaccard sur l'ensemble des chaînes de labels EXACTES (non normalisées) "
                        "des features interprétables (interp_score==1) — même métrique que "
                        "feature_group_reproducibility_test.py::exact_label_overlap.",
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Écrit : {OUT_PATH}")


if __name__ == "__main__":
    main()
