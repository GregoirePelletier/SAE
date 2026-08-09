"""
scripts/compare_to_frozen_benchmark.py — comparaison à un baseline GELÉ,
plutôt qu'au "run principal" courant (dérive silencieuse identifiée dans
l'audit méthodologique du 2026-08-07 : chaque ablation de ce dépôt s'est
jusqu'ici comparée à `results_v10_emails_main` en le relisant à chaque fois,
sans jamais figer un point de référence versionné — si ce run venait à
changer ou être remplacé, toutes les comparaisons passées perdraient leur
sens sans qu'on s'en aperçoive).

`benchmarks/frozen_baseline_v10_emails_main.json` fige les métriques clés du
run principal (SEED=42) au 2026-08-07. Ce script compare un NOUVEAU run à ce
baseline figé (jamais au run principal courant relu depuis le disque) :
  - `extension_interp_rate` : test à deux proportions + h de Cohen
    (`src/analysis/stats.py::two_proportion_test`) — la seule métrique dont
    on connaît le n, donc la seule testable statistiquement au sens strict.
  - Les autres métriques (rho_sae, fve_pretrained, silhouette, dead_pct,
    l0_mean, clf_acc) : dérive en % par rapport au point figé, seuil
    d'alerte explicite (WARN_THRESHOLD_PCT) plutôt qu'un jugement qualitatif
    au cas par cas.

Usage :
    PYTHONPATH=. .venv/bin/python scripts/compare_to_frozen_benchmark.py <run_dir>

    ex. PYTHONPATH=. .venv/bin/python scripts/compare_to_frozen_benchmark.py results_v13_ablation_seed123
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.stats import two_proportion_test

FROZEN_PATH = os.path.join(os.path.dirname(__file__), "..", "benchmarks",
                            "frozen_baseline_v10_emails_main.json")
WARN_THRESHOLD_PCT = 5.0    # dérive relative au-delà de laquelle on signale (arbitraire mais explicite)
FAIL_THRESHOLD_PCT = 15.0

CONTINUOUS_METRICS = ["rho_sae", "fve_pretrained", "silhouette_email_axes",
                       "dead_pct", "l0_mean", "clf_acc_sae_diffcorpus"]


def load_run_metrics(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "results.json"), encoding="utf-8") as f:
        results = json.load(f)
    p1 = results.get("P1_Gemma3_SAE", {})
    metrics = {
        "rho_sae": p1.get("rho_sae"),
        "fve_pretrained": p1.get("fve_pretrained"),
        "silhouette_email_axes": p1.get("silhouette"),
        "dead_pct": p1.get("dead_pct"),
        "l0_mean": p1.get("L0"),
        "active_features": p1.get("active_features"),
        "clf_acc_sae_diffcorpus": p1.get("clf_acc_sae"),
    }
    judge_cache = os.path.join(run_dir, "cache", "p1_judge_labels_extended.json")
    if os.path.exists(judge_cache):
        with open(judge_cache, encoding="utf-8") as f:
            judge_data = json.load(f)
        n_interp = sum(1 for v in judge_data.values() if v.get("interp_score") == 1)
        metrics["extension_interp_rate"] = {"n_interp": n_interp, "n_total": len(judge_data),
                                             "rate": n_interp / len(judge_data)}
    return metrics


def classify_drift(pct_diff: float) -> str:
    a = abs(pct_diff)
    if a >= FAIL_THRESHOLD_PCT:
        return "FAIL"
    if a >= WARN_THRESHOLD_PCT:
        return "WARN"
    return "OK"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Dossier du run à comparer (ex. results_v13_ablation_seed123)")
    args = parser.parse_args()

    with open(FROZEN_PATH, encoding="utf-8") as f:
        frozen = json.load(f)
    frozen_metrics = frozen["metrics"]
    print(f"[bench] Baseline gelé : {frozen['source_run']} ({frozen['frozen_on']})")
    print(f"[bench] Comparé à : {args.run_dir}\n")

    new_metrics = load_run_metrics(args.run_dir)

    print(f"{'Métrique':<28} {'Gelé':>12} {'Nouveau':>12} {'Dérive':>10} {'Statut':>7}")
    print("-" * 74)

    if "extension_interp_rate" in new_metrics:
        f_ir = frozen_metrics["extension_interp_rate"]
        n_ir = new_metrics["extension_interp_rate"]
        res = two_proportion_test(n_ir["n_interp"], n_ir["n_total"], f_ir["n_interp"], f_ir["n_total"])
        status = "FAIL" if res.p < 0.01 else ("WARN" if res.p < 0.05 else "OK")
        print(f"{'extension_interp_rate':<28} {f_ir['rate']:>11.1%} {n_ir['rate']:>11.1%} "
              f"{res.diff:>+9.1%} {status:>7}  (p={res.p:.3f}, h={res.cohens_h:.3f})")

    for key in CONTINUOUS_METRICS:
        f_val, n_val = frozen_metrics.get(key), new_metrics.get(key)
        if f_val is None or n_val is None:
            print(f"{key:<28} {'--':>12} {'--':>12} {'N/A':>10} {'SKIP':>7}")
            continue
        pct_diff = 100 * (n_val - f_val) / (abs(f_val) + 1e-12)
        status = classify_drift(pct_diff)
        print(f"{key:<28} {f_val:>12.4f} {n_val:>12.4f} {pct_diff:>+9.1f}% {status:>7}")

    print("\n[bench] OK < ±5% | WARN 5-15% (ou p<0.05 pour interp_rate) | FAIL > ±15% (ou p<0.01)")


if __name__ == "__main__":
    main()
