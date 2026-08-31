"""
scripts/plot_model_scale_curve.py -- Figure statique (PNG) : taux
d'interprétabilité en fonction de l'échelle du modèle extracteur/juge
(1B/4B/12B/27B), avec IC95% de Wilson, sous méthodologie totalement
homogène (stratifié, juge Qwen3.8-27B, négatif corrigé). Lit uniquement
p1_judge_labels_extended.json déjà en cache -- aucun calcul GPU.

À exécuter UNE FOIS la campagne de rejugement n=300 terminée pour les 4
points d'échelle (1B/4B/12B/27B) -- sinon lit les caches n=150 pré-correctif
encore présents et produit une figure provisoire non représentative.

Usage : .venv/bin/python scripts/plot_model_scale_curve.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.analysis.stats import proportion_with_ci
OUT_PATH = os.path.join(REPO_ROOT, "report", "figures", "model_scale_curve.png")

# (label, SAVE_DIR, taille en Md de paramètres pour l'axe x)
RUNS = [
    ("1B", "results_v29_ablation_classic_setup_k5_25m_model_scale_1b", 1),
    ("4B", "results_v30_ablation_classic_setup_k5_25m_model_scale_4b", 4),
    ("12B (R0)", "results_v27_ablation_classic_setup_k5_25m_layer31", 12),
    ("27B", "results_v31_ablation_classic_setup_k5_25m_model_scale_27b", 27),
]


def load_rate(save_dir):
    path = os.path.join(REPO_ROOT, save_dir, "cache", "p1_judge_labels_extended.json")
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    n = len(d)
    succ = sum(1 for v in d.values() if v["interp_score"] == 1)
    return succ, n


def main():
    labels, xs, rates, los, his, ns = [], [], [], [], [], []
    for label, save_dir, x in RUNS:
        succ, n = load_rate(save_dir)
        r = proportion_with_ci(succ, n)
        labels.append(label)
        xs.append(x)
        rates.append(100 * r.rate)
        los.append(100 * (r.rate - r.ci_low))
        his.append(100 * (r.ci_high - r.rate))
        ns.append(n)
        print(f"  {label}: {succ}/{n} = {100*r.rate:.1f}% [{100*r.ci_low:.1f}, {100*r.ci_high:.1f}]")

    if len(set(ns)) > 1:
        print(f"  [WARN] n hétérogène entre points : {dict(zip(labels, ns))} -- "
              "vérifier que la campagne de rejugement est bien terminée partout.")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(xs, rates, yerr=[los, his], fmt="o-", color="#2A6F97",
                capsize=4, markersize=7, linewidth=1.5)
    for x, r, n in zip(xs, rates, ns):
        ax.annotate(f"{r:.1f}%\n(n={n})", (x, r), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=9)

    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Échelle du modèle extracteur/juge")
    ax.set_ylabel("Taux d'interprétabilité (%)")
    ax.set_title("Échelle du modèle, méthodologie totalement homogène\n(stratifié, Qwen3.8-27B, négatif corrigé, $n=300$) --- pas de tendance")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200)
    print(f"\nÉcrit : {OUT_PATH}")


if __name__ == "__main__":
    main()
