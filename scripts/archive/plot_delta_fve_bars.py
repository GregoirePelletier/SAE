"""
scripts/plot_delta_fve_bars.py -- Figure statique (PNG) : gain de variance
expliquée (Delta FVE) du SAE entraine contre les deux sanity checks a
decodeur fige aleatoire (Korznikov et al. 2026), RESULTS_TESTS.md
S97/S105/S115. Valeurs deja mesurees et documentees, aucun calcul GPU.

Usage : .venv/bin/python scripts/plot_delta_fve_bars.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_PATH = os.path.join(REPO_ROOT, "report", "figures", "delta_fve_bars.png")

# RESULTS_TESTS.md S97 (R0), S98/S115 (C1, init iso), S105 (C1b, init cov).
DATA = [
    ("R0\n(SAE entraîné)", 0.1393),
    ("C1b\n(décodeur figé, init cov)", 0.0086),
    ("C1\n(décodeur figé, init iso)", 0.0001),
]

def main():
    labels = [d[0] for d in DATA]
    values = [d[1] for d in DATA]
    colors = ["#2A6F97", "#A9A9A9", "#A9A9A9"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for bar, v in zip(bars, values):
        ax.annotate(f"+{v:.4f}", (bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)

    ax.set_ylabel(r"$\Delta$FVE (gain de variance expliquée)")
    ax.set_title("Variance expliquée gagnée par l'extension,\nSAE entraîné contre décodeur figé aléatoire")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(values) * 1.25)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200)
    print(f"Écrit : {OUT_PATH}")


if __name__ == "__main__":
    main()
