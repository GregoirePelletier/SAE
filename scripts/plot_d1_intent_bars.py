"""
scripts/plot_d1_intent_bars.py -- Figure statique (PNG) : accuracy SAE contre
TF-IDF+LogReg sur les 5 intentions client (D1, RESULTS_TESTS.md S104), avec
astérisque pour les écarts significatifs après correction BH (McNemar
apparié). Valeurs déjà mesurées et documentées, aucun calcul GPU.

Usage : .venv/bin/python scripts/plot_d1_intent_bars.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_PATH = os.path.join(REPO_ROOT, "report", "figures", "d1_intent_bars.png")

# (intention, acc_SAE, acc_TFIDF, p_BH significatif ?)
DATA = [
    ("Réclamation", 98.0, 97.6, False),
    ("Résiliation", 98.2, 96.0, True),
    ("Remboursement", 94.8, 90.4, True),
    ("Information", 87.9, 85.6, True),
    ("Urgence", 95.8, 95.8, False),
]


def main():
    labels = [d[0] for d in DATA]
    sae = [d[1] for d in DATA]
    tfidf = [d[2] for d in DATA]
    sig = [d[3] for d in DATA]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, sae, w, label="Codes SAE (R0)", color="#2A6F97")
    ax.bar(x + w / 2, tfidf, w, label="TF-IDF + LogReg", color="#A9A9A9")

    for i, is_sig in enumerate(sig):
        if is_sig:
            top = max(sae[i], tfidf[i])
            ax.annotate("*", (x[i], top), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=16, fontweight="bold")

    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(80, 101)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("D1 --- Détection d'intention : codes SAE contre TF-IDF+LogReg (n=3300)\n* : écart significatif, McNemar apparié, correction BH")
    ax.legend(loc="lower left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200)
    print(f"Écrit : {OUT_PATH}")


if __name__ == "__main__":
    main()
