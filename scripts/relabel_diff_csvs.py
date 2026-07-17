"""
scripts/relabel_diff_csvs.py — Réattribue les labels Neuronpedia (cache local
canonique, cf. src/config.NEURONPEDIA_LABELS_PATH) aux CSV/HTML de diffing déjà
produits, sans rien recalculer (pas de GPU, pas de réextraction d'activations).

Contexte : les runs `scripts/baseline_gemmascope.py` (results_v9_test/cache_baseline*)
ont tourné avec le cluster hors-ligne -> Neuronpedia injoignable -> colonne `label`
des diff_*.csv remplie de F{idx} bruts (non-interprétables). Le cache local canonique
existe maintenant (local_data/neuronpedia_labels/) : on réapplique juste le mapping
feature_id -> label sur les fichiers déjà écrits.

Usage :
    python scripts/relabel_diff_csvs.py results_v9_test/cache_baseline
    python scripts/relabel_diff_csvs.py results_v9_test/cache_baseline_full
"""
from __future__ import annotations

import glob
import os
import sys

import pandas as pd

from src.config import NEURONPEDIA_LABELS_PATH
from src.sae.neuronpedia_labels import fetch_neuronpedia_labels
from src.analysis.visualization import plot_corpus_diff


def relabel_dir(cache_dir: str) -> None:
    labels = fetch_neuronpedia_labels(cache_path=NEURONPEDIA_LABELS_PATH)
    if not labels:
        print(f"[relabel] WARN: {NEURONPEDIA_LABELS_PATH} vide/absent -- rien à faire.")
        return

    csv_paths = sorted(glob.glob(os.path.join(cache_dir, "diff_*.csv")))
    if not csv_paths:
        print(f"[relabel] Aucun diff_*.csv trouvé sous {cache_dir}.")
        return

    n_relabeled = 0
    for path in csv_paths:
        df = pd.read_csv(path)
        if "feature_id" not in df.columns:
            continue
        new_labels = df["feature_id"].map(lambda i: labels.get(int(i), f"F{int(i)}"))
        changed = (new_labels != df["label"]).sum()
        df["label"] = new_labels
        df.to_csv(path, index=False)
        html_path = path[:-4] + ".html"
        plot_corpus_diff(df, path=html_path)
        n_relabeled += changed
        print(f"[relabel] {os.path.basename(path)}: {changed}/{len(df)} labels mis à jour.")

    print(f"[relabel] Terminé : {len(csv_paths)} fichiers, {n_relabeled} labels changés au total "
          f"({len(labels)} labels Neuronpedia disponibles).")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        relabel_dir(d)
