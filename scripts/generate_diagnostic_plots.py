"""
scripts/generate_diagnostic_plots.py — Agrégation RÉTROACTIVE des diagnostics.

Lit uniquement des artefacts déjà sur disque (aucun modèle chargé, aucun GPU,
aucun rerun) et produit, sous chaque `results_*/plots/` concerné, des figures
HTML autonomes (Plotly, via src/analysis/plotting.py) :

  1. Courbes d'entraînement Pipeline 1 (p1_extended_sae.pt -- le "history"
     est déjà embarqué dans TOUS les checkpoints existants, même ceux
     entraînés avant l'ajout du logging par step + validation loss,
     cf. sae_shared.py::load_or_train_extended_sae) et Pipeline 2
     (*_history.json, déjà par step depuis le début).
  2. Balayages d'hyperparamètres déjà menés (K_EXTRA, D_EXTRA, volume de
     tokens, layer, hook-point, échelle du modèle) : une figure consolidée
     par famille au lieu d'une table texte isolée par balayage dans
     RESULTS_TESTS.md. Le manifeste SWEEP_MANIFEST ci-dessous mappe
     run_dir -> valeur d'hyperparamètre (non stocké dans results.json,
     seulement dans les scripts SLURM/RESULTS_TESTS.md -- documenté ici une
     fois pour toutes).
  3. Distribution de rho_interp par statut d'interprétabilité (juge), à
     partir de p1_top_extended_features.json -- diagnostic direct de la
     question "qu'est-ce qui distingue une feature interprétée d'une qui ne
     l'est pas".

CE QUI N'EST PAS RÉTROACTIF (nécessite un rerun, pas fait ici) :
  - Heatmap de corrélation NPMI (RESULTS_TESTS.md §24) : aucune matrice de
    co-activation n'a jamais été persistée sur disque, seul un résumé texte
    existe. `scripts/compute_interesting_correlations_retro.py` calcule des
    corrélations mais pas de matrice complète top-10 -- à étendre.
  - Histogramme de sensibilité à l'ordre du juge (§13.1) : le script ayant
    produit ce résultat n'a jamais persisté les taux d'accord par feature,
    seul le résumé agrégé (31,3%) est dans RESULTS_TESTS.md.

Usage : .venv/bin/python scripts/generate_diagnostic_plots.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import pandas as pd
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.analysis.plotting import (
    plot_training_curves, plot_metric_vs_hyperparam, plot_activation_distribution,
)

# ─────────────────────────────────────────────────────────────────────────────
# Manifeste des balayages déjà menés : run_dir -> valeur d'hyperparamètre.
# Valeurs croisées avec RESULTS_TESTS.md (§17/§18.2/§23.4/§25/§27/§28/§51/§53)
# au moment de l'écriture -- à étendre au fil des nouveaux balayages (Phase 2/3).
# ─────────────────────────────────────────────────────────────────────────────
SWEEP_MANIFEST = {
    "k_extra": {
        "x_label": "K_EXTRA",
        "runs": [("results_v10_emails_main", 32), ("results_v13_ablation_k_extra5", 5)],
    },
    "d_extra": {
        "x_label": "D_EXTRA",
        "runs": [("results_v10_emails_main", 1024), ("results_v13_ablation_d_extra2048_only", 2048)],
    },
    "volume_tokens": {
        "x_label": "N_TOKENS_EXTRA_TRAIN",
        "runs": [
            ("results_v10_ablation_tok100k", 100_000),
            ("results_v10_ablation_tok2M", 2_000_000),
            ("results_v13_ablation_volume25m", 25_000_000),
        ],
    },
    "layer": {
        "x_label": "LAYER",
        "runs": [
            ("results_v13_ablation_layer12", 12),
            ("results_v10_emails_main", 24),
            ("results_v13_ablation_layer31", 31),
            ("results_v13_ablation_layer41", 41),
        ],
    },
    "hook_point": {
        "x_label": "HOOK_TYPE",
        "x_categorical": True,
        "runs": [
            ("results_v10_emails_main", "resid_post"),
            ("results_v13_ablation_mlp_out", "mlp_out"),
            ("results_v13_ablation_attn_out", "attn_out"),
        ],
    },
    "model_scale": {
        "x_label": "MODEL_SIZE",
        "x_categorical": True,
        "runs": [
            ("results_v13_ablation_model_scale_1b", "1b"),
            ("results_v13_ablation_model_scale_4b", "4b"),
            ("results_v10_emails_main", "12b"),
        ],
    },
}


def _load_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _interp_rate(run_dir):
    feats = _load_json(os.path.join(run_dir, "p1_top_extended_features.json"))
    if not feats:
        return None
    scores = [v["interp_score"] for v in feats.values()]
    return sum(scores) / len(scores)


def generate_training_curves(run_dir):
    """Pipeline 1 (checkpoint) + Pipeline 2 (*_history.json), si présents."""
    made = []
    plots_dir = os.path.join(run_dir, "plots")

    p1_ckpt = os.path.join(run_dir, "p1_extended_sae.pt")
    if os.path.exists(p1_ckpt):
        try:
            ckpt = torch.load(p1_ckpt, map_location="cpu", weights_only=False)
            history = ckpt.get("history") or {}
            if history.get("loss"):
                os.makedirs(plots_dir, exist_ok=True)
                out = os.path.join(plots_dir, "p1_training_curves.html")
                plot_training_curves(history, title=f"Pipeline 1 — {os.path.basename(run_dir)}", path=out)
                made.append(out)
        except Exception as e:
            print(f"  [P1] {run_dir} : échec lecture checkpoint ({e})")

    for p2_hist_path in glob.glob(os.path.join(run_dir, "p2_sae_dim*_history.json")):
        history = _load_json(p2_hist_path)
        if history and history.get("loss"):
            os.makedirs(plots_dir, exist_ok=True)
            base = os.path.basename(p2_hist_path).replace("_history.json", "")
            out = os.path.join(plots_dir, f"{base}_training_curves.html")
            plot_training_curves(history, title=f"Pipeline 2 ({base}) — {os.path.basename(run_dir)}", path=out)
            made.append(out)

    return made


def generate_rho_interp_distribution(run_dir):
    feats = _load_json(os.path.join(run_dir, "p1_top_extended_features.json"))
    if not feats:
        return None
    rho_values = [v["rho_interp"] for v in feats.values() if v.get("rho_interp") is not None]
    labels = ["interprétée" if v["interp_score"] else "non interprétée"
              for v in feats.values() if v.get("rho_interp") is not None]
    if not rho_values:
        return None
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out = os.path.join(plots_dir, "rho_interp_distribution.html")
    plot_activation_distribution(
        rho_values, labels=labels,
        title=f"Distribution de ρ_interp par interprétabilité — {os.path.basename(run_dir)}",
        path=out,
    )
    return out


def generate_sweep_plots():
    """Une figure par famille de balayage, sous results_diagnostics/plots/."""
    out_dir = os.path.join(REPO_ROOT, "results_diagnostics", "plots")
    os.makedirs(out_dir, exist_ok=True)
    made = []
    for family, spec in SWEEP_MANIFEST.items():
        rows = []
        for run_dir, x_val in spec["runs"]:
            full_dir = os.path.join(REPO_ROOT, run_dir)
            results = _load_json(os.path.join(full_dir, "results.json"))
            if not results:
                print(f"  [sweep:{family}] {run_dir} introuvable ou sans results.json, ignoré.")
                continue
            p1 = results.get("P1_Gemma3_SAE", {})
            interp = _interp_rate(full_dir)
            rows.append({
                spec["x_label"]: x_val,
                "rho_sae": p1.get("rho_sae"),
                "dead_pct": p1.get("dead_pct"),
                "fve_pretrained": p1.get("fve_pretrained"),
                "interp_rate": interp,
            })
        if len(rows) < 2:
            print(f"  [sweep:{family}] moins de 2 runs disponibles, ignoré.")
            continue
        df = pd.DataFrame(rows).sort_values(spec["x_label"])
        y_cols = [c for c in ["interp_rate", "rho_sae", "dead_pct", "fve_pretrained"]
                  if df[c].notna().any()]
        out = os.path.join(out_dir, f"sweep_{family}.html")
        plot_metric_vs_hyperparam(
            df, spec["x_label"], y_cols,
            title=f"Balayage {family} ({spec['x_label']})",
            x_is_categorical=spec.get("x_categorical", False),
            path=out,
        )
        made.append(out)
    return made


def main():
    result_dirs = sorted(glob.glob(os.path.join(REPO_ROOT, "results_*")))
    result_dirs = [d for d in result_dirs if os.path.isdir(d) and os.path.basename(d) != "results_diagnostics"]

    n_curves, n_rho = 0, 0
    for run_dir in result_dirs:
        made = generate_training_curves(run_dir)
        n_curves += len(made)
        if generate_rho_interp_distribution(run_dir):
            n_rho += 1

    sweep_made = generate_sweep_plots()

    print(f"\n{len(result_dirs)} runs scannés — {n_curves} courbes d'entraînement, "
          f"{n_rho} distributions rho_interp, {len(sweep_made)} figures de balayage "
          f"({os.path.relpath(os.path.join(REPO_ROOT, 'results_diagnostics', 'plots'), REPO_ROOT)}).")
    print("Non couvert par ce script (nécessite un rerun) : heatmap de corrélation "
          "NPMI (§24), histogramme de sensibilité à l'ordre du juge (§13.1) -- cf. "
          "docstring en tête de fichier.")


if __name__ == "__main__":
    main()
