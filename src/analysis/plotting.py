"""
src/analysis/plotting.py — Figures de diagnostic réutilisables (Plotly).

Convention : chaque fonction retourne un go.Figure ; si `path` est fourni, écrit
aussi un HTML autonome (fig.write_html(path)) et retourne le chemin -- même
contrat que src/analysis/visualization.py, pour rester appelable à la fois par
des scripts (artefact HTML sous SAVE_DIR/plots/) et par le dashboard Streamlit
(st.plotly_chart(fig) direct, pas de fichier intermédiaire). Point d'entrée
unique pour les graphiques de diagnostic d'entraînement et de balayage
d'hyperparamètres, consommé par scripts/generate_diagnostic_plots.py et par
src/visualization/dashboard.py (onglet "Diagnostics d'entraînement").
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _emit(fig: go.Figure, path: Optional[str]):
    if path:
        fig.write_html(path)
        return path
    return fig


def plot_training_curves(history: dict, title: str = "Courbes d'entraînement",
                          path: Optional[str] = None):
    """Loss/L0/dead_frac/aux_loss (+ val_loss si présent) vs step.

    Format d'entrée : le dict `history` produit par
    sae_shared.load_or_train_extended_sae (Pipeline 1) ou
    phrase_sae.load_or_train_sae (Pipeline 2) -- même schéma pour les deux
    (`step`/`loss`/`l0`/`dead_frac`/`aux_loss`, `val_epoch`/`val_loss`
    optionnels), donc une seule fonction couvre les deux pipelines.
    """
    metrics = [m for m in ["loss", "l0", "dead_frac", "aux_loss"] if history.get(m)]
    if not metrics:
        raise ValueError("historique vide ou format non reconnu (attend au moins 'loss').")
    has_val = bool(history.get("val_loss"))
    titles_map = {"loss": "Loss", "l0": "L0", "dead_frac": "Fraction de features mortes",
                  "aux_loss": "Aux loss (AuxK)"}
    colors = {"loss": "#1f77b4", "l0": "#2ca02c", "dead_frac": "#ff7f0e", "aux_loss": "#9467bd"}

    rows = len(metrics)
    fig = make_subplots(rows=rows, cols=1, subplot_titles=[titles_map[m] for m in metrics],
                         vertical_spacing=0.06)
    x = history.get("step", list(range(len(history[metrics[0]]))))
    for i, m in enumerate(metrics, start=1):
        fig.add_trace(go.Scatter(x=x, y=history[m], name=m, mode="lines",
                                  line=dict(color=colors[m])), row=i, col=1)
        if m == "loss" and has_val:
            fig.add_trace(go.Scatter(
                x=history.get("val_epoch", []), y=history["val_loss"],
                name="val_loss", mode="lines+markers", line=dict(color="#d62728")
            ), row=i, col=1)

    fig.update_layout(title=title, template="plotly_white", height=220 * rows, showlegend=True)
    fig.update_xaxes(title_text="step (points val : par époque)", row=rows, col=1)
    return _emit(fig, path)


def plot_metric_vs_hyperparam(df: pd.DataFrame, x_col: str, y_cols: Sequence[str],
                               title: Optional[str] = None, x_is_categorical: bool = False,
                               path: Optional[str] = None):
    """Une sous-figure par métrique de y_cols, vs la valeur d'hyperparamètre x_col.

    Générique : réutilisable pour n'importe quel balayage (K_EXTRA, D_EXTRA,
    volume de tokens, layer, hook-point...).
    """
    rows = len(y_cols)
    fig = make_subplots(rows=rows, cols=1, subplot_titles=list(y_cols), vertical_spacing=0.08)
    x = df[x_col].astype(str) if x_is_categorical else df[x_col]
    for i, y in enumerate(y_cols, start=1):
        fig.add_trace(go.Scatter(x=x, y=df[y], mode="lines+markers", name=y), row=i, col=1)
    fig.update_layout(title=title or f"Métriques vs {x_col}", template="plotly_white",
                       height=260 * rows, showlegend=False)
    fig.update_xaxes(title_text=x_col, row=rows, col=1)
    return _emit(fig, path)


def plot_activation_distribution(magnitudes, labels=None,
                                  title: str = "Distribution des magnitudes d'activation",
                                  path: Optional[str] = None):
    """Histogramme des magnitudes d'activation (max par doc/feature).

    `labels` optionnel (même longueur que `magnitudes`, ex. interprétable/non
    d'après le juge) : superpose un histogramme par groupe au lieu d'un
    histogramme global -- sert à visualiser si les features non interprétées
    correspondent à des activations plus faibles/rares.
    """
    magnitudes = np.asarray(magnitudes)
    fig = go.Figure()
    if labels is None:
        fig.add_trace(go.Histogram(x=magnitudes, nbinsx=50, marker_color="#1f77b4"))
    else:
        labels = pd.Series(labels).reset_index(drop=True)
        for val in sorted(labels.unique(), key=str):
            fig.add_trace(go.Histogram(
                x=magnitudes[labels.values == val], nbinsx=50, name=str(val), opacity=0.65))
        fig.update_layout(barmode="overlay")
    fig.update_layout(title=title, template="plotly_white",
                       xaxis_title="magnitude d'activation", yaxis_title="compte")
    return _emit(fig, path)


def plot_correlation_heatmap(corr_matrix, feature_labels=None,
                              title: str = "Corrélation NPMI entre features",
                              path: Optional[str] = None):
    """Heatmap d'une matrice de corrélation/co-activation carrée (N,N).

    Prévu pour être appelé restreint aux ~10 features les plus actives par
    intention (pas la matrice complète, illisible).
    """
    corr_matrix = np.asarray(corr_matrix)
    n = corr_matrix.shape[0]
    labels = feature_labels if feature_labels is not None else [str(i) for i in range(n)]
    fig = go.Figure(go.Heatmap(z=corr_matrix, x=labels, y=labels, colorscale="RdBu",
                                zmid=0, colorbar=dict(title="NPMI")))
    fig.update_layout(title=title, template="plotly_white",
                       height=max(400, 24 * n), width=max(450, 24 * n))
    return _emit(fig, path)


def plot_judge_agreement_histogram(agreement_rates,
                                    title: str = "Sensibilité du juge à l'ordre des exemples",
                                    path: Optional[str] = None):
    """Histogramme des taux d'accord (0-1) entre permutations de présentation
    des exemples au juge LLM, un point par feature testée.
    """
    agreement_rates = np.asarray(agreement_rates, dtype=float)
    fig = go.Figure(go.Histogram(x=agreement_rates, nbinsx=20, marker_color="#1f77b4"))
    fig.add_vline(x=float(agreement_rates.mean()), line_dash="dash", line_color="#d62728",
                  annotation_text=f"moyenne={agreement_rates.mean():.2f}")
    fig.update_layout(title=title, template="plotly_white",
                       xaxis_title="taux d'accord entre permutations", yaxis_title="nb features")
    return _emit(fig, path)
