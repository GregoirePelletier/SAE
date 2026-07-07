"""
visualization.py — Sorties Plotly HTML autonomes (pas de serveur requis).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go


def plot_feature_space_clusters(emb2d, labels, texts, path="clusters_sae.html"):
    hover = [t[:120].replace("\n", " ") for t in texts]
    fig = go.Figure(go.Scattergl(
        x=emb2d[:, 0], y=emb2d[:, 1], mode="markers",
        marker=dict(color=labels, colorscale="Turbo", size=5),
        text=hover, hoverinfo="text"))
    fig.update_layout(title="Clustering en espace de features SAE (TF-IDF + HDBSCAN)",
                      template="plotly_white")
    fig.write_html(path)
    return path


def plot_corpus_diff(df_diff: pd.DataFrame, top_n=30, path="corpus_diff.html"):
    d = df_diff[df_diff["significant"]].head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d["log_odds_ratio"], y=d["label"], orientation="h",
        marker_color=np.where(d["log_odds_ratio"] > 0, "#d62728", "#1f77b4"),
        customdata=np.stack([d["freq_A"], d["freq_B"], d["q"]], axis=1),
        hovertemplate="LOR=%{x:.2f}<br>freq_A=%{customdata[0]:.3f} "
                      "freq_B=%{customdata[1]:.3f}<br>q=%{customdata[2]:.1e}"))
    fig.update_layout(title="Diff de corpus — log-odds-ratio (Fisher + BH, q<0.05)",
                      template="plotly_white", height=max(400, 22 * len(d)))
    fig.write_html(path)
    return path


def plot_cooccurrence_graph(G: nx.Graph, path="cooc_graph.html"):
    pos = nx.spring_layout(G, weight="npmi", seed=0)
    ex, ey = [], []
    for u, v in G.edges:
        ex += [pos[u][0], pos[v][0], None]
        ey += [pos[u][1], pos[v][1], None]
    nodes = list(G.nodes)
    fig = go.Figure([
        go.Scatter(x=ex, y=ey, mode="lines",
                   line=dict(width=0.5, color="#bbb"), hoverinfo="none"),
        go.Scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
            mode="markers",
            marker=dict(size=[6 + 40 * G.nodes[n]["freq"] for n in nodes],
                        color=[G.nodes[n].get("community", 0) for n in nodes],
                        colorscale="Turbo"),
            text=[f'{G.nodes[n]["label"]}<br>freq={G.nodes[n]["freq"]:.3f}'
                  f'<br>com={G.nodes[n].get("community")}' for n in nodes],
            hoverinfo="text"),
    ])
    fig.update_layout(title="Graphe de co-occurrence NPMI (communautés Louvain)",
                      template="plotly_white", showlegend=False,
                      xaxis_visible=False, yaxis_visible=False)
    fig.write_html(path)
    return path


def plot_pollution_report(report, path="pollution.html"):
    d = report.per_feature.head(50)
    fig = go.Figure(go.Scatter(
        x=d["orphan"], y=1 - d["ami_alignment"], mode="markers",
        marker=dict(size=8 + 6 * d["pollution_score"].clip(lower=0),
                    color=d["pollution_score"], colorscale="Reds", showscale=True),
        text=[f'F{f}<br>score={s:.2f}<br>com={c}'
              for f, s, c in zip(d["feature_id"], d["pollution_score"], d["community"])],
        hoverinfo="text"))
    fig.update_layout(title=f"Features suspectes (n_flagged={report.n_flagged}, "
                            f"model_score={report.model_score:.3f})",
                      xaxis_title="orphan (1 - corr appariée)",
                      yaxis_title="1 - AMI(communauté, labels)",
                      template="plotly_white")
    fig.write_html(path)
    return path