"""
interp_embed est une dépendance volontairement non vendorisée (inspiration
méthodologique seulement, jamais installée dans le venv qui exécute tests/ --
seul `external/interp_embed/.venv` l'a). L'équivalent projet réellement
maintenu est `corpus_diff_stats` (Fisher exact + BH) dans
`src.analysis.cooccurrence` — ce test l'exerce sur des données synthétiques.
"""
import numpy as np
import pandas as pd
import torch

from src.analysis.cooccurrence import corpus_diff_stats, select_top_diff_features_by_frequency


def test_corpus_diff_stats_shape_and_types():
    rng = np.random.default_rng(0)
    n_docs, n_features = 200, 32
    doc_acts = torch.tensor(rng.random((n_docs, n_features)), dtype=torch.float32)
    # Rend quelques features clairement séparables entre groupes A/B pour éviter un
    # DataFrame vide (corpus_diff_stats ignore les features jamais actives).
    group_mask = np.zeros(n_docs, dtype=bool)
    group_mask[: n_docs // 2] = True
    doc_acts[group_mask, 0] += 1.0
    doc_acts[~group_mask, 1] += 1.0

    df = corpus_diff_stats(doc_acts, group_mask)

    assert set(["feature_id", "freq_A", "freq_B", "log_odds_ratio", "p", "q", "significant", "label"]) <= set(df.columns)
    assert len(df) > 0
    assert df["q"].between(0, 1).all()


def _make_diff_df(freq_diffs: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "feature_id": range(len(freq_diffs)),
        "freq_A": [0.5 + d / 2 for d in freq_diffs],
        "freq_B": [0.5 - d / 2 for d in freq_diffs],
        "log_odds_ratio": [0.0] * len(freq_diffs),
        "p": [1.0] * len(freq_diffs), "q": [1.0] * len(freq_diffs),
        "significant": [False] * len(freq_diffs), "label": [f"F{i}" for i in range(len(freq_diffs))],
    })


def test_select_top_diff_features_filters_below_threshold():
    # App D.2 : seuil 0.03 -- une différence de fréquence de 0.02 doit être exclue.
    df = _make_diff_df([0.02, 0.05, -0.10, 0.30])
    selected = select_top_diff_features_by_frequency(df, threshold=0.03, top_n=200)
    assert 0 not in selected["feature_id"].values
    assert set(selected["feature_id"].values) == {1, 2, 3}


def test_select_top_diff_features_sorted_by_absolute_diff_descending():
    df = _make_diff_df([0.05, -0.30, 0.10])
    selected = select_top_diff_features_by_frequency(df, threshold=0.03, top_n=200)
    assert list(selected["feature_id"].values) == [1, 2, 0]   # |−0.30| > |0.10| > |0.05|


def test_select_top_diff_features_respects_top_n():
    df = _make_diff_df([0.9, 0.8, 0.7, 0.6, 0.5])
    selected = select_top_diff_features_by_frequency(df, threshold=0.03, top_n=2)
    assert len(selected) == 2
    assert set(selected["feature_id"].values) == {0, 1}   # les deux plus grands écarts
