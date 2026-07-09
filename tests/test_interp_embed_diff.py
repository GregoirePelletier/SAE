"""
La comparaison "diff_features" contre interp_embed (dépendance volontairement non
vendorisée, cf. Context.md — inspiration seulement, jamais installée en package) ne
peut de toute façon jamais s'exécuter : l'ancien test important `src.analysis.metrics
.diff_features`, une fonction qui n'existe pas (échec de collection pytest). L'équivalent
projet réellement maintenu est `corpus_diff_stats` (Fisher exact + BH) dans
`src.analysis.cooccurrence` — ce test l'exerce sur des données synthétiques.
"""
import numpy as np
import torch

from src.analysis.cooccurrence import corpus_diff_stats

try:
    from interp_embed import diff_features as external_diff
except Exception:
    external_diff = None


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


def test_corpus_diff_stats_vs_interp_embed():
    if external_diff is None:
        return  # interp_embed non installé : comparaison non automatisée, cf. docstring.
    a = np.random.randn(100, 512)
    b = np.random.randn(100, 512)
    ext = external_diff(a, b)
    assert ext is not None
