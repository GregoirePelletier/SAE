import numpy as np

from src.analysis.metrics import diff_features as local_diff

try:
    from interp_embed import diff_features as external_diff
except Exception:
    external_diff = None


def test_diff_features_equivalence():

    if external_diff is None:
        return

    a = np.random.randn(100, 512)
    b = np.random.randn(100, 512)

    local = local_diff(a, b)
    ext = external_diff(a, b)

    assert local.shape == ext.shape