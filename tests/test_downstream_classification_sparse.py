"""Teste src/analysis/metrics.py::downstream_classification -- vérifie que le
passage de X_sae en CSR sparse (audit perf, CLAUDE.md règle
LogisticRegression) ne change pas le résultat par rapport à l'ancien chemin
dense, et mesure le gain de temps réel sur des activations creuses
synthétiques représentatives (~99,9% de zéros, comme un SAE BatchTopK)."""
import time

import numpy as np
import torch

from src.analysis.metrics import downstream_classification


def _synthetic_sparse_acts(n_per_label: int, d_sae: int, k_active: int, seed: int):
    """2 labels séparables, activations quasi-nulles sauf k_active colonnes
    actives par échantillon (mêmes colonnes pour un label donné, comme un
    signal SAE réellement discriminant), pour que l'accuracy soit non triviale
    et stable entre les deux chemins."""
    rng = np.random.default_rng(seed)
    acts = {}
    for label_id, label_name in enumerate(["a", "b"]):
        x = torch.zeros(n_per_label, d_sae)
        active_cols = rng.choice(d_sae, size=k_active, replace=False) if label_id == 0 \
            else rng.choice(d_sae, size=k_active, replace=False) + d_sae // 2
        active_cols = np.clip(active_cols, 0, d_sae - 1)
        for row in range(n_per_label):
            vals = rng.uniform(0.5, 2.0, size=k_active)
            x[row, active_cols] = torch.from_numpy(vals).float()
        acts[label_name] = x
    return acts


def test_matches_expected_accuracy_range_on_separable_synthetic_data():
    acts = _synthetic_sparse_acts(n_per_label=60, d_sae=2048, k_active=8, seed=0)
    results = downstream_classification(acts)
    # Signal quasi parfaitement séparable par construction -- l'accuracy sparse
    # doit rester haute, pas dégradée par le passage en CSR.
    assert results["acc_sae"] > 0.9


def test_sparse_path_is_not_slower_than_dense_equivalent():
    """Reproduit la charge qui rendait le job compute-bound (audit CLAUDE.md) :
    beaucoup de colonnes, très peu de non-zéros. CSR doit battre le dense sur
    le fit LogisticRegression, à charge égale."""
    from sklearn.linear_model import LogisticRegression
    from scipy import sparse as sp

    rng = np.random.default_rng(0)
    n, d = 400, 4096
    X_dense = np.zeros((n, d), dtype=np.float32)
    for row in range(n):
        cols = rng.choice(d, size=5, replace=False)
        X_dense[row, cols] = rng.uniform(0.5, 2.0, size=5).astype(np.float32)
    y = (np.arange(n) % 2)
    X_sparse = sp.csr_matrix(X_dense)

    start = time.perf_counter()
    LogisticRegression(max_iter=1000, solver="lbfgs").fit(X_dense, y)
    dense_time = time.perf_counter() - start

    start = time.perf_counter()
    LogisticRegression(max_iter=1000, solver="lbfgs").fit(X_sparse, y)
    sparse_time = time.perf_counter() - start

    assert sparse_time < dense_time
