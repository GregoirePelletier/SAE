"""Teste le format CSR maison de src/storage/fragment_store.py -- l'ancien
contenu de ce fichier (torch.to_sparse_coo() générique, sans import src) ne
couvrait aucun code du dépôt malgré son nom (AUDIT_REPO_2026-08-07.md §4.3)."""
import numpy as np
import torch

from src.storage.fragment_store import save_fragment, load_fragment, feature_column, doc_maxpool


def _make_dense(seed=0):
    rng = np.random.default_rng(seed)
    dense = torch.zeros(6, 10)
    for t in range(6):
        n_active = rng.integers(0, 4)
        idx = rng.choice(10, size=n_active, replace=False)
        dense[t, idx] = torch.tensor(rng.random(n_active), dtype=torch.float32) + 0.1
    return dense  # ligne(s) potentiellement entièrement nulle -- cas limite rowptr


def test_csr_roundtrip_preserves_values(tmp_path):
    dense = _make_dense()
    save_fragment(str(tmp_path), 0, token_strings=[f"tok{i}" for i in range(6)], acts_dense=dense)
    frag = load_fragment(str(tmp_path), 0)
    assert frag["shape"] == (6, 10)

    for f_idx in range(10):
        col = feature_column(frag, f_idx)
        expected = dense[:, f_idx].numpy()
        # feature_column ne renvoie que ce qui dépasse eps (1e-6 dans _dense_to_csr)
        mask = expected > 1e-6
        assert np.allclose(col[mask], expected[mask], atol=1e-6)
        assert np.all(col[~mask] == 0)


def test_doc_maxpool_matches_dense_max(tmp_path):
    dense = _make_dense(seed=1)
    save_fragment(str(tmp_path), 1, token_strings=[f"tok{i}" for i in range(6)], acts_dense=dense)
    frag = load_fragment(str(tmp_path), 1)
    pooled = doc_maxpool(frag)
    expected = dense.max(dim=0).values
    assert torch.allclose(pooled, expected, atol=1e-6)


def test_all_zero_row_does_not_break_rowptr(tmp_path):
    dense = torch.zeros(4, 5)
    dense[1, 2] = 0.5  # une seule ligne non vide, entourée de lignes nulles
    save_fragment(str(tmp_path), 2, token_strings=[f"tok{i}" for i in range(4)], acts_dense=dense)
    frag = load_fragment(str(tmp_path), 2)
    col = feature_column(frag, 2)
    assert np.allclose(col, [0, 0.5, 0, 0], atol=1e-6)
