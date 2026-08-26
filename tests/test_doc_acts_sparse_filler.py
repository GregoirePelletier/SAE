"""Teste src/sae/sae_shared.py::{save,load}_doc_acts_sparse_filler et
load_all_doc_acts -- N10 (AUDIT_SAE_2026-08.md §8) : les lignes filler
d'all_doc_sae_acts (jamais lues en aval) ne sont plus stockées sur le cache
d'extraction partagé, seulement reconstruites (zéro) au chargement."""
import torch

from src.sae.sae_shared import (
    save_doc_acts_sparse_filler,
    load_doc_acts_sparse_filler,
    load_all_doc_acts,
)


def _make_tensor(n_train=5, n_filler=20, n_test=3, d=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    train = torch.rand(n_train, d, generator=g)
    filler = torch.rand(n_filler, d, generator=g)  # non-zéro exprès : la compaction ne dépend PAS de la valeur
    test = torch.rand(n_test, d, generator=g)
    full = torch.cat([train, filler, test], dim=0)
    return full, train, filler, test


def test_roundtrip_preserves_non_filler_rows_exactly(tmp_path):
    full, train, filler, test = _make_tensor()
    n_train, n_filler = train.shape[0], filler.shape[0]
    path = str(tmp_path / "doc_acts.pt")

    save_doc_acts_sparse_filler(full, n_train, n_filler, path)
    restored = load_doc_acts_sparse_filler(path)

    assert restored.shape == full.shape
    assert torch.equal(restored[:n_train], train)
    assert torch.equal(restored[n_train + n_filler:], test)


def test_filler_range_reconstructed_as_zero_regardless_of_original_content(tmp_path):
    # La plage filler est stockée non-nulle exprès (_make_tensor) pour
    # vérifier que la compaction l'ignore par INDEX, pas par détection de
    # valeur -- reconstruite à zéro même si l'original ne l'était pas
    # (cohérent avec "jamais lue en aval", la valeur exacte n'a jamais
    # d'importance).
    full, train, filler, test = _make_tensor()
    n_train, n_filler = train.shape[0], filler.shape[0]
    path = str(tmp_path / "doc_acts.pt")

    save_doc_acts_sparse_filler(full, n_train, n_filler, path)
    restored = load_doc_acts_sparse_filler(path)

    assert torch.equal(restored[n_train:n_train + n_filler], torch.zeros_like(filler))
    assert not torch.equal(filler, torch.zeros_like(filler))  # la garantie n'est pas triviale


def test_saved_file_is_much_smaller_than_dense_when_filler_dominates(tmp_path):
    import os
    full, train, filler, test = _make_tensor(n_train=5, n_filler=5000, n_test=3, d=64)
    n_train, n_filler = train.shape[0], filler.shape[0]
    compact_path = str(tmp_path / "compact.pt")
    dense_path = str(tmp_path / "dense.pt")

    save_doc_acts_sparse_filler(full, n_train, n_filler, compact_path)
    torch.save(full, dense_path)

    assert os.path.getsize(compact_path) < os.path.getsize(dense_path) / 10


def test_load_all_doc_acts_dispatches_on_sparse_filler_format(tmp_path):
    full, train, filler, test = _make_tensor()
    n_train, n_filler = train.shape[0], filler.shape[0]
    path = str(tmp_path / "doc_acts.pt")
    save_doc_acts_sparse_filler(full, n_train, n_filler, path)

    restored = load_all_doc_acts(path)
    assert torch.equal(restored[:n_train], train)
    assert torch.equal(restored[n_train + n_filler:], test)


def test_load_all_doc_acts_still_reads_plain_dense_tensor(tmp_path):
    # p1_all_doc_acts_ext_d*.pt (ré-encodage privé) reste un tenseur dense
    # classique, jamais compacté (N10 ne concerne que le cache PARTAGÉ) --
    # load_all_doc_acts doit rester compatible avec ce format existant.
    full, _, _, _ = _make_tensor()
    path = str(tmp_path / "plain.pt")
    torch.save(full, path)

    restored = load_all_doc_acts(path)
    assert torch.equal(restored, full)
