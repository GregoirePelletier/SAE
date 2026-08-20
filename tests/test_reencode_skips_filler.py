"""Teste src/data/preparation.py::build_reencode_targets -- le ré-encodage
SAEBoostResidualSAE (saev5.py) doit ignorer le bloc filler entre train et
test/diff, puisqu'aucun consommateur en aval ne relit jamais cette tranche
(train_doc_acts/test_doc_acts/diff_doc_acts, la sélection de features, le
juge -- tous indexent explicitement autour du filler, jamais dedans)."""
import numpy as np

from src.data.preparation import build_reencode_targets


def test_filler_range_entirely_excluded():
    n_train, n_filler, n_test, n_diff = 5, 20, 3, 2
    n_total = n_train + n_filler + n_test + n_diff
    targets = build_reencode_targets(n_train, n_filler, n_total)

    filler_range = set(range(n_train, n_train + n_filler))
    assert not (filler_range & set(targets))


def test_train_and_test_diff_both_fully_included_in_order():
    n_train, n_filler, n_test, n_diff = 5, 20, 3, 2
    n_total = n_train + n_filler + n_test + n_diff
    targets = build_reencode_targets(n_train, n_filler, n_total)

    expected = list(range(n_train)) + list(range(n_train + n_filler, n_total))
    assert targets == expected
    assert len(targets) == n_train + n_test + n_diff


def test_no_filler_is_a_no_op():
    """N_VOLUME_FILLER_TARGET_CHUNKS=0 (défaut) : le ré-encodage doit couvrir
    exactement tout le corpus, comme avant ce correctif."""
    n_train, n_filler, n_total = 10, 0, 25
    targets = build_reencode_targets(n_train, n_filler, n_total)
    assert targets == list(range(n_total))


def test_reduces_work_proportionally_to_filler_size():
    """Cas du run de référence (540k filler sur 584k documents) : le
    ré-encodage ne doit porter que sur la fraction non-filler."""
    n_train, n_filler, n_test, n_diff = 41176, 540000, 2177, 900
    n_total = n_train + n_filler + n_test + n_diff
    targets = build_reencode_targets(n_train, n_filler, n_total)
    assert len(targets) == n_train + n_test + n_diff
    assert len(targets) / n_total < 0.1  # >90% de réduction du volume traité


def test_resume_position_maps_back_to_correct_doc_index():
    """Invariant dont dépend la reprise (saev5.py) : re_encode_targets[pos]
    doit retrouver le même indice de document, quelle que soit la position de
    reprise -- utilisé pour reconstruire all_doc_sae_acts depuis les fragments
    déjà traités."""
    n_train, n_filler, n_test, n_diff = 8, 15, 4, 1
    n_total = n_train + n_filler + n_test + n_diff
    targets = build_reencode_targets(n_train, n_filler, n_total)

    # Position n_train (juste après tout le train) doit tomber pile sur le
    # premier document de test (juste après le filler en indice de document).
    assert targets[n_train] == n_train + n_filler
    # Dernière position -> dernier document du corpus.
    assert targets[-1] == n_total - 1
    # Toutes les positions sont strictement croissantes (ordre préservé).
    assert targets == sorted(targets)
