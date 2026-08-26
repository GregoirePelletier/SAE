"""Teste src/analysis/correlations_verified.py (App E.1/E.3/K.2) : filtre des
labels syntaxiques et vérification de présence (mock de _batched_generate,
CPU uniquement), NPMI_verified/CO sur des matrices de présence synthétiques
au résultat connu à la main, et le filtre de paire triviale (fragments
synthétiques via fragment_store, même pattern que test_sparse_storage.py)."""
from unittest.mock import patch

import numpy as np
import torch

from src.analysis.correlations_verified import (
    compute_verified_npmi,
    conditional_occurrence,
    filter_syntactic_labels,
    is_trivial_same_token_pair,
    verify_pair_presence,
)
from src.storage.fragment_store import save_fragment


def test_filter_syntactic_labels_parses_yes_no():
    labels = {1: "offensive language", 2: "conjunctions and prepositions", 3: "mentions religion"}

    def _fake(model, tokenizer, list_of_messages, max_new_tokens, batch_size=16):
        return ["1: YES\n2: NO\n3: YES"]

    with patch("src.analysis.correlations_verified._batched_generate", _fake):
        keep = filter_syntactic_labels(model=None, tokenizer=None, labels=labels)

    assert keep == {1: True, 2: False, 3: True}


def test_filter_syntactic_labels_unparsable_defaults_to_keep():
    labels = {1: "some label"}

    def _fake(model, tokenizer, list_of_messages, max_new_tokens, batch_size=16):
        return ["not the expected format at all"]

    with patch("src.analysis.correlations_verified._batched_generate", _fake):
        keep = filter_syntactic_labels(model=None, tokenizer=None, labels=labels)
    assert keep == {1: True}


def test_verify_pair_presence_parses_bracket_list():
    documents = ["doc about offense and religion", "doc about neither", "doc about religion only"]

    def _fake(model, tokenizer, list_of_messages, max_new_tokens, batch_size=16):
        return ["[1, 1]", "[0, 0]", "[0, 1]"]

    with patch("src.analysis.correlations_verified._batched_generate", _fake):
        presence = verify_pair_presence(model=None, tokenizer=None, label_i="offensive", label_j="religion",
                                          documents=documents)

    expected = np.array([[1, 0, 0], [1, 0, 1]])
    np.testing.assert_array_equal(presence, expected)


def test_verify_pair_presence_falls_back_to_loose_digit_parse():
    documents = ["doc"]

    def _fake(model, tokenizer, list_of_messages, max_new_tokens, batch_size=16):
        return ["feature 1: 1, feature 2: 0"]

    with patch("src.analysis.correlations_verified._batched_generate", _fake):
        presence = verify_pair_presence(model=None, tokenizer=None, label_i="a", label_j="b", documents=documents)
    np.testing.assert_array_equal(presence, np.array([[1], [0]]))


def test_compute_verified_npmi_perfect_cooccurrence_is_one():
    # i et j actifs exactement ensemble sur la moitié des documents -> NPMI=1 (cooc parfaite)
    presence = np.array([[1, 1, 0, 0], [1, 1, 0, 0]])
    npmi = compute_verified_npmi(presence)
    assert abs(npmi - 1.0) < 1e-5


def test_compute_verified_npmi_uncorrelated_pattern_is_exactly_zero():
    # i actif sur {0,1}, j actif sur {0,2} : cooc=1/4, p_i=p_j=1/2 -> pmi=log(1)=0
    # exactement (calculé à la main, pas juste "pas très corrélé").
    presence = np.array([[1, 1, 0, 0], [1, 0, 1, 0]])
    npmi = compute_verified_npmi(presence)
    assert abs(npmi) < 1e-5


def test_conditional_occurrence_directional():
    # j actif partout où i est actif (4 docs), mais i actif seulement la moitié du temps où j l'est
    # -> P(i|j) < P(j|i)=1 -> CO=1.0
    i = [1, 1, 0, 0]
    j = [1, 1, 1, 1]
    co = conditional_occurrence(np.array([i, j]))
    assert abs(co - 1.0) < 1e-9


def test_conditional_occurrence_zero_when_never_cooccur():
    presence = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    assert conditional_occurrence(presence) == 0.0


def _write_doc(tmp_path, doc_id, n_tokens, active_i, active_j, f_i=0, f_j=1, d=4):
    dense = torch.zeros(n_tokens, d)
    for pos in active_i:
        dense[pos, f_i] = 0.5
    for pos in active_j:
        dense[pos, f_j] = 0.5
    save_fragment(str(tmp_path), doc_id, token_strings=[f"t{k}" for k in range(n_tokens)], acts_dense=dense)


def test_trivial_same_token_pair_detected_when_positions_coincide(tmp_path):
    # 3 docs, i et j piquent systématiquement au même token (ou adjacent) -> trivial=True
    _write_doc(tmp_path, 0, 6, active_i=[2], active_j=[2])
    _write_doc(tmp_path, 1, 6, active_i=[4], active_j=[5])
    _write_doc(tmp_path, 2, 6, active_i=[1], active_j=[1])
    assert is_trivial_same_token_pair(str(tmp_path), [0, 1, 2], f_i=0, f_j=1) is True


def test_non_trivial_pair_with_scattered_positions_not_flagged(tmp_path):
    # positions systématiquement éloignées -> pas un artefact de même-token
    _write_doc(tmp_path, 0, 10, active_i=[1], active_j=[8])
    _write_doc(tmp_path, 1, 10, active_i=[0], active_j=[9])
    _write_doc(tmp_path, 2, 10, active_i=[2], active_j=[7])
    assert is_trivial_same_token_pair(str(tmp_path), [0, 1, 2], f_i=0, f_j=1) is False


def test_trivial_pair_skips_documents_where_a_feature_never_fires(tmp_path):
    # doc 1 : j ne s'active jamais -> ignoré du calcul, pas compté comme "non proche"
    _write_doc(tmp_path, 0, 6, active_i=[2], active_j=[2])
    _write_doc(tmp_path, 1, 6, active_i=[3], active_j=[])
    assert is_trivial_same_token_pair(str(tmp_path), [0, 1], f_i=0, f_j=1) is True
