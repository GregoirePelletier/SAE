"""Teste src/data/preparation.py::group_indices_by_doc -- remplace le
np.where(doc_ids == doc_idx) par document (O(n_docs * n_phrases)) utilisé dans
saev5.py:1512 (audit perf, item 10). Vérifie l'équivalence exacte avec
np.where et la borne de complexité O(n) (pas de dépendance quadratique au
nombre de documents)."""
import time

import numpy as np

from src.data.preparation import group_indices_by_doc


def test_matches_np_where_exactly():
    doc_ids = [0, 0, 1, 2, 1, 0, 3, 1]
    groups = group_indices_by_doc(doc_ids)
    doc_ids_arr = np.array(doc_ids)
    for doc_idx in range(4):
        expected = np.where(doc_ids_arr == doc_idx)[0].tolist()
        assert groups.get(doc_idx, []) == expected


def test_missing_doc_returns_empty_like_np_where():
    doc_ids = [0, 0, 2]
    groups = group_indices_by_doc(doc_ids)
    assert groups.get(1, []) == []


def test_accepts_numpy_array_input():
    doc_ids = np.array([2, 0, 0, 1])
    groups = group_indices_by_doc(doc_ids)
    assert groups[0] == [1, 2]
    assert groups[1] == [3]
    assert groups[2] == [0]


def test_linear_not_quadratic_in_n_docs():
    """n_docs très supérieur à n_phrases/doc : la version O(n_docs * n)
    (np.where en boucle) exploserait ici, group_indices_by_doc doit rester
    rapide (quelques dizaines de ms pour 200k documents/phrases)."""
    n = 200_000
    rng = np.random.default_rng(0)
    doc_ids = rng.integers(0, n, size=n)  # ~1 phrase/doc en moyenne
    start = time.perf_counter()
    groups = group_indices_by_doc(doc_ids.tolist())
    elapsed = time.perf_counter() - start
    assert len(groups) <= n
    assert elapsed < 5.0  # O(n) : large marge, juste pour attraper une régression O(n^2)
