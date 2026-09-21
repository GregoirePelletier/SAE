"""Tests CPU rapides de src/post_stage/stability.py (E05) -- donnees
synthetiques uniquement, aucun checkpoint/tenseur reel."""
import numpy as np

from src.post_stage.stability import (
    anisotropic_null_dictionary, group_partner, louvain_labels, matched_column_corr,
    mutual_knn_edges, nn_match, normalize_rows, resample_group_same_strata,
    subspace_basis, subspace_overlap, topk_jaccard,
)


def test_nn_match_recovers_permuted_dictionary_and_uses_signed_cosine():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((30, 16))
    perm = rng.permutation(30)
    idx, cos = nn_match(W, W[perm])
    assert (perm[idx] == np.arange(30)).all()
    assert np.allclose(cos, 1.0)
    # cosinus SIGNE : la direction opposee ne s'apparie PAS a 1
    _, cos_neg = nn_match(W[:1], -W[:1])
    assert cos_neg[0] < 0


def test_matched_column_corr_nan_on_constant_column():
    X = np.array([[1., 0.], [2., 0.], [3., 0.], [4., 0.]])
    c = matched_column_corr(X, X, np.array([0, 1]), np.array([0, 1]))
    assert np.isclose(c[0], 1.0) and np.isnan(c[1])


def test_topk_jaccard_identical_and_disjoint():
    s = np.arange(100, dtype=float)
    assert topk_jaccard(s, s, 20) == 1.0
    assert topk_jaccard(s, -s, 20) == 0.0


def test_anisotropic_null_is_unit_norm_and_respects_low_rank_geometry():
    rng = np.random.default_rng(1)
    # dictionnaire vivant dans un sous-espace de dim 3 d'un espace de dim 20
    basis = rng.standard_normal((3, 20))
    W = rng.standard_normal((50, 3)) @ basis
    null = anisotropic_null_dictionary(W, 40, rng)
    assert np.allclose(np.linalg.norm(null, axis=1), 1.0)
    Q, _ = np.linalg.qr(basis.T)
    resid = null - (null @ Q) @ Q.T
    # la moyenne des lignes est dans le sous-espace aussi (les lignes de W y vivent)
    assert np.linalg.norm(resid) < 1e-6


def test_louvain_recovers_two_planted_cliques_and_keeps_isolates():
    edges = [(i, j, 1.0) for i in range(4) for j in range(i + 1, 4)]
    edges += [(i, j, 1.0) for i in range(4, 8) for j in range(i + 1, 8)]
    labels = louvain_labels(9, edges, seed=0)
    assert len(set(labels[:4])) == 1 and len(set(labels[4:8])) == 1
    assert labels[0] != labels[4]
    assert (labels == labels[8]).sum() == 1  # noeud 8 : isolat


def test_mutual_knn_edges_are_mutual_and_ordered():
    rng = np.random.default_rng(2)
    W = rng.standard_normal((20, 8))
    for i, j, c in mutual_knn_edges(W, k=3):
        assert i < j and -1.0 <= c <= 1.0


def test_subspace_overlap_identical_is_one_and_orthogonal_is_zero_with_rank_reported():
    rng = np.random.default_rng(3)
    D = rng.standard_normal((6, 30))
    Q, _ = subspace_basis(D)
    ov, r = subspace_overlap(Q, Q)
    assert np.isclose(ov, 1.0) and r == Q.shape[1]
    Qa = np.eye(30)[:, :2]
    Qb = np.eye(30)[:, 2:4]
    ov0, r0 = subspace_overlap(Qa, Qb)
    assert ov0 == 0.0 and r0 == 2


def test_subspace_rank_cap_and_group_rank_bounded_by_group_size():
    rng = np.random.default_rng(4)
    Q, _ = subspace_basis(rng.standard_normal((40, 64)), r_cap=5)
    assert Q.shape[1] <= 5
    Q2, _ = subspace_basis(rng.standard_normal((2, 64)))
    assert Q2.shape[1] <= 2


def test_group_partner_purity_with_split():
    match_idx = np.array([0, 1, 2, 3, 4, 5])          # membre i -> feature i de B
    labels_b = np.array([7, 7, 7, 9, 9, 9])           # B : deux groupes
    partner, purity, n = group_partner(np.array([0, 1, 2, 3]), match_idx, labels_b)
    assert partner == 7 and n == 4 and np.isclose(purity, 0.75)


def test_resample_group_same_strata_preserves_strata_and_size():
    rng = np.random.default_rng(5)
    strata = np.array([0] * 10 + [1] * 10)
    pool = np.arange(20)
    members = np.array([0, 1, 12])
    out = resample_group_same_strata(members, strata, pool, rng)
    assert len(out) == 3 and len(set(out.tolist())) == 3
    assert strata[out].tolist() == strata[members].tolist()
