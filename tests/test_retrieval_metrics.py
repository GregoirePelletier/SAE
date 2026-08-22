"""Métriques de retrieval interp-embed App. G (AUDIT_SAE_2026-08.md, §1)."""
import math

from src.analysis.metrics import (
    average_precision,
    precision_at_k,
    mean_average_precision,
    mean_precision_at_k,
    reciprocal_rank_fusion,
    rank_biased_overlap,
)


def test_average_precision_perfect_ranking():
    assert average_precision([True, True, True]) == 1.0


def test_average_precision_known_value():
    # rels aux rangs 1, 3, 4 sur 5 -- |R|=3
    # precision@1=1/1, @3=2/3, @4=3/4 -> AP = (1 + 2/3 + 3/4) / 3
    relevance = [True, False, True, True, False]
    expected = (1 / 1 + 2 / 3 + 3 / 4) / 3
    assert math.isclose(average_precision(relevance), expected, rel_tol=1e-9)


def test_average_precision_no_relevant():
    assert average_precision([False, False, False]) == 0.0


def test_precision_at_k():
    relevance = [True, False, True, True, False]
    assert precision_at_k(relevance, 1) == 1.0
    assert math.isclose(precision_at_k(relevance, 2), 0.5)
    assert math.isclose(precision_at_k(relevance, 4), 0.75)


def test_precision_at_k_zero():
    assert precision_at_k([True, False], 0) == 0.0


def test_mean_average_precision_averages_queries():
    q1 = [True, True]      # AP = (1/1 + 2/2) / 2 = 1.0
    q2 = [False, True]     # AP = (1/1) * (1/2) / 1 = 0.5 (le seul relevant est au rang 2)
    q3 = [False, False]    # AP = 0.0
    result = mean_average_precision([q1, q2, q3])
    assert math.isclose(result, (1.0 + 0.5 + 0.0) / 3, rel_tol=1e-9)


def test_mean_average_precision_empty():
    assert mean_average_precision([]) == 0.0


def test_mean_precision_at_k_averages_queries():
    q1 = [True, True, False]
    q2 = [False, True, True]
    result = mean_precision_at_k([q1, q2], k=2)
    assert math.isclose(result, (1.0 + 0.5) / 2, rel_tol=1e-9)


def test_reciprocal_rank_fusion_combines_two_rankings():
    ranking_a = ["x", "y", "z"]
    ranking_b = ["y", "x", "z"]
    fused = reciprocal_rank_fusion([ranking_a, ranking_b], k=60)
    scores = dict(fused)
    # x: rank 1 en a (1/61), rank 2 en b (1/62) ; y: rank 2 en a (1/62), rank 1 en b (1/61)
    # x et y ont un score identique par symétrie, z (rang 3 partout) strictement plus bas
    assert math.isclose(scores["x"], scores["y"], rel_tol=1e-9)
    assert scores["x"] > scores["z"]
    assert math.isclose(scores["z"], 2 / 63, rel_tol=1e-9)


def test_reciprocal_rank_fusion_sorted_descending():
    fused = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
    assert [doc for doc, _ in fused] == ["a", "b", "c"]


def test_reciprocal_rank_fusion_missing_doc_no_contribution():
    ranking_a = ["a", "b"]
    ranking_b = ["b"]  # "a" absent de ranking_b
    fused = dict(reciprocal_rank_fusion([ranking_a, ranking_b], k=60))
    assert math.isclose(fused["a"], 1 / 61, rel_tol=1e-9)
    assert math.isclose(fused["b"], 1 / 62 + 1 / 61, rel_tol=1e-9)


def test_rank_biased_overlap_identical_rankings_matches_closed_form():
    # Classements identiques : overlap(d) = d à chaque profondeur, donc
    # RBO = (1-p) * Sum_{d=1}^{depth} p^(d-1) = 1 - p^depth (somme géométrique).
    # RBO de base (non extrapolée) : converge vers 1 seulement quand depth -> infini,
    # ne vaut PAS 1 à profondeur finie même pour des classements identiques (propriété
    # connue de la formule, cf. docstring de rank_biased_overlap).
    ranking = ["a", "b", "c", "d"]
    p = 0.98
    depth = 4
    expected = 1 - p ** depth
    assert math.isclose(rank_biased_overlap(ranking, ranking, p=p), expected, rel_tol=1e-9)


def test_rank_biased_overlap_identical_beats_disjoint():
    identical_score = rank_biased_overlap(["a", "b", "c"], ["a", "b", "c"], p=0.98)
    disjoint_score = rank_biased_overlap(["a", "b", "c"], ["x", "y", "z"], p=0.98)
    assert identical_score > disjoint_score
    assert disjoint_score == 0.0


def test_rank_biased_overlap_disjoint_rankings_is_zero():
    a = ["a", "b", "c"]
    b = ["x", "y", "z"]
    assert rank_biased_overlap(a, b, p=0.98) == 0.0


def test_rank_biased_overlap_known_value_depth_2():
    # a=[x,y], b=[y,x], p=0.5 : d=1 overlap=0 -> terme 0 ; d=2 overlap=2 (les deux ensembles = {x,y}) -> terme p^1 * 2/2 = 0.5
    # RBO = (1-p) * (0 + 0.5) = 0.5 * 0.5 = 0.25
    result = rank_biased_overlap(["x", "y"], ["y", "x"], p=0.5)
    assert math.isclose(result, 0.25, rel_tol=1e-9)


def test_rank_biased_overlap_empty_rankings():
    assert rank_biased_overlap([], [], p=0.98) == 0.0
