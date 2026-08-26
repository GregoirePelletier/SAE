"""Teste src/analysis/reader_size_ablation.py -- F1 activation réelle vs
classification du juge (App I). Pur calcul numérique, CPU uniquement."""
import numpy as np

from src.analysis.reader_size_ablation import (
    f1_activation_vs_judge,
    compute_reader_size_ablation,
)


def test_f1_perfect_agreement_gives_one():
    acts = np.array([1, 1, 0, 0, 1])
    judge = np.array([1, 1, 0, 0, 1])
    assert f1_activation_vs_judge(acts, judge) == 1.0


def test_f1_no_overlap_gives_zero():
    acts = np.array([1, 1, 0, 0])
    judge = np.array([0, 0, 1, 1])
    assert f1_activation_vs_judge(acts, judge) == 0.0


def test_f1_both_all_zero_gives_zero_not_exception():
    acts = np.zeros(5, dtype=int)
    judge = np.zeros(5, dtype=int)
    assert f1_activation_vs_judge(acts, judge) == 0.0


def test_f1_partial_overlap_between_zero_and_one():
    acts = np.array([1, 1, 1, 0, 0])
    judge = np.array([1, 1, 0, 0, 0])
    f1 = f1_activation_vs_judge(acts, judge)
    assert 0.0 < f1 < 1.0


def test_compute_reader_size_ablation_uses_median_not_mean():
    # Un outlier bas (F1=0) ne doit pas dominer l'agrégat comme le ferait
    # une moyenne -- App I reporte des médianes (Figures 28-29).
    activation_by_feature = {
        0: np.array([1, 1, 1, 1]),
        1: np.array([1, 1, 1, 1]),
        2: np.array([0, 0, 0, 0]),   # F1=0 avec un judge tout à 1 ci-dessous
    }
    judge_by_feature = {
        0: np.array([1, 1, 1, 1]),   # F1=1.0
        1: np.array([1, 1, 0, 0]),   # F1<1.0 mais >0
        2: np.array([1, 1, 1, 1]),   # F1=0.0
    }
    result = compute_reader_size_ablation(activation_by_feature, judge_by_feature)
    assert result.n_features == 3
    assert result.per_feature_f1[0] == 1.0
    assert result.per_feature_f1[2] == 0.0
    assert result.median_f1 == np.median(list(result.per_feature_f1.values()))


def test_compute_reader_size_ablation_only_uses_common_features():
    activation_by_feature = {0: np.array([1, 0]), 1: np.array([1, 1])}
    judge_by_feature = {0: np.array([1, 0]), 2: np.array([1, 1])}   # feature 1/2 ne se recoupent pas
    result = compute_reader_size_ablation(activation_by_feature, judge_by_feature)
    assert result.n_features == 1
    assert set(result.per_feature_f1.keys()) == {0}


def test_compute_reader_size_ablation_empty_input():
    result = compute_reader_size_ablation({}, {})
    assert result.n_features == 0
    assert np.isnan(result.median_f1)
