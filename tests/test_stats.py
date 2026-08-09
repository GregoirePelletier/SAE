from src.analysis.stats import (
    cochran_armitage_trend_test,
    fdr_bh,
    minimum_detectable_effect,
    paired_mcnemar_test,
    proportion_with_ci,
    two_proportion_test,
)


def test_cochran_armitage_matches_results_tests_md_section_30_3():
    # RESULTS_TESTS.md §30.3 : 1b=18/150, 4b=42/150, 12b=68/150, scores linéaires
    # -> z=6,399, p=1,57e-10 (recalculé indépendamment ici, pas rejoué depuis le
    # même code -- sert de non-régression sur l'implémentation de ce module).
    res = cochran_armitage_trend_test([18, 42, 68], [150, 150, 150])
    assert abs(res.z - 6.399) < 0.01
    assert res.p < 1e-8


def test_paired_mcnemar_symmetric_discordant_gives_p_near_1():
    res = paired_mcnemar_test(b=10, c=10)
    assert res.p > 0.9
    assert res.n_discordant == 20


def test_paired_mcnemar_strongly_asymmetric_is_significant():
    res = paired_mcnemar_test(b=2, c=30)
    assert res.p < 0.01


def test_two_proportion_test_identical_rates_not_significant():
    res = two_proportion_test(45, 150, 45, 150)
    assert res.p > 0.9
    assert res.cohens_h == 0.0


def test_proportion_with_ci_contains_point_estimate():
    res = proportion_with_ci(68, 150)
    assert res.ci_low < res.rate < res.ci_high


def test_fdr_bh_never_more_significant_than_raw_pvalues():
    pvals = [0.001, 0.02, 0.03, 0.5, 0.8]
    q = fdr_bh(pvals)
    assert all(q[i] >= pvals[i] for i in range(len(pvals)))


def test_minimum_detectable_effect_shrinks_with_more_samples():
    small_n = minimum_detectable_effect(n_per_group=30, baseline_rate=0.45)
    large_n = minimum_detectable_effect(n_per_group=300, baseline_rate=0.45)
    assert large_n < small_n
