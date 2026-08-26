from src.analysis.stats import (
    chance_corrected_rate,
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


def test_chance_corrected_rate_pure_guessing_gives_zero():
    # N6 : si le taux observé égale exactement le hasard moyen, le taux
    # corrigé doit être nul (aucune interprétabilité au-delà du hasard).
    n_items = [4, 4, 10, 10]           # hasard 25%, 25%, 10%, 10%
    chance = sum(1 / n for n in n_items) / len(n_items)
    scores = [chance] * 4              # taux observé = hasard moyen exactement
    res = chance_corrected_rate(scores, n_items)
    assert abs(res.corrected_rate) < 1e-9


def test_chance_corrected_rate_perfect_score_gives_one():
    res = chance_corrected_rate([1, 1, 1, 1], [4, 4, 10, 10])
    assert abs(res.corrected_rate - 1.0) < 1e-9


def test_chance_corrected_rate_lower_with_easier_items_at_same_raw_rate():
    # Même taux brut (60%), mais un lot a des items plus faciles (hasard plus
    # élevé, moins d'items) -- le taux corrigé doit être plus BAS pour ce lot,
    # une partie du score brut est imputable au hasard plus généreux.
    easy = chance_corrected_rate([1, 1, 1, 0, 0], [4] * 5)     # hasard 25%
    hard = chance_corrected_rate([1, 1, 1, 0, 0], [10] * 5)    # hasard 10%
    assert easy.obs_rate == hard.obs_rate == 0.6
    assert easy.corrected_rate < hard.corrected_rate
