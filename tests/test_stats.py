import numpy as np

from src.analysis.stats import (
    bootstrap_ci_by_group,
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


def test_bootstrap_ci_by_group_contains_true_mean_for_iid_data():
    rng = np.random.default_rng(0)
    n_groups = 200
    values = rng.normal(loc=0.7, scale=0.1, size=n_groups)
    groups = np.arange(n_groups)  # un groupe par observation -- cas i.i.d. degenere
    res = bootstrap_ci_by_group(values, groups, n_boot=1000, seed=1)
    assert res.ci_low < 0.7 < res.ci_high
    assert res.n_groups == n_groups


def test_bootstrap_ci_by_group_deterministic_for_same_seed():
    rng = np.random.default_rng(0)
    values = rng.normal(size=50)
    groups = rng.integers(0, 10, size=50)
    a = bootstrap_ci_by_group(values, groups, seed=7)
    b = bootstrap_ci_by_group(values, groups, seed=7)
    assert a == b


def test_bootstrap_ci_by_group_respects_group_structure_not_just_n_observations():
    # Memes valeurs repetees 5x PAR GROUPE (5 variantes identiques par mail) :
    # un bootstrap qui rechantillonnerait les observations individuellement
    # (au lieu des groupes) sous-estimerait la variance -- l'IC doit rester
    # LARGE ici (peu de groupes reellement independants), pas se resserrer
    # artificiellement parce qu'il y a beaucoup de LIGNES.
    rng = np.random.default_rng(3)
    n_groups = 6
    group_means = rng.normal(loc=0.5, scale=0.3, size=n_groups)
    values = np.repeat(group_means, 5)  # 5 "variantes" identiques par groupe
    groups = np.repeat(np.arange(n_groups), 5)
    res_grouped = bootstrap_ci_by_group(values, groups, n_boot=2000, seed=5)

    # Repli (mauvaise pratique, pour comparaison) : bootstrap NAIF par ligne --
    # traite chaque groupe de 5 lignes identiques comme si "peu de variance" y regnait.
    res_naive = bootstrap_ci_by_group(values, np.arange(len(values)), n_boot=2000, seed=5)
    width_grouped = res_grouped.ci_high - res_grouped.ci_low
    width_naive = res_naive.ci_high - res_naive.ci_low
    assert width_grouped > width_naive


def test_bootstrap_ci_by_group_rejects_mismatched_lengths():
    import pytest
    with pytest.raises(ValueError):
        bootstrap_ci_by_group([1.0, 2.0, 3.0], ["a", "b"])
