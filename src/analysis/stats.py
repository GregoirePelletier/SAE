"""
src/analysis/stats.py — Module de tests statistiques partagé (McNemar sur
données appariées, correction multi-test Benjamini-Hochberg, tendance
dose-réponse Cochran-Armitage, IC de Wilson, taille d'effet standardisée --
h de Cohen, analyse de puissance) : point d'entrée unique pour tout script
d'ablation, plutôt que de réinventer un test par script.

Aucune réimplémentation de calcul statistique de bas niveau : tout délègue à
`statsmodels`/`scipy` (déjà dépendances du projet, `pyproject.toml`). Ce
module ne fait qu'exposer une API stable et documentée par-dessus.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from statsmodels.stats.contingency_tables import mcnemar as _mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    proportion_confint,
    proportion_effectsize,
    proportions_ztest,
)


def fdr_bh(pvalues: list[float] | np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Correction Benjamini-Hochberg (FDR). Ré-exporté depuis
    `src.analysis.cooccurrence` (implémentation identique, `multipletests`
    directement) pour que tous les scripts d'audit importent depuis un seul
    endroit plutôt que de dupliquer l'appel `multipletests(..., method="fdr_bh")`."""
    return multipletests(pvalues, alpha=alpha, method="fdr_bh")[1]


@dataclass
class ProportionResult:
    rate: float
    ci_low: float
    ci_high: float
    n: int
    method: str = "wilson"


def proportion_with_ci(n_success: int, n_total: int, alpha: float = 0.05) -> ProportionResult:
    """Intervalle de Wilson (recommandé sur l'IC normal classique pour n modeste
    ou taux proche de 0/1)."""
    lo, hi = proportion_confint(n_success, n_total, alpha=alpha, method="wilson")
    return ProportionResult(rate=n_success / n_total, ci_low=lo, ci_high=hi, n=n_total)


@dataclass
class TwoProportionResult:
    rate_a: float
    rate_b: float
    diff: float
    z: float
    p: float
    cohens_h: float


def two_proportion_test(n_success_a: int, n_a: int, n_success_b: int, n_b: int) -> TwoProportionResult:
    """Test à deux proportions INDÉPENDANTES + taille d'effet h de Cohen (lacune
    §30.4 : "jamais calculée"). N'utiliser QUE pour des échantillons
    indépendants — cf. `paired_mcnemar_test` si les deux mesures portent sur
    les MÊMES items (piège identifié §30 : c'est l'erreur faite avant
    correction sur `judge_robustness_check.py`/`multilingual_judge_bias_test.py`)."""
    count = np.array([n_success_a, n_success_b])
    nobs = np.array([n_a, n_b])
    z, p = proportions_ztest(count, nobs)
    h = proportion_effectsize(n_success_a / n_a, n_success_b / n_b)
    return TwoProportionResult(
        rate_a=n_success_a / n_a, rate_b=n_success_b / n_b,
        diff=n_success_a / n_a - n_success_b / n_b, z=float(z), p=float(p), cohens_h=float(h),
    )


@dataclass
class McNemarResult:
    statistic: float
    p: float
    n_discordant: int
    exact: bool


def paired_mcnemar_test(b: int, c: int) -> McNemarResult:
    """McNemar sur table 2x2 appariée. `b`/`c` = les deux cellules discordantes
    (A réussit/B échoue ; A échoue/B réussit) — PAS les totaux. Utilise
    l'exact binomial si b+c < 25 (recommandation statsmodels), sinon
    l'approximation chi²."""
    table = [[0, b], [c, 0]]
    exact = (b + c) < 25
    res = _mcnemar(table, exact=exact, correction=not exact)
    return McNemarResult(statistic=float(res.statistic), p=float(res.pvalue),
                          n_discordant=b + c, exact=exact)


@dataclass
class CochranArmitageResult:
    z: float
    p: float
    scores_used: list[float]


def cochran_armitage_trend_test(
    successes: list[int], totals: list[int], scores: list[float] | None = None,
) -> CochranArmitageResult:
    """Test de tendance de Cochran-Armitage (k groupes ordonnés, ex. tailles de
    modèle 1b/4b/12b). Pas d'implémentation directe dans statsmodels/scipy --
    formule standard (Agresti, *Categorical Data Analysis*, §3.4.2), vectorisée
    ici une seule fois plutôt que recopiée par script. `scores` : espacement
    des groupes (défaut = linéaire 0..k-1,
    §30.3 a vérifié la robustesse au choix linéaire vs log)."""
    successes = np.asarray(successes, dtype=float)
    totals = np.asarray(totals, dtype=float)
    if scores is None:
        scores = np.arange(len(successes), dtype=float)
    else:
        scores = np.asarray(scores, dtype=float)

    n = totals.sum()
    p_bar = successes.sum() / n
    s_bar = (totals * scores).sum() / n
    num = (successes * scores - totals * scores * p_bar).sum()
    var = p_bar * (1 - p_bar) * (totals * (scores - s_bar) ** 2).sum()
    z = num / np.sqrt(var)
    p = 2 * (1 - norm.cdf(abs(z)))
    return CochranArmitageResult(z=float(z), p=float(p), scores_used=scores.tolist())


def minimum_detectable_effect(n_per_group: int, baseline_rate: float, power: float = 0.8,
                               alpha: float = 0.05) -> float:
    """Analyse de puissance a priori (lacune §30.4 : "jamais formalisée") :
    plus petit écart de taux détectable à `power` avec `n_per_group` par
    bras, en partant de `baseline_rate`. Retourne l'écart en points de taux
    (pas en h de Cohen, plus lisible pour documenter un protocole avant de
    le lancer, ex. "n=150 par bras détecte un écart >= X points à 80% de
    puissance")."""
    analysis = NormalIndPower()
    # Recherche par dichotomie sur h (monotone en |diff|) plutôt qu'une formule
    # fermée : proportion_effectsize n'est pas trivialement inversible en diff
    # de taux à baseline_rate fixé.
    lo, hi = 1e-4, min(baseline_rate, 1 - baseline_rate) - 1e-4
    for _ in range(60):
        mid = (lo + hi) / 2
        h = abs(proportion_effectsize(baseline_rate + mid, baseline_rate))
        achieved_power = analysis.power(effect_size=h, nobs1=n_per_group, alpha=alpha, ratio=1.0)
        if achieved_power < power:
            lo = mid
        else:
            hi = mid
    return hi


@dataclass
class ChanceCorrectedResult:
    obs_rate: float
    chance_rate: float
    corrected_rate: float
    n: int


def chance_corrected_rate(scores: list[int] | np.ndarray,
                           n_items: list[int] | np.ndarray) -> ChanceCorrectedResult:
    """Taux d'interprétabilité corrigé du hasard variable (N6,
    AUDIT_SAE_2026-08.md §8) : le protocole odd-one-out d'`odd_one_out_judge`
    (src/sae/judge.py) ne fixe PAS le nombre d'items présentés au juge -- une
    feature rare peut n'atteindre que 3 positifs + 1 négatif (4 items, hasard
    25%), une feature dense les 9+1=10 prévus (hasard 10%), confondu avec le
    bin de fréquence (donc avec la stratification, N3). Formule de type Cohen
    kappa appliquée au niveau agrégé (pas par item, pour éviter qu'un
    micro-échantillon par feature ne rende le correctif par-item instable) :
    `(obs - c) / (1 - c)`, où `c = moyenne(1/n_items)` sur le MÊME ensemble de
    features que `obs = moyenne(scores)` -- le niveau de hasard attendu sous
    H0 ("chaque item jugé indépendamment au hasard uniforme"), pas un seuil
    fixe arbitraire.

    `scores` : 0/1 par feature (`interp_score`). `n_items` : nombre d'items
    RÉELLEMENT présentés à cette feature (`odd_one_out_judge(...)[f]["n_items"]`,
    absent des caches produits avant N6 -- reconstructible depuis
    `len(pos_examples) + 1` si `neg_example` est présent)."""
    scores = np.asarray(scores, dtype=np.float64)
    n_items = np.asarray(n_items, dtype=np.float64)
    obs_rate = float(scores.mean())
    chance_rate = float((1.0 / n_items).mean())
    denom = 1.0 - chance_rate
    corrected = (obs_rate - chance_rate) / denom if denom > 0 else float("nan")
    return ChanceCorrectedResult(obs_rate=obs_rate, chance_rate=chance_rate,
                                  corrected_rate=corrected, n=len(scores))


def horvitz_thompson_mean(values: list[float] | np.ndarray,
                           inclusion_probs: list[float] | np.ndarray) -> float:
    """Estimateur de Horvitz-Thompson de la moyenne populationnelle à partir
    d'un échantillon à probabilités d'inclusion inégales (N3,
    AUDIT_SAE_2026-08.md §8) : ŷ = (Σ y_i/π_i) / (Σ 1/π_i). Se réduit à la
    moyenne stratifiée classique (Σ_h N_h·ȳ_h / N) quand π_i est constant au
    sein de chaque strate -- exactement le cas de
    `feature_selection_stratified_by_frequency` (src/sae/judge.py), qui
    échantillonne un nombre à peu près fixe de features par bin de fréquence
    quelle que soit la taille du bin (π_i ≈ bin_n_sampled/bin_population).
    Corrige le biais qu'introduirait une moyenne brute non pondérée sur un tel
    échantillon : les bins rares (peu de features, mêmes qu'un bin dense dans
    l'échantillon) y seraient sur-représentés par rapport à la population
    complète de features vivantes."""
    values = np.asarray(values, dtype=np.float64)
    inclusion_probs = np.asarray(inclusion_probs, dtype=np.float64)
    weights = 1.0 / inclusion_probs
    return float(np.sum(values * weights) / np.sum(weights))
