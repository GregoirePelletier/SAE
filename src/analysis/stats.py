"""
src/analysis/stats.py — Module de tests statistiques partagé (Phase 1.3 de
l'audit méthodologique, plan approuvé 2026-08-07).

Contexte : `RESULTS_TESTS.md` §30 (audit de méthodologie statistique du
2026-07-29) a trouvé et corrigé rétroactivement 3 lacunes (McNemar sur
données appariées au lieu d'un test à 2 proportions indépendantes,
correction multi-test manquante malgré `fdr_bh` déjà disponible dans
`cooccurrence.py`, tendance dose-réponse testée par Cochran-Armitage plutôt
qu'une régression naïve) — mais chaque test a été recalculé ad-hoc, une
seule fois, sans module réutilisable. §30.4 liste explicitement 3 lacunes
non corrigées à cette passe : IC (Wilson) jamais rapportés systématiquement,
taille d'effet standardisée (h de Cohen) jamais calculée, analyse de
puissance jamais formalisée. Ce module consolide les tests déjà utilisés
UNE fois + comble ces 3 lacunes, pour que tout nouveau script d'audit
(Phase 1.2, 1.4, 2, 3) les utilise par défaut au lieu de réinventer un test
par script.

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
    ou taux proche de 0/1 — cf. lacune identifiée RESULTS_TESTS.md §30.4)."""
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
    (A réussit/B échoue ; A échoue/B réussit) — PAS les totaux. Utiliser
    l'exact binomial si b+c < 25 (recommandation statsmodels), sinon
    l'approximation chi². Convention adoptée dans RESULTS_TESTS.md §30.1-30.2."""
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
    modèle 1b/4b/12b, cf. RESULTS_TESTS.md §30.3). Pas d'implémentation directe
    dans statsmodels/scipy — formule standard (Agresti, *Categorical Data
    Analysis*, §3.4.2), vectorisée ici une seule fois plutôt que recopiée par
    script. `scores` : espacement des groupes (défaut = linéaire 0..k-1,
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
