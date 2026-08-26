"""Teste src/analysis/metrics.py::dead_pct_core_extension -- avant ce
correctif, `dead_pct` (saev5.py) mélangeait CORE (GemmaScope figé,
généraliste) et EXTENSION (entraînée) en un seul chiffre, sur-estimant
structurellement le taux de mort de l'extension (déjà noté une fois à la
main, RESULTS_TESTS.md §17.4, jamais calculé systématiquement par le
pipeline depuis)."""
import torch

from src.analysis.metrics import dead_pct_core_extension


def test_splits_core_and_extension_independently():
    n_docs, d_core, d_extra = 10, 4, 2
    doc_acts = torch.zeros(n_docs, d_core + d_extra)
    # Core : features 0,1 vivantes, 2,3 mortes (50% mort).
    doc_acts[:, 0] = 1.0
    doc_acts[:, 1] = 1.0
    # Extension : feature 4 vivante, 5 morte (50% mort aussi, mais compté
    # séparément -- pas le même dénominateur que le core).
    doc_acts[:, 4] = 1.0

    dead_core, dead_ext = dead_pct_core_extension(doc_acts, d_core)
    assert dead_core == 50.0
    assert dead_ext == 50.0


def test_all_core_dead_extension_alive():
    n_docs, d_core, d_extra = 5, 3, 2
    doc_acts = torch.zeros(n_docs, d_core + d_extra)
    doc_acts[:, d_core:] = 1.0  # extension entièrement vivante

    dead_core, dead_ext = dead_pct_core_extension(doc_acts, d_core)
    assert dead_core == 100.0
    assert dead_ext == 0.0


def test_no_frozen_core_returns_global_and_nan():
    """d_core >= largeur totale (pas d'extension, ex. Latent Terms
    token-level) -- la distinction n'a pas de sens, retourne le taux
    global côté core et NaN côté extension plutôt qu'un 0%/100% trompeur."""
    n_docs, d_sae = 8, 4
    doc_acts = torch.zeros(n_docs, d_sae)
    doc_acts[:, 0] = 1.0  # 1/4 vivante -> 75% mort

    dead_core, dead_ext = dead_pct_core_extension(doc_acts, d_core=d_sae)
    assert dead_core == 75.0
    assert dead_ext != dead_ext  # NaN != NaN


def test_matches_blended_dead_pct_when_recombined():
    """Sanity check : le nombre de features mortes core + extension doit
    correspondre exactement au dead_pct global (même calcul, juste
    décomposé par plage)."""
    torch.manual_seed(0)
    n_docs, d_core, d_extra = 20, 50, 20
    doc_acts = (torch.rand(n_docs, d_core + d_extra) > 0.9).float()

    dead_core, dead_ext = dead_pct_core_extension(doc_acts, d_core)
    n_dead_core = round(dead_core / 100 * d_core)
    n_dead_ext = round(dead_ext / 100 * d_extra)
    n_dead_global = int((doc_acts.sum(dim=0) == 0).sum().item())
    assert n_dead_core + n_dead_ext == n_dead_global
