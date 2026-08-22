"""Teste src/analysis/metrics.py::normalize_by_p90_and_score -- correctif
fidélité interp-embed Fig. 10 étape 1 (AUDIT_SAE_2026-08.md) :
property_based_retrieval (saev5.py) sommait les activations BRUTES pondérées
par rang, laissant les magnitudes JumpReLU non bornées du core (outliers
~1e5) écraser le poids de rang. Chaque latent est maintenant normalisé par
le 90e percentile de ses activations non nulles avant la somme pondérée."""
import torch

from src.analysis.metrics import normalize_by_p90_and_score as _normalize_by_p90_and_score


def test_outlier_magnitude_latent_no_longer_dominates():
    """2 documents, 2 latents matchés : latent 0 a une échelle ~1e5 (outlier
    JumpReLU) mais un rang de pertinence faible (poids petit) ; latent 1 a une
    échelle normale mais le rang le plus pertinent (poids le plus grand).
    Sans normalisation, le score serait dominé par latent 0 malgré son poids
    plus faible -- avec normalisation, le classement documentaire doit suivre
    le latent le plus PERTINENT (poids), pas le plus DENSE (magnitude)."""
    # doc A : fort sur le latent outlier (0), faible sur le latent pertinent (1)
    # doc B : l'inverse
    matched_acts = torch.tensor([
        [1.0e5, 1.0],
        [1.0, 100.0],
    ])
    # poids décroissant avec le rang : latent 1 (rang 0, poids fort) plus
    # pertinent que latent 0 (rang 1, poids faible).
    weights = torch.tensor([0.1, 0.9])

    scores = _normalize_by_p90_and_score(matched_acts, weights)
    # doc B (fort sur le latent pertinent une fois normalisé) doit dominer.
    assert scores[1] > scores[0]


def test_zero_activation_column_does_not_divide_by_zero():
    matched_acts = torch.tensor([[0.0, 5.0], [0.0, 3.0]])
    weights = torch.tensor([0.5, 0.5])
    scores = _normalize_by_p90_and_score(matched_acts, weights)
    assert torch.isfinite(scores).all()


def test_normalized_scale_is_comparable_across_latents():
    """Deux latents de magnitude très différente (1e5 vs 1.0) mais identiques
    une fois normalisés par leur propre p90 doivent contribuer également au
    score (à poids égal)."""
    matched_acts = torch.tensor([
        [1.0e5, 1.0],
        [5.0e4, 0.5],
    ])
    weights = torch.tensor([0.5, 0.5])
    scores = _normalize_by_p90_and_score(matched_acts, weights)
    # doc 0 est à p90 (ratio 1.0) sur les deux latents, doc 1 à ratio 0.5 sur
    # les deux -- le score de doc 0 doit être environ le double de doc 1.
    assert torch.isclose(scores[0], 2 * scores[1], rtol=0.05)
