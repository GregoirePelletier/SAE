"""Teste le pooling documentaire réel de src/analysis/activations.py --
l'ancien contenu de ce fichier (acts.max(dim=0) sur du bruit, sans import
src) ne couvrait aucun code du dépôt malgré son nom (AUDIT_REPO_2026-08-07.md
§4.3)."""
import torch

from src.analysis.activations import scatter_maxpool, norm_outlier_mask


def test_scatter_maxpool_matches_per_doc_max():
    # 5 unités (tokens), 2 documents, d=3
    values = torch.tensor([
        [1.0, 0.0, 2.0],   # doc 0
        [3.0, 1.0, 0.0],   # doc 0
        [0.0, 5.0, 0.0],   # doc 1
        [2.0, 2.0, 2.0],   # doc 1
        [0.0, 0.0, 9.0],   # doc 1
    ])
    unit_to_doc = torch.tensor([0, 0, 1, 1, 1])
    pooled = scatter_maxpool(values, unit_to_doc, n_docs=2, d=3)
    expected = torch.tensor([
        [3.0, 1.0, 2.0],   # max sur les unités du doc 0
        [2.0, 5.0, 9.0],   # max sur les unités du doc 1
    ])
    assert torch.equal(pooled, expected)


def test_scatter_maxpool_doc_without_units_is_zero_not_inf():
    values = torch.tensor([[1.0, 2.0]])
    unit_to_doc = torch.tensor([0])
    pooled = scatter_maxpool(values, unit_to_doc, n_docs=3, d=2)  # docs 1 et 2 : aucune unité
    assert torch.equal(pooled[0], torch.tensor([1.0, 2.0]))
    assert torch.equal(pooled[1], torch.zeros(2))
    assert torch.equal(pooled[2], torch.zeros(2))
    assert not torch.isinf(pooled).any()


def test_norm_outlier_mask_excludes_extreme_norms():
    # batch=1, T=100 tokens : normes ~1.0, une extrême (activation massive Gemma-3).
    # Échantillon volontairement grand : mu/sd incluent l'outlier lui-même (pas de
    # statistique robuste), un seul point extrême parmi trop peu de tokens gonfle sd
    # au point de masquer son propre z-score (vérifié empiriquement en écrivant ce
    # test -- comportement réel de la fonction, pas un bug).
    resid = torch.ones(1, 100, 4)
    resid[0, 5] = 1000.0  # un token à norme massive
    mask = torch.ones(1, 100, dtype=torch.bool)
    out = norm_outlier_mask(resid, mask, sigma_clip=4.0)
    assert not out[0, 5]  # le token massif est exclu
    assert out[0, :5].all() and out[0, 6:].all()  # les autres restent


def test_norm_outlier_mask_noop_below_min_sample_size():
    resid = torch.randn(1, 5, 4)  # < 8 tokens valides : garde-fou "vals.numel() < 8"
    mask = torch.ones(1, 5, dtype=torch.bool)
    out = norm_outlier_mask(resid, mask)
    assert torch.equal(out, mask)
