"""SAEBoostResidualSAE.decoder_init (bras temoin d'independance a l'init, E05) :
"pca" est deterministe sur le reservoir et donc INDEPENDANT de SEED (seul
l'ordre des mini-lots varie) ; "random" change les directions selon SEED et
laisse toute la calibration (input_scale, encoder_input_scale, biais) identique."""
from unittest.mock import MagicMock

import pytest
import torch
from sae_lens import SAE

from src.sae.frozen_core import SAEBoostResidualSAE

D_MODEL, D_CORE, D_EXTRA = 16, 8, 8


def _core():
    core = MagicMock(spec=SAE)
    cfg = MagicMock()
    cfg.d_in, cfg.d_sae = D_MODEL, D_CORE
    core.cfg = cfg
    return core


def _data():
    g = torch.Generator().manual_seed(123)
    residuals = torch.randn(64, D_MODEL, generator=g) * torch.linspace(3.0, 0.1, D_MODEL)
    inputs = torch.randn(64, D_MODEL, generator=g) + 0.5
    return residuals, inputs


def _build(seed, init):
    torch.manual_seed(seed)
    residuals, inputs = _data()
    return SAEBoostResidualSAE(_core(), d_extra=D_EXTRA, k_extra=2, domain_residuals=residuals,
                               domain_inputs=inputs, decoder_init=init)


def test_pca_init_is_independent_of_seed():
    a, b = _build(1, "pca"), _build(2, "pca")
    assert torch.allclose(a.W_dec_extra, b.W_dec_extra)
    assert torch.allclose(a.W_enc_extra, b.W_enc_extra)


def test_random_init_depends_on_seed_and_is_unit_norm():
    a, b = _build(1, "random"), _build(2, "random")
    assert not torch.allclose(a.W_dec_extra, b.W_dec_extra)
    assert torch.allclose(a.W_dec_extra.norm(dim=1), torch.ones(D_EXTRA), atol=1e-5)
    assert torch.allclose(a.W_enc_extra, a.W_dec_extra.T)


def test_random_init_differs_from_pca_but_calibration_is_identical():
    pca, rnd = _build(1, "pca"), _build(1, "random")
    assert not torch.allclose(pca.W_dec_extra, rnd.W_dec_extra)
    assert torch.allclose(pca.input_scale, rnd.input_scale)
    assert torch.allclose(pca.encoder_input_scale, rnd.encoder_input_scale)
    _, inputs = _data()
    mean_input = inputs.float().mean(dim=0)
    expected_bias = -(mean_input / rnd.encoder_input_scale) @ rnd.W_enc_extra.data
    assert torch.allclose(rnd.b_enc_extra.data, expected_bias, atol=1e-5)


def test_invalid_decoder_init_raises():
    with pytest.raises(ValueError):
        SAEBoostResidualSAE(_core(), d_extra=D_EXTRA, k_extra=2, decoder_init="kmeans")
