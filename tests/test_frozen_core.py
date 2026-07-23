import torch
import pytest
from unittest.mock import MagicMock
from sae_lens import SAE
from src.sae.frozen_core import FrozenCoreResidualSAE, FrozenDecoderExtendedSAE

def test_output_shape():
    batch = 32
    d_model = 2304
    d_sae_core = 16384
    d_extra = 2048
    
    # 1. Création d'un mock robuste pour l'objet SAE de sae_lens
    mock_core_sae = MagicMock(spec=SAE)
    
    # Mock de la configuration interne requise par FrozenCoreResidualSAE
    mock_cfg = MagicMock()
    mock_cfg.d_in = d_model
    mock_cfg.d_sae = d_sae_core
    mock_core_sae.cfg = mock_cfg
    
    # Mock des comportements d'encodage/décodage
    mock_core_sae.encode.return_value = torch.zeros(batch, d_sae_core)
    mock_core_sae.decode.return_value = torch.zeros(batch, d_model)
    
    # 2. Instanciation avec la signature cible exacte de la surcouche EDF
    sae = FrozenCoreResidualSAE(
        core_sae=mock_core_sae,
        d_extra=d_extra,
        k_extra=32
    )
    
    # 3. Vérification des shapes de sortie du flux direct (forward)
    x = torch.randn(batch, d_model)
    out = sae(x)
    
    assert out["sae_out"].shape == (batch, d_model)
    assert out["feature_acts"].shape == (batch, d_sae_core + d_extra)
    assert out["extra_acts"].shape == (batch, d_extra)


def test_frozen_decoder_stays_frozen_during_training():
    """Sanity-check (Korznikov et al. 2026) : W_dec_extra ne doit JAMAIS bouger, même
    après plusieurs pas d'optimiseur, alors que W_enc_extra doit bouger normalement."""
    batch, d_model, d_sae_core, d_extra = 32, 256, 512, 128

    mock_core_sae = MagicMock(spec=SAE)
    mock_cfg = MagicMock()
    mock_cfg.d_in = d_model
    mock_cfg.d_sae = d_sae_core
    mock_core_sae.cfg = mock_cfg
    mock_core_sae.encode.return_value = torch.zeros(batch, d_sae_core)
    mock_core_sae.decode.return_value = torch.zeros(batch, d_model)

    sae = FrozenDecoderExtendedSAE(core_sae=mock_core_sae, d_extra=d_extra, k_extra=8)
    assert sae.W_dec_extra.requires_grad is False

    dec_before = sae.W_dec_extra.detach().clone()
    enc_before = sae.W_enc_extra.detach().clone()

    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-2)
    for _ in range(5):
        x = torch.randn(batch, d_model)
        optimizer.zero_grad()
        out = sae(x)
        out["loss"].backward()
        sae.normalize_decoder()
        optimizer.step()
        sae.normalize_decoder()

    assert torch.equal(sae.W_dec_extra, dec_before), "W_dec_extra a bougé alors qu'il doit rester figé"
    assert not torch.equal(sae.W_enc_extra, enc_before), "W_enc_extra n'a pas bougé — l'entraînement n'a rien fait"