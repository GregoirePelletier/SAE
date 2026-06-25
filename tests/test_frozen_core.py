import torch
import pytest
from unittest.mock import MagicMock
from sae_lens import SAE
from src.sae.frozen_core import FrozenCoreResidualSAE

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