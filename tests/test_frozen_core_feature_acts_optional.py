"""Teste src/sae/frozen_core.py::FrozenCoreResidualSAE.forward(return_feature_acts=...)
et son câblage dans src/sae/sae_shared.py::load_or_train_extended_sae (audit
perf §2.4 : feature_acts ([B, d_core+d_extra] fp32) alloué à chaque step sans
être jamais lu par ce harnais d'entraînement)."""
from unittest.mock import MagicMock

import torch
from sae_lens import SAE

from src.sae.frozen_core import FrozenCoreResidualSAE
from src.sae.sae_shared import load_or_train_extended_sae


def _mock_core_sae(batch, d_model, d_sae_core):
    mock_core_sae = MagicMock(spec=SAE)
    mock_cfg = MagicMock()
    mock_cfg.d_in = d_model
    mock_cfg.d_sae = d_sae_core
    mock_core_sae.cfg = mock_cfg
    mock_core_sae.encode.return_value = torch.zeros(batch, d_sae_core)
    mock_core_sae.decode.return_value = torch.zeros(batch, d_model)
    return mock_core_sae


def test_feature_acts_present_by_default():
    batch, d_model, d_sae_core, d_extra = 4, 16, 8, 4
    sae = FrozenCoreResidualSAE(core_sae=_mock_core_sae(batch, d_model, d_sae_core),
                                 d_extra=d_extra, k_extra=2)
    out = sae(torch.randn(batch, d_model))
    assert "feature_acts" in out
    assert out["feature_acts"].shape == (batch, d_sae_core + d_extra)


def test_feature_acts_omitted_when_disabled():
    batch, d_model, d_sae_core, d_extra = 4, 16, 8, 4
    sae = FrozenCoreResidualSAE(core_sae=_mock_core_sae(batch, d_model, d_sae_core),
                                 d_extra=d_extra, k_extra=2)
    out = sae(torch.randn(batch, d_model), return_feature_acts=False)
    assert "feature_acts" not in out
    # Les clés dont dépend la boucle d'entraînement doivent rester présentes.
    for key in ("loss", "l0_extra", "dead_frac", "aux_loss"):
        assert key in out


def test_training_loop_disables_feature_acts_for_frozen_core_models(tmp_path):
    """load_or_train_extended_sae doit appeler forward() avec
    return_feature_acts=False pour un modèle FrozenCoreResidualSAE (détecté
    via l'attribut core_sae), sans changer le résultat de l'entraînement."""
    batch_dummy = 8
    d_model, d_sae_core, d_extra = 16, 8, 4
    sae = FrozenCoreResidualSAE(core_sae=_mock_core_sae(batch_dummy, d_model, d_sae_core),
                                 d_extra=d_extra, k_extra=2)

    calls = []
    orig_forward = sae.forward

    def spy_forward(x, return_feature_acts=True):
        calls.append(return_feature_acts)
        return orig_forward(x, return_feature_acts=return_feature_acts)

    sae.forward = spy_forward

    acts_train = torch.randn(64, d_model)
    _, history = load_or_train_extended_sae(
        model=sae, model_name="stub_frozen_core", acts_train=acts_train,
        epochs=1, lr=1e-3, save_dir=str(tmp_path), device="cpu",
    )

    # epochs=1 -> exactement un appel de validation en fin d'époque (forward(vb),
    # sans return_feature_acts explicite -> défaut True) après les appels
    # d'entraînement (return_feature_acts=False) : le dernier appel capturé est
    # donc la validation, tous les précédents sont les steps d'entraînement.
    assert len(calls) == len(history["loss"]) + 1
    assert all(c is False for c in calls[:-1]), "feature_acts encore calculé dans la boucle d'entraînement"
    assert calls[-1] is True  # forward de validation, comportement par défaut inchangé
