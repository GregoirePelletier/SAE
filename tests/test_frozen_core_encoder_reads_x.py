"""Teste le correctif SAE Boost (Koriagin 2025, §3.1) de src/sae/frozen_core.py :
l'encodeur extra doit lire x (l'entrée du core gelé), pas le résidu
e = x - x̂_core (AUDIT_SAE_2026-08.md §1.3, "Écart 1"). Les tests existants de
test_frozen_core.py ne pouvaient pas distinguer les deux cas : leur mock
core_sae.decode() retourne toujours des zéros, donc residual == x et le bug
était invisible. Ici core_out != x pour rendre la distinction observable."""
from unittest.mock import MagicMock

import torch
from sae_lens import SAE

from src.sae.frozen_core import SAEBoostResidualSAE, FrozenCoreResidualSAE, FrozenDecoderExtendedSAE


def _mock_core_sae(batch, d_model, d_sae_core, core_out):
    mock_core_sae = MagicMock(spec=SAE)
    mock_cfg = MagicMock()
    mock_cfg.d_in = d_model
    mock_cfg.d_sae = d_sae_core
    mock_core_sae.cfg = mock_cfg
    mock_core_sae.encode.return_value = torch.zeros(batch, d_sae_core)
    mock_core_sae.decode.return_value = core_out
    return mock_core_sae


def test_forward_encoder_input_is_x_not_residual():
    batch, d_model, d_sae_core, d_extra = 4, 16, 8, 4
    x = torch.randn(batch, d_model)
    core_out = (0.5 * x).to(torch.bfloat16)  # != 0 et != x -> residual != x, distinguable
    mock_core_sae = _mock_core_sae(batch, d_model, d_sae_core, core_out)

    sae = FrozenCoreResidualSAE(core_sae=mock_core_sae, d_extra=d_extra, k_extra=2)

    calls = []
    orig_pre_extra = sae._pre_extra
    sae._pre_extra = lambda v: (calls.append(v.clone()), orig_pre_extra(v))[1]

    sae(x)

    assert len(calls) == 1
    x_bf16_f32 = x.to(torch.bfloat16).float()
    residual = x_bf16_f32 - core_out.float()
    assert torch.allclose(calls[0], x_bf16_f32)
    assert not torch.allclose(calls[0], residual)


def test_encode_reads_x_and_no_longer_calls_core_decode():
    """encode() n'a plus besoin de core_out une fois l'encodeur branché sur x
    -- core_sae.decode() ne doit plus être appelé du tout dans ce chemin
    (audit perf §1.3/§2.5 : un decode complet du core économisé à chaque
    encode(), et le besoin de decode_core_sparse au ré-encodage disparaît)."""
    batch, d_model, d_sae_core, d_extra = 4, 16, 8, 4
    x = torch.randn(batch, d_model)
    mock_core_sae = _mock_core_sae(batch, d_model, d_sae_core, torch.zeros(batch, d_model))

    sae = FrozenCoreResidualSAE(core_sae=mock_core_sae, d_extra=d_extra, k_extra=2)
    sae.encode(x)

    mock_core_sae.decode.assert_not_called()
    mock_core_sae.encode.assert_called_once()


def test_forward_loss_target_is_still_the_residual():
    """La cible de reconstruction (mse_loss) doit rester e = x - x̂_core,
    inchangée par ce correctif -- seule l'ENTRÉE de l'encodeur change."""
    batch, d_model, d_sae_core, d_extra = 4, 16, 8, 4
    x = torch.randn(batch, d_model)
    core_out = (0.5 * x).to(torch.bfloat16)
    mock_core_sae = _mock_core_sae(batch, d_model, d_sae_core, core_out)

    sae = FrozenCoreResidualSAE(core_sae=mock_core_sae, d_extra=d_extra, k_extra=2)
    out = sae(x)

    residual = x.to(torch.bfloat16).float() - core_out.float()
    extra_out = (out["extra_acts"] @ sae.W_dec_extra.float()) * sae.input_scale
    expected_mse = torch.nn.functional.mse_loss(extra_out, residual)
    expected_var = (residual - residual.mean(dim=0)).pow(2).mean()
    expected_nmse = expected_mse / (expected_var + 1e-8)
    assert torch.allclose(out["normalized_mse"], expected_nmse, atol=1e-5)


def test_backward_does_not_raise_dtype_error():
    """Non-régression du piège dtype documenté dans frozen_core.py (bf16 sans
    grad_fn mêlé à un tenseur fp32 avec grad_fn cassait le backward)."""
    batch, d_model, d_sae_core, d_extra = 8, 32, 16, 8
    x = torch.randn(batch, d_model)
    core_out = (0.3 * x).to(torch.bfloat16)
    mock_core_sae = _mock_core_sae(batch, d_model, d_sae_core, core_out)

    sae = FrozenCoreResidualSAE(core_sae=mock_core_sae, d_extra=d_extra, k_extra=2)
    out = sae(x)
    out["loss"].backward()  # ne doit pas lever "Found dtype BFloat16 but expected Float"
    assert sae.W_enc_extra.grad is not None


def test_direct_encode_extra_acts_call_matches_encode_method():
    """Non-régression du bug trouvé dans saev5.py (passe de ré-encodage,
    `ext_sae._encode_extra_acts(...)` appelée directement plutôt que via
    `encode()`) : ce site d'appel privé avait été oublié lors du correctif
    SAE Boost et continuait de passer le résidu e à l'encodeur alors que
    `encode()`/`forward()` avaient déjà été corrigés pour x -- divergence
    silencieuse (mêmes noms de fonctions, mauvais argument), invisible tant
    qu'on ne compare pas explicitement les deux chemins sur le même x.
    Garde-fou : quiconque appelle `_encode_extra_acts` directement doit lui
    passer x, comme `encode()` le fait en interne -- ce test échoue si les
    deux se remettent à diverger."""
    batch, d_model, d_sae_core, d_extra = 4, 16, 8, 4
    x = torch.randn(batch, d_model)
    core_out = (0.4 * x).to(torch.bfloat16)
    mock_core_sae = _mock_core_sae(batch, d_model, d_sae_core, core_out)

    sae = FrozenCoreResidualSAE(core_sae=mock_core_sae, d_extra=d_extra, k_extra=2)

    full_encoding = sae.encode(x)
    extra_from_encode = full_encoding[:, d_sae_core:]

    # Appel direct, tel que saev5.py le fait dans la passe de ré-encodage :
    # DOIT recevoir x, exactement comme encode() le fait en interne.
    extra_direct = sae._encode_extra_acts(x.to(torch.bfloat16).float())

    assert torch.equal(extra_from_encode, extra_direct)


def test_encoder_input_scale_calibrated_on_x_not_on_residual():
    """encoder_input_scale et input_scale doivent être calibrés sur deux
    distributions différentes (x vs e) -- ici volontairement séparées de
    plusieurs ordres de grandeur pour rendre un mélange des deux détectable."""
    batch, d_model, d_sae_core, d_extra = 64, 16, 8, 4
    mock_core_sae = _mock_core_sae(batch, d_model, d_sae_core, torch.zeros(batch, d_model))

    torch.manual_seed(0)
    domain_inputs = torch.randn(200, d_model) * 1000.0     # x : grande échelle (activations massives)
    domain_residuals = torch.randn(200, d_model) * 0.5     # e : petite échelle

    sae = SAEBoostResidualSAE(
        core_sae=mock_core_sae, d_extra=d_extra, k_extra=2,
        domain_residuals=domain_residuals, domain_inputs=domain_inputs,
    )

    assert sae.encoder_input_scale.item() > 100.0   # calibré sur x, pas sur e
    assert sae.input_scale.item() < 5.0              # calibré sur e, inchangé


def test_frozen_decoder_extended_sae_calibrates_scale_but_not_decoder():
    """FrozenDecoderExtendedSAE (sanity-check Korznikov et al.) : décodeur
    doit rester pseudo-aléatoire (pas de PCA), mais encoder_input_scale doit
    quand même être calibré sur x pour éviter des pré-activations explosées."""
    batch, d_model, d_sae_core, d_extra = 4, 16, 8, 4
    mock_core_sae = _mock_core_sae(batch, d_model, d_sae_core, torch.zeros(batch, d_model))

    domain_inputs = torch.randn(200, d_model) * 1000.0

    torch.manual_seed(42)
    sae_calibrated = FrozenDecoderExtendedSAE(
        core_sae=mock_core_sae, d_extra=d_extra, k_extra=2, domain_inputs=domain_inputs,
    )
    torch.manual_seed(42)
    sae_uncalibrated = FrozenDecoderExtendedSAE(
        core_sae=mock_core_sae, d_extra=d_extra, k_extra=2,
    )

    assert sae_calibrated.encoder_input_scale.item() > 100.0
    assert sae_uncalibrated.encoder_input_scale.item() == 1.0  # défaut, non calibré
    # Même graine, seul domain_inputs diffère : le décodeur (et l'encodeur, pas de
    # PCA dans cette classe) doivent être IDENTIQUES malgré domain_inputs -- seule
    # l'échelle change, jamais une direction data-informée.
    assert torch.equal(sae_calibrated.W_dec_extra, sae_uncalibrated.W_dec_extra)
    assert torch.equal(sae_calibrated.W_enc_extra, sae_uncalibrated.W_enc_extra)
    assert sae_calibrated.W_dec_extra.requires_grad is False
