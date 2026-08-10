"""Smoke-test PyTorch générique (dtype bf16) -- ne teste aucun code du dépôt
(AUDIT_REPO_2026-08-07.md §4.3). La garantie bf16 réelle du projet (pas
d'overflow sur les activations massives Gemma-3) n'a pas de test dédié."""
import torch

def test_bfloat16_roundtrip():

    x = torch.randn(1024, 2304).to(torch.bfloat16)

    y = x.float().to(torch.bfloat16)

    assert y.dtype == torch.bfloat16

    err = (x.float() - y.float()).abs().mean()

    assert err < 1e-2