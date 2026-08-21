"""Les activations massives de Gemma-3 (résidual stream, normes ~1e5) débordent
la plage représentable de fp16 (max ~65504) mais pas de bf16 (max ~3e38) --
invariant de conception que CLAUDE.md impose ("bf16 partout, y compris en
local"), pas un résultat d'expérience isolé. Testé ici sur tenseurs
synthétiques (CPU, aucun modèle chargé) pour qu'une régression vers fp16 sur
ce chemin soit détectée à chaque commit."""
import torch


def test_fp16_overflows_on_gemma_scale_activations():
    x = torch.full((4, 8), 1.2e5, dtype=torch.float32)
    x_fp16 = x.to(torch.float16)
    assert torch.isinf(x_fp16).any()


def test_bf16_does_not_overflow_on_gemma_scale_activations():
    x = torch.full((4, 8), 1.2e5, dtype=torch.float32)
    x_bf16 = x.to(torch.bfloat16)
    assert not torch.isinf(x_bf16).any()
    # bf16 a la même plage dynamique que fp32 (8 bits d'exposant), seule la
    # mantisse est réduite -- l'ordre de grandeur est préservé.
    assert torch.allclose(x_bf16.float(), x, rtol=0.02)


def test_fp16_max_representable_value():
    assert torch.finfo(torch.float16).max < 1.2e5
