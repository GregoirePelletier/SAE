"""Valide le MÉCANISME de hook utilisé pour attn_out/mlp_out (saev5.py, layer/
hook-point sweep, RESULTS_TESTS.md §36) sur un module PyTorch jouet -- pas le
modèle Gemma-3 réel (indisponible sans GPU), mais l'API register_forward_hook/
register_forward_pre_hook elle-même, la partie la plus susceptible de bug
silencieux (signature, ordre des arguments)."""
import torch
import torch.nn as nn


def test_forward_pre_hook_captures_input_before_projection():
    linear = nn.Linear(8, 8)
    capture = {}

    def _capture_in(module, args, kwargs):
        capture["acts"] = args[0] if args else kwargs["input"]

    linear.register_forward_pre_hook(_capture_in, with_kwargs=True)
    x = torch.randn(2, 8)
    linear(x)
    assert torch.equal(capture["acts"], x)


def test_forward_hook_captures_output_after_module():
    ln = nn.LayerNorm(8)
    capture = {}

    def _capture_out(module, args, output):
        capture["acts"] = output

    ln.register_forward_hook(_capture_out)
    x = torch.randn(2, 8)
    out = ln(x)
    assert torch.equal(capture["acts"], out)
    assert not torch.equal(capture["acts"], x)  # layernorm a bien transformé x


def test_hook_removal_stops_capture():
    linear = nn.Linear(4, 4)
    capture = {}
    handle = linear.register_forward_hook(lambda m, a, o: capture.setdefault("n", 0) or capture.update(n=capture["n"] + 1))
    linear(torch.randn(1, 4))
    handle.remove()
    linear(torch.randn(1, 4))
    assert capture["n"] == 1
