"""Teste src/sae/phrase_sae.py::compute_sae_metrics -- l'entrée ne doit plus
être castée en bf16 avant un PhraseLevelSAE fp32 (audit perf §2.7 : les
métriques publiées étaient calculées sur une entrée dégradée par rapport à
l'entraînement, qui lui reste fp32)."""
import torch

from src.sae.phrase_sae import PhraseLevelSAE, compute_sae_metrics


def test_forward_input_stays_float32():
    d_in, d_sae, k = 8, 16, 2
    sae = PhraseLevelSAE(d_in, d_sae, k)

    calls = []
    orig_forward = sae.forward

    def spy_forward(x):
        calls.append(x.dtype)
        return orig_forward(x)

    sae.forward = spy_forward

    embeddings = torch.randn(20, d_in)  # fp32, comme le retour de extract_f2llm_embeddings
    compute_sae_metrics(sae, embeddings, batch_size=8)

    assert len(calls) > 0
    assert all(dt == torch.float32 for dt in calls)


def test_metrics_match_direct_fp32_forward():
    """Non-régression numérique : NMSE/L0 doivent correspondre exactement à un
    forward fp32 direct, pas à une version dégradée bf16->fp32."""
    torch.manual_seed(0)
    d_in, d_sae, k = 8, 16, 2
    sae = PhraseLevelSAE(d_in, d_sae, k)
    sae.eval()
    embeddings = torch.randn(8, d_in)

    with torch.no_grad():
        expected = sae(embeddings)

    metrics = compute_sae_metrics(sae, embeddings, batch_size=8)
    assert abs(metrics["NMSE"] - expected["normalized_mse"].item()) < 1e-6
    assert abs(metrics["L0"] - expected["l0"].item()) < 1e-6
