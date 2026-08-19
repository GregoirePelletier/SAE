"""Tests unitaires rapides, CPU-only, de src/sae/retrieval/latent_terms.py
(shape/dtype/non-régression -- pas de F2LLM/GPU ici, cf. convention `tests/`
de CLAUDE.md). Le nom du fichier ne correspondait à rien avant cette
version : il contenait un smoke-test cosinus générique sans rapport avec
latent_terms.py (documenté comme tel dans son propre docstring)."""
import numpy as np
import torch

from src.sae.retrieval.latent_terms import LatentTermsSAE, LatentTermsIndex


def test_latent_terms_sae_topk_exact_train_and_eval():
    torch.manual_seed(0)
    sae = LatentTermsSAE(d_in=16, d_sae=64, k=4)
    x = torch.randn(8, 16)

    sae.train()
    out_train = sae(x)
    l0_train = (out_train["feature_acts"] > 1e-6).sum(dim=-1)
    assert torch.all(l0_train <= 4)  # top-k per-échantillon, jamais plus de k actifs

    sae.eval()
    out_eval = sae(x)
    l0_eval = (out_eval["feature_acts"] > 1e-6).sum(dim=-1)
    assert torch.all(l0_eval <= 4)
    assert out_eval["sae_out"].shape == x.shape
    assert torch.isfinite(out_eval["sae_out"]).all()


def test_latent_terms_sae_decoder_unit_norm_after_step():
    torch.manual_seed(0)
    sae = LatentTermsSAE(d_in=8, d_sae=32, k=2)
    x = torch.randn(16, 8)
    out = sae(x)
    out["loss"].backward()
    sae.normalize_decoder()
    norms = sae.W_dec.data.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_latent_terms_index_ranks_exact_overlap_first():
    # 3 documents, vocabulaire latent à 4 dimensions ; doc 0 partage tous les
    # termes de la requête, doc 1 en partage un seul, doc 2 aucun.
    W_docs = np.array([
        [2.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 1.0],
    ])
    from scipy import sparse
    index = LatentTermsIndex(sparse.csr_matrix(W_docs))
    w_q = np.array([1.0, 1.0, 0.0, 0.0])
    ranked = [doc_id for doc_id, _ in index.search(w_q, top_k=3)]
    assert ranked[0] == 0
    assert 2 not in ranked  # score nul (aucune intersection) -> exclu, cf. search()
