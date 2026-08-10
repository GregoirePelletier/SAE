"""Smoke-test PyTorch générique (retrieval cosinus) -- ne teste PAS
src/sae/retrieval/latent_terms.py (BM25 sur vocabulaire latent), malgré le nom
(AUDIT_REPO_2026-08-07.md §4.3)."""
import torch

def test_cosine_retrieval():

    docs = torch.randn(100, 512)

    query = docs[0]

    scores = torch.nn.functional.cosine_similarity(
        docs,
        query.unsqueeze(0),
        dim=1
    )

    idx = scores.argmax()

    assert idx == 0