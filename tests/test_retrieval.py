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


def test_latent_terms_index_stores_csc_not_csr():
    """search() n'accède qu'à des colonnes (audit perf §1.4, item 5) -- l'index
    doit convertir en CSC à la construction, pas garder la CSR d'entrée."""
    from scipy import sparse
    W_docs = sparse.csr_matrix(np.eye(4))
    index = LatentTermsIndex(W_docs)
    assert isinstance(index.W, sparse.csc_matrix)


def test_latent_terms_index_csc_matches_csr_getcol_scores():
    """Non-régression : les scores de search() sur l'index CSC doivent être
    identiques à ceux qu'on obtiendrait en accédant aux colonnes de la CSR
    d'origine (même calcul, ordre d'accès mémoire différent seulement)."""
    from scipy import sparse
    rng = np.random.default_rng(0)
    W_docs = sparse.random(200, 64, density=0.05, random_state=rng, format="csr")
    W_docs.data = np.abs(W_docs.data)
    index = LatentTermsIndex(W_docs)
    w_q = np.zeros(64)
    w_q[[3, 10, 40]] = 1.0

    # Référence : même formule BM25, colonnes lues directement sur la CSR d'origine.
    N = W_docs.shape[0]
    expected_scores = np.zeros(N)
    for j in np.nonzero(w_q > 0)[0]:
        col = W_docs.getcol(j)
        d_idx, f = col.indices, col.data.astype(np.float64)
        contrib = index.idf[j] * f * (index.k1 + 1) / (f + index.k1 * index.K[d_idx])
        expected_scores[d_idx] += float(w_q[j]) * contrib

    got = index.search(w_q, top_k=N)
    got_scores = np.zeros(N)
    for doc_id, score in got:
        got_scores[doc_id] = score
    np.testing.assert_allclose(got_scores, expected_scores, rtol=1e-9)


def test_latent_terms_index_csc_column_access_is_faster_than_csr():
    """Reproduit le goulot mesuré : accès colonne répété sur une grande matrice
    -- CSC doit être nettement plus rapide que CSR (l'ancien `self.W.getcol(j)`
    sur CSR est O(nnz total) par appel, pas O(nnz de la colonne)."""
    import time
    from scipy import sparse
    rng = np.random.default_rng(0)
    n_docs, n_terms = 20_000, 8192
    W = sparse.random(n_docs, n_terms, density=0.002, random_state=rng, format="csr")
    W_csc = W.tocsc()
    cols = rng.integers(0, n_terms, size=200)

    start = time.perf_counter()
    for j in cols:
        W.getcol(j)
    csr_time = time.perf_counter() - start

    start = time.perf_counter()
    for j in cols:
        W_csc.getcol(j)
    csc_time = time.perf_counter() - start

    assert csc_time < csr_time
