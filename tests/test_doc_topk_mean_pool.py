"""Tests CPU rapides pour src/storage/fragment_store.py::doc_topk_mean_pool
(alternative de pooling E01, plan §6.3)."""
import torch

from src.storage.fragment_store import doc_topk_mean_pool, doc_maxpool


def _make_frag(dense: torch.Tensor) -> dict:
    """Construit un fragment CSR a partir d'un tenseur dense [T, d] (les
    zeros ne sont jamais stockes -- meme convention que le vrai pipeline)."""
    T, d = dense.shape
    rows, cols = torch.nonzero(dense, as_tuple=True)
    vals = dense[rows, cols]
    # rowptr non utilise par doc_topk_mean_pool/doc_maxpool (ils ne lisent
    # que cols/vals/shape) -- construit correctement quand meme pour rester
    # un fragment CSR valide et reutilisable par d'autres fonctions.
    counts = torch.bincount(rows, minlength=T)
    rowptr = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])
    return {"rowptr": rowptr, "cols": cols.to(torch.int32), "vals": vals.float(), "shape": (T, d)}


def test_topk1_matches_maxpool():
    torch.manual_seed(0)
    dense = torch.rand(10, 5) * (torch.rand(10, 5) > 0.5)  # sparse aleatoire
    frag = _make_frag(dense)
    top1 = doc_topk_mean_pool(frag, k=1)
    mx = doc_maxpool(frag)
    assert torch.allclose(top1, mx, atol=1e-6)


def test_topk3_matches_brute_force_dense_reference():
    torch.manual_seed(1)
    T, d = 8, 4
    dense = torch.rand(T, d) * (torch.rand(T, d) > 0.3)
    frag = _make_frag(dense)
    result = doc_topk_mean_pool(frag, k=3)

    expected = torch.zeros(d)
    k_eff = min(3, T)
    for f in range(d):
        col = dense[:, f]
        top_vals, _ = torch.topk(col, k=k_eff)
        expected[f] = top_vals.sum() / k_eff
    assert torch.allclose(result, expected, atol=1e-6)


def test_feature_with_fewer_nonzero_than_k_includes_implicit_zeros():
    # Feature 0 n'a qu'UNE seule activation non nulle sur 10 tokens (T=10>=k=3) :
    # top-3 = [valeur, 0, 0], diviseur=3 (T>=k) -- pas juste diviseur=1.
    T, d = 10, 2
    dense = torch.zeros(T, d)
    dense[3, 0] = 9.0
    frag = _make_frag(dense)
    result = doc_topk_mean_pool(frag, k=3)
    assert torch.isclose(result[0], torch.tensor(9.0 / 3))
    assert result[1] == 0.0


def test_document_shorter_than_k_divides_by_document_length():
    # T=2 < k=3 : diviseur = T = 2, pas 3 -- moyenne sur tout ce qui existe.
    T, d = 2, 1
    dense = torch.tensor([[4.0], [2.0]])
    frag = _make_frag(dense)
    result = doc_topk_mean_pool(frag, k=3)
    assert torch.isclose(result[0], torch.tensor((4.0 + 2.0) / 2))


def test_empty_fragment_returns_zeros():
    frag = {
        "rowptr": torch.zeros(1, dtype=torch.int64),
        "cols": torch.zeros(0, dtype=torch.int32),
        "vals": torch.zeros(0, dtype=torch.float32),
        "shape": (5, 4),
    }
    result = doc_topk_mean_pool(frag, k=3)
    assert torch.equal(result, torch.zeros(4))


def test_output_never_negative_and_is_shape_d():
    torch.manual_seed(2)
    T, d = 20, 6
    dense = torch.rand(T, d) * (torch.rand(T, d) > 0.6)
    frag = _make_frag(dense)
    result = doc_topk_mean_pool(frag, k=3)
    assert result.shape == (d,)
    assert (result >= 0).all()
