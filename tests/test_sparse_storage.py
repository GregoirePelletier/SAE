import torch

def test_coo_roundtrip():

    dense = torch.randn(100, 500)

    dense[dense.abs() < 2.0] = 0

    sparse = dense.to_sparse_coo()

    recovered = sparse.to_dense()

    assert torch.allclose(dense, recovered)