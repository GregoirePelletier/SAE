import torch

def test_max_pool():

    acts = torch.randn(64, 4096)

    pooled = acts.max(dim=0).values

    assert pooled.shape == (4096,)