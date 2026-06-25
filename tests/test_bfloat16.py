import torch

def test_bfloat16_roundtrip():

    x = torch.randn(1024, 2304).to(torch.bfloat16)

    y = x.float().to(torch.bfloat16)

    assert y.dtype == torch.bfloat16

    err = (x.float() - y.float()).abs().mean()

    assert err < 1e-2