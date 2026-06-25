import torch
from sae_lens.saes.batchtopk_sae import BatchTopK


def batch_topk_encode(pre_acts: torch.Tensor, k: int, training: bool) -> torch.Tensor:
    if training:
        batch_topk = BatchTopK(k=float(k))
        return batch_topk(pre_acts)

    k_clamp = min(k, pre_acts.shape[-1])
    topk_vals, topk_idx = pre_acts.topk(k_clamp, dim=-1)
    return torch.zeros_like(pre_acts).scatter_(-1, topk_idx, topk_vals)
