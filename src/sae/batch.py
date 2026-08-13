"""
batch.py — BatchTopK d'entraînement + seuil global θ pour l'inférence.

  - Train : BatchTopK sélectionne les k·B plus grandes pré-activations DU BATCH
    (budget partagé — L0 moyen = k, mais variable par échantillon).
  - Eval : un TopK per-sample forcerait L0 = k exactement, une distribution
    d'activations différente de celle vue en train. À la place, JumpReLU avec
    seuil global (Bussmann et al. 2024) :
        θ = E_batches[ min{ z_i > 0 sélectionnés } ]
    estimé par EMA pendant l'entraînement. L'inférence devient per-sample,
    déterministe, indépendante de la composition du batch.

API :
  - BatchTopKEncoder : module à état (buffers threshold/calibrated), à instancier
    une fois par SAE.
"""
import torch
import torch.nn as nn


def _batch_topk(pre_acts: torch.Tensor, k: int) -> torch.Tensor:
    """Vrai BatchTopK : top (k·B) sur le batch aplati, budget partagé."""
    B = pre_acts.shape[0]
    n_keep = min(k * B, pre_acts.numel())
    flat = pre_acts.reshape(-1)
    vals, idx = flat.topk(n_keep)
    out = torch.zeros_like(flat).scatter_(0, idx, vals.clamp(min=0.0))
    return out.reshape(pre_acts.shape)


class BatchTopKEncoder(nn.Module):
    """
    Activation BatchTopK avec calibration du seuil d'inférence.
    Buffers persistants → θ sauvegardé/restauré avec le state_dict du SAE.
    """

    def __init__(self, k: int, ema: float = 0.99):
        super().__init__()
        self.k = k
        self.ema = ema
        self.register_buffer("threshold", torch.tensor(0.0))
        self.register_buffer("calibrated", torch.tensor(False))

    def forward(self, pre_acts: torch.Tensor) -> torch.Tensor:
        if self.training:
            acts = _batch_topk(pre_acts, self.k)
            with torch.no_grad():
                pos = acts[acts > 0]
                if pos.numel() > 0:
                    m = pos.min().float()
                    if not bool(self.calibrated):
                        self.threshold.fill_(m)
                        self.calibrated.fill_(True)
                    else:
                        self.threshold.mul_(self.ema).add_((1 - self.ema) * m)
            return acts

        if bool(self.calibrated):
            # JumpReLU global : z * 1[z > θ]
            return pre_acts * (pre_acts > self.threshold)

        # Fallback si jamais entraîné (chargement d'anciens checkpoints) :
        k_clamp = min(self.k, pre_acts.shape[-1])
        vals, idx = pre_acts.topk(k_clamp, dim=-1)
        return torch.zeros_like(pre_acts).scatter_(-1, idx, vals.clamp(min=0.0))