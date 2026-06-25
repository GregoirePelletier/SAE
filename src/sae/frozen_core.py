"""
frozen_core.py — Architecture FrozenCoreResidualSAE et ExtendedSAE.
Gestion rigoureuse de la précision : training/inférence en bf16, export final en float32.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sae_lens import SAE

from src.sae.batch import batch_topk_encode


class FrozenCoreResidualSAE(nn.Module):
    def __init__(self, core_sae: SAE, d_extra: int = 1024, k_extra: int = 32):
        super().__init__()
        self.core_sae = core_sae
        self.core_sae.requires_grad_(False)
        self.d_in = core_sae.cfg.d_in
        self.d_extra = d_extra
        self.k_extra = k_extra

        W_dec = F.normalize(torch.randn(d_extra, self.d_in), dim=1)
        self.W_dec_extra = nn.Parameter(W_dec)
        self.W_enc_extra = nn.Parameter(W_dec.T.clone())
        # FIX: b_enc_extra initialisé à zéro ; doit être mis à jour via
        # b_enc_extra.data = domain_residuals.mean(dim=0) @ W_enc_extra
        # pour éviter la solution triviale identité.
        self.b_enc_extra = nn.Parameter(torch.zeros(d_extra))

    def _encode_extra_acts(self, x: torch.Tensor) -> torch.Tensor:
        # FIX: F.relu supprimé — batch_topk_encode impose déjà la parcimonie ;
        # le relu avant TopK éliminait les directions négatives du résidu.
        pre = x @ self.W_enc_extra + self.b_enc_extra
        return batch_topk_encode(pre, self.k_extra, self.training)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            core_acts = self.core_sae.encode(x.to(torch.bfloat16))
        extra_acts = self._encode_extra_acts(x.to(torch.bfloat16))
        return torch.cat([core_acts, extra_acts], dim=-1)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        d_core = self.core_sae.cfg.d_sae
        core_acts = acts[:, :d_core].to(torch.bfloat16)
        extra_acts = acts[:, d_core:].to(torch.bfloat16)
        # FIX: no_grad ajouté sur core_sae.decode pour cohérence avec encode
        # et éviter l'accumulation de gradients sur le cœur gelé hors training.
        with torch.no_grad():
            core_out = self.core_sae.decode(core_acts)
        return core_out + extra_acts @ self.W_dec_extra

    def forward(self, x: torch.Tensor) -> dict:
        x_bf16 = x.to(torch.bfloat16)
        with torch.no_grad():
            core_acts = self.core_sae.encode(x_bf16)
            core_out = self.core_sae.decode(core_acts)

        residual = x_bf16 - core_out
        extra_acts = self._encode_extra_acts(residual)
        extra_out = extra_acts @ self.W_dec_extra

        mse_loss = F.mse_loss(extra_out, residual)

        # FIX: dénominateur NMSE = Var(résidu), PAS Var(x_original).
        # L'extension ne modélise que le résidu ; normaliser par Var(x) rendait
        # nmse artificiellement petit quand ‖r‖ ≪ ‖x‖.
        var_residual = (residual - residual.mean(dim=0)).pow(2).mean()
        nmse = mse_loss / (var_residual + 1e-8)

        return {
            "sae_out": core_out + extra_out,
            "feature_acts": torch.cat([core_acts, extra_acts], dim=-1),
            "core_acts": core_acts,
            "extra_acts": extra_acts,
            "normalized_mse": nmse,
            "l0_extra": (extra_acts.abs() > 1e-6).float().sum(dim=-1).mean(),
            "dead_frac": ((extra_acts.abs() > 1e-6).float().sum(dim=0) == 0).float().mean(),
        }

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec_extra.data = F.normalize(self.W_dec_extra.data, dim=1)

    def export_to_fp32(self, save_path: str):
        print(f"  [Export] Export de l'adaptation sémantique française en float32 -> {save_path}")
        state_dict_fp32 = {k: v.cpu().float() for k, v in self.state_dict().items()}
        torch.save(state_dict_fp32, save_path)


class ExtendedSAE(FrozenCoreResidualSAE):
    def __init__(self, core_sae: SAE, d_extra: int = 1024, k_extra: int = 32, domain_residuals=None):
        super().__init__(core_sae, d_extra, k_extra)
        if domain_residuals is not None:
            self._init_from_residual_pca(domain_residuals)

    def _init_from_residual_pca(self, residuals: torch.Tensor) -> None:
        """SVD/PCA sur les résidus pour aligner l'initialisation sur la distribution locale."""
        print("  [ExtendedSAE] Initialisation PCA sur la distribution d'erreurs locale...")
        sample = residuals[:min(8192, len(residuals))].float()
        centered = sample - sample.mean(dim=0)
        try:
            _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
            n_comp = min(self.d_extra, Vt.shape[0])
            W_init = F.normalize(Vt[:n_comp].float(), dim=1)
            if n_comp < self.d_extra:
                pad = F.normalize(torch.randn(self.d_extra - n_comp, self.d_in), dim=1)
                W_init = torch.cat([W_init, pad], dim=0)
            self.W_dec_extra.data.copy_(W_init.to(self.W_dec_extra.dtype))
            self.W_enc_extra.data.copy_(W_init.T.to(self.W_enc_extra.dtype))
            # Initialise b_enc_extra à la moyenne empirique projetée pour éviter
            # la solution triviale (b = 0 -> activations systématiquement centrées sur 0)
            mean_residual = centered.mean(dim=0)
            self.b_enc_extra.data.copy_(
                (mean_residual @ self.W_enc_extra.data).to(self.b_enc_extra.dtype)
            )
            print(f"  [ExtendedSAE] Initialisation réussie : {n_comp} directions PCA principales injectées.")
        except Exception as e:
            print(f"  [ExtendedSAE] Échec SVD ({e}), initialisation pseudo-aléatoire conservée.")