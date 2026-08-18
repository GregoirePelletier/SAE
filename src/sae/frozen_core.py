"""
frozen_core.py — FrozenCoreResidualSAE / ExtendedSAE : extension résiduelle
avec BatchTopKEncoder (seuil θ persistant) et AuxK sur la branche extra.

encode/decode concaténés [core | extra], core gelé, décodeur normalisé,
branche extra en fp32, initialisation PCA sur le résidu.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sae_lens import SAE

try:
    from src.sae.batch import BatchTopKEncoder
except ImportError:
    from batch import BatchTopKEncoder

AUX_ALPHA = 1.0 / 32.0


class FrozenCoreResidualSAE(nn.Module):
    def __init__(self, core_sae: SAE, d_extra: int = 1024, k_extra: int = 32,
                 dead_steps_threshold: int = 200):
        super().__init__()
        self.core_sae = core_sae
        self.core_sae.requires_grad_(False)
        self.d_in = core_sae.cfg.d_in
        self.d_extra = d_extra
        self.k_extra = k_extra
        self.k_aux = min(2 * k_extra, d_extra // 2)
        self.dead_steps_threshold = dead_steps_threshold

        W_dec = F.normalize(torch.randn(d_extra, self.d_in), dim=1)
        self.W_dec_extra = nn.Parameter(W_dec)
        self.W_enc_extra = nn.Parameter(W_dec.T.clone())
        self.b_enc_extra = nn.Parameter(torch.zeros(d_extra))
        self.topk_extra = BatchTopKEncoder(k_extra)
        self.register_buffer("steps_since_active_extra", torch.zeros(d_extra))
        self.register_buffer("input_scale", torch.tensor(1.0))

    def _pre_extra(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() / self.input_scale) @ self.W_enc_extra.float() + self.b_enc_extra.float()

    def _encode_extra_acts(self, x: torch.Tensor) -> torch.Tensor:
        return self.topk_extra(self._pre_extra(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_bf16 = x.to(torch.bfloat16)
        with torch.no_grad():
            core_acts = self.core_sae.encode(x_bf16)
            core_out = self.core_sae.decode(core_acts)
        extra_acts = self._encode_extra_acts(x_bf16 - core_out)
        return torch.cat([core_acts.float(), extra_acts.float()], dim=-1)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        d_core = self.core_sae.cfg.d_sae
        core_acts = acts[:, :d_core].to(torch.bfloat16)   # le core GemmaScope attend du bf16
        extra_acts = acts[:, d_core:].float()             # la branche extra travaille en fp32
        with torch.no_grad():
            core_out = self.core_sae.decode(core_acts)
        extra_out = (extra_acts @ self.W_dec_extra.float()) * self.input_scale
        return core_out.float() + extra_out
    
    def _aux_loss(self, pre: torch.Tensor, err: torch.Tensor) -> torch.Tensor:
        dead = self.steps_since_active_extra > self.dead_steps_threshold
        n_dead = int(dead.sum())
        if n_dead == 0:
            return torch.zeros((), device=pre.device, dtype=pre.dtype)
        k_aux = min(self.k_aux, n_dead)
        pre_dead = pre.masked_fill(~dead.unsqueeze(0), float("-inf"))
        vals, idx = pre_dead.topk(k_aux, dim=-1)
        f_aux = torch.zeros_like(pre).scatter_(-1, idx, vals.clamp(min=0.0))
        e_hat = (f_aux @ self.W_dec_extra.float()) * self.input_scale
        return F.mse_loss(e_hat, err) / (err.pow(2).mean() + 1e-8)

    def forward(self, x: torch.Tensor) -> dict:
        x_bf16 = x.to(torch.bfloat16)
        with torch.no_grad():
            core_acts = self.core_sae.encode(x_bf16)
            core_out = self.core_sae.decode(core_acts)

        # `residual` doit être fp32 dès cette ligne : core_sae travaille en bf16 (cf.
        # commentaire decode()) mais toute la branche "extra" (paramètres, loss) est fp32
        # partout ailleurs dans ce fichier. Laisser `residual` en bf16 jusqu'ici et compter
        # sur la promotion implicite fp32/bf16 dans mse_loss/var_residual/_aux_loss casse le
        # backward ("Found dtype BFloat16 but expected Float" — la promotion marche en
        # forward mais pas de façon fiable pour le gradient d'une op mêlant un tenseur fp32
        # avec grad_fn et un tenseur bf16 sans grad_fn). Confirmé par isolation empirique.
        #
        # Upcast AVANT la soustraction, pas après : x-core_out annule presque
        # entièrement (résidu ≈ quelques % de la norme de x) -- soustraire en bf16 perd
        # une grande partie de la précision utile avant même le cast, upcaster ensuite ne
        # la récupère pas. Mesuré empiriquement (`RESULTS_TESTS.md` §61) : ~6-7%
        # d'erreur relative injectée dans le résidu que l'extension apprend.
        residual = x_bf16.float() - core_out.float()
        pre = self._pre_extra(residual)
        extra_acts = self.topk_extra(pre)
        extra_out = (extra_acts @ self.W_dec_extra.float()) * self.input_scale

        mse_loss = F.mse_loss(extra_out, residual)
        var_residual = (residual - residual.mean(dim=0)).pow(2).mean()
        nmse = mse_loss / (var_residual + 1e-8)

        aux = torch.zeros((), device=x.device, dtype=x_bf16.dtype)
        if self.training:
            with torch.no_grad():
                active = (extra_acts > 1e-6).any(dim=0)
                self.steps_since_active_extra[active] = 0
                self.steps_since_active_extra[~active] += 1
            aux = self._aux_loss(pre, (residual - extra_out).detach())

        return {
            "sae_out": core_out + extra_out,
            "feature_acts": torch.cat([core_acts, extra_acts], dim=-1),
            "core_acts": core_acts,
            "extra_acts": extra_acts,
            "normalized_mse": nmse,
            "aux_loss": aux,
            "loss": nmse + AUX_ALPHA * aux,
            "l0_extra": (extra_acts.abs() > 1e-6).float().sum(dim=-1).mean(),
            "dead_frac": (self.steps_since_active_extra > self.dead_steps_threshold).float().mean(),
        }

    @torch.no_grad()
    def normalize_decoder(self):
        """Projection du gradient parallèle (si présent) puis renormalisation."""
        if self.W_dec_extra.grad is not None:
            parallel = (self.W_dec_extra.grad * self.W_dec_extra.data).sum(-1, keepdim=True) \
                       * self.W_dec_extra.data
            self.W_dec_extra.grad -= parallel
        self.W_dec_extra.data = F.normalize(self.W_dec_extra.data, dim=1)

    def export_to_fp32(self, save_path: str):
        print(f"  [Export] Export de l'adaptation sémantique française en float32 -> {save_path}")
        torch.save({k: v.cpu().float() for k, v in self.state_dict().items()}, save_path)


class FrozenDecoderExtendedSAE(FrozenCoreResidualSAE):
    """Sanity-check (Korznikov et al. 2026, "Sanity Checks for Sparse
    Autoencoders : Do SAEs Beat Random Baselines?") : W_dec_extra reste figé à
    son initialisation ALÉATOIRE (jamais mis à jour par le gradient) ; seuls
    l'encodeur (W_enc_extra, b_enc_extra) et le seuil BatchTopK sont appris
    normalement. Réplique leur baseline "Frozen Decoder", qui égalait un SAE
    entraîné sur interprétabilité/sparse probing/causal editing dans leur
    étude — teste si nos métriques (juge odd-one-out, sondes de
    classification) distinguent réellement un apprentissage de features
    significatif d'un simple ajustement de l'encodeur à des directions
    arbitraires. Volontairement PAS de sous-classe d'ExtendedSAE : ce dernier
    initialise le décodeur par PCA sur le résidu (des directions déjà
    informées par les données), ce qui affaiblirait le test — la baseline de
    référence doit partir d'un décodeur ALÉATOIRE, pas data-informed."""

    def __init__(self, core_sae, d_extra: int = 1024, k_extra: int = 32):
        super().__init__(core_sae, d_extra, k_extra)
        self.W_dec_extra.requires_grad_(False)

    @torch.no_grad()
    def normalize_decoder(self):
        """No-op : la renormalisation systématique du parent (`F.normalize` sur
        `.data`, appelée 2x/step par le harnais d'entraînement) introduirait un
        bruit flottant cumulatif sur des directions censées rester STRICTEMENT
        figées bit-à-bit sur des milliers de pas — annulée ici pour garantir un
        décodeur véritablement inchangé du premier au dernier pas."""
        pass


class ExtendedSAE(FrozenCoreResidualSAE):
    def __init__(self, core_sae: SAE, d_extra: int = 1024, k_extra: int = 32, domain_residuals=None):
        super().__init__(core_sae, d_extra, k_extra)
        if domain_residuals is not None:
            self._init_from_residual_pca(domain_residuals)

    def _init_from_residual_pca(self, residuals: torch.Tensor) -> None:
        print("  [ExtendedSAE] Initialisation PCA sur la distribution d'erreurs locale...")
        sample = residuals[:min(8192, len(residuals))].float()
        self.input_scale = sample.norm(dim=-1).median().to(self.input_scale.dtype)
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
            mean_residual = sample.mean(dim=0)
            self.b_enc_extra.data.copy_(
                (-(mean_residual / self.input_scale) @ self.W_enc_extra.data).to(self.b_enc_extra.dtype))
            print(f"  [ExtendedSAE] Initialisation réussie : {n_comp} directions PCA injectées.")
        except Exception as e:
            print(f"  [ExtendedSAE] Échec SVD ({e}), initialisation pseudo-aléatoire conservée.")