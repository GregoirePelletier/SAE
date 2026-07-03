#!/usr/bin/env python3
"""Diagnostic overflow fp16 branche extra — à lancer depuis src/sae/."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gemma_scope_loader import load_gemma_scope_sae
from frozen_core import ExtendedSAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "/home/h21486/SAE/results_v9_test"
CACHE_DIR = "/home/h21486/SAE/results_v9_test/cache"
D_EXTRA, K_EXTRA = 1024, 32

resid_path = os.path.join(CACHE_DIR, "p1_raw_residuals.pt")
ckpt_path = os.path.join(SAVE_DIR, f"p1_frozen_core_d{D_EXTRA}_k{K_EXTRA}.pt")

assert os.path.exists(resid_path), f"introuvable: {resid_path}"
assert os.path.exists(ckpt_path), f"introuvable: {ckpt_path}"

core_sae = load_gemma_scope_sae(
    sae_dir="/home/h21486/SAE/saes/gemma-scope-2-12b-it-res/snapshots/0000000000000000000000000000000000000000/resid_post/layer_24_width_262k_l0_medium",
).to(DEVICE).to(torch.bfloat16).eval()
core_sae.requires_grad_(False)

model = ExtendedSAE(core_sae, d_extra=D_EXTRA, k_extra=K_EXTRA).to(DEVICE).to(torch.bfloat16)
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()

raw_residuals = torch.load(resid_path, map_location="cpu", weights_only=True)
print(f"raw_residuals: {raw_residuals.shape}")

with torch.no_grad():
    for i in range(0, len(raw_residuals), 4096):
        x = raw_residuals[i:i+4096].to(DEVICE).to(torch.bfloat16)
        core_acts = model.core_sae.encode(x)
        core_out = model.core_sae.decode(core_acts)
        residual = x - core_out

        pre = model._pre_extra(residual)
        pre_max = pre.float().abs().max().item()
        n_over_fp16 = (pre.float().abs() > 65504).sum().item()
        resid_norm = residual.float().norm(dim=-1)
        n_outlier_resid = (resid_norm > 1000).sum().item()

        if pre_max > 1000 or n_over_fp16 > 0:
            print(f"batch {i:>7d} | pre_max={pre_max:.1f} | >fp16max: {n_over_fp16} | "
                  f"resid_norm_max={resid_norm.max().item():.1f} | outliers_resid={n_outlier_resid}")

        extra_acts = model.topk_extra(pre)
        if not torch.isfinite(extra_acts).all():
            print(f"batch {i:>7d} : extra_acts NON FINI (nan/inf détecté)")

print("Scan terminé.")