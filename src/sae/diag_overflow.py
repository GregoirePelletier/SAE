#!/usr/bin/env python3
"""Diagnostic overflow fp16 branche extra — lancer avec: python3 -u diag_overflow.py"""
import os, sys, gc, resource, torch
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gemma_scope_loader import load_gemma_scope_sae
from frozen_core import ExtendedSAE

def rss_gib(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB->GB

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE={DEVICE} | RSS={rss_gib():.2f} GiB", flush=True)

SAVE_DIR = "/home/h21486/SAE/results_v9_test"
CACHE_DIR = "/home/h21486/SAE/results_v9_test/cache"
D_EXTRA, K_EXTRA = 1024, 32
BATCH = 512  # réduit pour isoler : si ça meurt pareil, ce n'est pas la taille du batch

resid_path = os.path.join(CACHE_DIR, "p1_raw_residuals.pt")
ckpt_path = os.path.join(SAVE_DIR, f"p1_frozen_core_d{D_EXTRA}_k{K_EXTRA}.pt")
assert os.path.exists(resid_path) and os.path.exists(ckpt_path)

core_sae = load_gemma_scope_sae(
    sae_dir="/home/h21486/SAE/saes/gemma-scope-2-12b-it-res/snapshots/0000000000000000000000000000000000000000/resid_post/layer_24_width_262k_l0_medium",
).to(DEVICE).to(torch.bfloat16).eval()
core_sae.requires_grad_(False)
print(f"core_sae chargé | RSS={rss_gib():.2f} GiB", flush=True)

model = ExtendedSAE(core_sae, d_extra=D_EXTRA, k_extra=K_EXTRA).to(DEVICE).to(torch.bfloat16)
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()
print(f"model chargé | RSS={rss_gib():.2f} GiB", flush=True)

raw_residuals = torch.load(resid_path, map_location="cpu", weights_only=True)
print(f"raw_residuals: {raw_residuals.shape} | RSS={rss_gib():.2f} GiB", flush=True)

with torch.no_grad():
    for i in range(0, len(raw_residuals), BATCH):
        x = raw_residuals[i:i+BATCH].to(DEVICE).to(torch.bfloat16)
        core_acts = model.core_sae.encode(x)
        core_out = model.core_sae.decode(core_acts)
        residual = x - core_out
        pre = model._pre_extra(residual)
        extra_acts = model.topk_extra(pre)

        pre_max = pre.float().abs().max().item()
        n_over_fp16 = (pre.float().abs() > 65504).sum().item()
        resid_norm_max = residual.float().norm(dim=-1).max().item()
        finite = torch.isfinite(extra_acts).all().item()

        print(f"batch {i:>7d} | RSS={rss_gib():.2f} GiB | pre_max={pre_max:.1f} | "
              f">fp16max={n_over_fp16} | resid_norm_max={resid_norm_max:.1f} | finite={finite}",
              flush=True)

        del x, core_acts, core_out, residual, pre, extra_acts
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

print("Scan terminé.", flush=True)