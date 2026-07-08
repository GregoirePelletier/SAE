import torch
from src.sae.gemma_scope_loader import load_gemma_scope_sae
from src.sae.frozen_core import ExtendedSAE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
core = load_gemma_scope_sae(
    sae_dir="/home/h21486/SAE/saes/gemma-scope-2-12b-it-res/snapshots/"
            "0000000000000000000000000000000000000000/resid_post/layer_24_width_262k_l0_medium",
    device=DEVICE).to(DEVICE).to(torch.bfloat16).eval()
core.requires_grad_(False)
ext = ExtendedSAE(core, d_extra=1024, k_extra=32).to(DEVICE).to(torch.bfloat16)
ckpt = torch.load("results_v9_test/p1_frozen_core_d1024_k32.pt", map_location=DEVICE, weights_only=False)
ext.load_state_dict(ckpt["state_dict"]); ext.eval()
raw = torch.load("results_v9_test/cache/p1_raw_residuals.pt", weights_only=True)[:8192].to(DEVICE).to(torch.bfloat16)
with torch.no_grad():
    norms = raw.float().norm(dim=-1)
    residual = raw - core.decode(core.encode(raw))
    pre = ext._pre_extra(residual).float()
for f in [262144, 262369, 262283, 262217, 262316, 262296, 262228, 262185]:
    rho = torch.corrcoef(torch.stack([pre[:, f - 262144], norms]))[0, 1].item()
    print(f"F{f}: Pearson(pre, ||x_t||) = {rho:.4f}")
