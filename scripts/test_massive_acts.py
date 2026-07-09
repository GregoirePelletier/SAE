"""
scripts/test_massive_acts.py — Diagnostic corrélation Pearson(activation "extra", norme
du token) sur des features spécifiques, pour détecter la pollution par massive activations.

⚠ Script ad-hoc lié à un run cluster précis (12b, SAE width 262k, indices de features et
checkpoints p1_frozen_core_d1024_k32.pt / p1_raw_residuals.pt générés par ce run) : les
indices `f in [...]` ci-dessous supposent un SAE core de largeur 262144 (d_core=262144,
offset des features "extra" = 262144). Ne se généralise pas tel quel à un autre MODEL_SIZE
(ex. 270m, d_core=65536) sans réajuster ces indices et régénérer les checkpoints/résidus
via saev5.py --USE_FROZEN_CORE=1 pour ce modèle. Chemin SAE corrigé pour lire src.config
au lieu d'un chemin cluster figé ; le reste (checkpoints results_v9_test/*) reste un
artefact de run spécifique, non fourni ici.
"""
import os

import torch

from src.sae.gemma_scope_loader import load_gemma_scope_sae
from src.sae.frozen_core import ExtendedSAE
from src.config import LOCAL_SAE_ROOT, SAE_SNAPSHOT, HOOK_TYPE, SAE_ID, DTYPE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if DTYPE == "bf16" else torch.float16

SAE_DIR = os.environ.get(
    "SAE_DIR", os.path.join(LOCAL_SAE_ROOT, "snapshots", SAE_SNAPSHOT, HOOK_TYPE, SAE_ID)
)
core = load_gemma_scope_sae(sae_dir=SAE_DIR, device=DEVICE).to(DEVICE).to(TORCH_DTYPE).eval()
core.requires_grad_(False)
D_CORE = core.cfg.d_sae  # offset des features "extra" = d_core (dépend de MODEL_SIZE)

# Pas de .to(TORCH_DTYPE) sur ExtendedSAE : la branche "extra" doit rester fp32
# (cf. commentaire équivalent dans saev5.py) ; seul `core` (ci-dessus) est en TORCH_DTYPE.
ext = ExtendedSAE(core, d_extra=1024, k_extra=32).to(DEVICE)
ckpt = torch.load("results_v9_test/p1_frozen_core_d1024_k32.pt", map_location=DEVICE, weights_only=False)
ext.load_state_dict(ckpt["state_dict"]); ext.eval()
raw = torch.load("results_v9_test/cache/p1_raw_residuals.pt", weights_only=True)[:8192].to(DEVICE).to(TORCH_DTYPE)
with torch.no_grad():
    norms = raw.float().norm(dim=-1)
    residual = raw - core.decode(core.encode(raw))
    pre = ext._pre_extra(residual).float()
for f in [D_CORE, D_CORE + 225, D_CORE + 139, D_CORE + 73, D_CORE + 172, D_CORE + 152, D_CORE + 84, D_CORE + 41]:
    rho = torch.corrcoef(torch.stack([pre[:, f - D_CORE], norms]))[0, 1].item()
    print(f"F{f}: Pearson(pre, ||x_t||) = {rho:.4f}")
