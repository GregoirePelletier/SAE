"""
a.py — Smoke test : valide que le SAE Gemma Scope se charge correctement.
"""

import os

from src.sae import load_gemma_scope_sae
from src.config import LOCAL_SAE_ROOT, SAE_SNAPSHOT, HOOK_TYPE, SAE_ID

SAE_DIR = os.environ.get(
    "SAE_DIR", os.path.join(LOCAL_SAE_ROOT, "snapshots", SAE_SNAPSHOT, HOOK_TYPE, SAE_ID)
)

if __name__ == "__main__":
    sae = load_gemma_scope_sae(SAE_DIR)
    print(sae.cfg)
    print("W_enc:", sae.W_enc.shape, "W_dec:", sae.W_dec.shape, "b_dec:", sae.b_dec.shape)
    print("[+] OK")