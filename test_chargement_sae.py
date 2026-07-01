"""
a.py — Smoke test : valide que le SAE Gemma Scope se charge correctement.
"""

from src.sae import load_gemma_scope_sae

SAE_DIR = (
    "/home/h21486/SAE/saes/gemma-scope-2-12b-it-res/snapshots/"
    "0000000000000000000000000000000000000000/"
    "resid_post/layer_24_width_16k_l0_medium"
)

if __name__ == "__main__":
    sae = load_gemma_scope_sae(SAE_DIR)
    print(sae.cfg)
    print("W_enc:", sae.W_enc.shape, "W_dec:", sae.W_dec.shape, "b_dec:", sae.b_dec.shape)
    print("[+] OK")