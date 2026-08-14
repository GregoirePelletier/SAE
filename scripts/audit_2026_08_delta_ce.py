"""
scripts/audit_2026_08_delta_ce.py — Intègre B.20 de `docs/AUDIT_2026-08.md` :
`src/sae/compare/crosslingual.py::ce_loss_increase` (ΔCE = CE(patched) -
CE(clean), métrique standard SAEBench/SAE Boost) est déjà implémentée mais
n'était jamais appelée. Off-by-one de couche corrigé au passage (E.6, cf.
crosslingual.py) -- jamais appelée avant ce correctif, aucun résultat publié
n'en dépendait.

Compare le core seul (GemmaScope, JumpReLU) au core+extension
(FrozenCoreResidualSAE, checkpoint déjà entraîné) sur un petit échantillon de
mails originaux -- répond directement à la question que B.20 pose : la
substitution x -> SAE(x) dégrade-t-elle la cross-entropy du LLM, et de
combien, dans un référentiel directement comparable à GemmaScope/SAE Boost ?

Usage : sbatch slurm/validation/run_audit_delta_ce.slurm
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import (
    MODEL_ID, HF_TOKEN, DTYPE, SAVE_DIR, LAYER, LOCAL_MAILS_PATH,
    LOCAL_AUGMENTED_MAILS_PATH, CORPUS_SPLIT_SEED, LOCAL_SAE_ROOT, SAE_SNAPSHOT,
    HOOK_TYPE, RELEASE_ID,
)
from src.data.preparation import build_email_train_test_corpus
from src.sae import load_gemma_scope_sae
from src.sae.frozen_core import ExtendedSAE
from src.sae.compare.crosslingual import ce_loss_increase

DEVICE = "cuda"
TORCH_DTYPE = torch.bfloat16 if DTYPE == "bf16" else torch.float16
N_DOCS = 60
SEED = 42
D_EXTRA, K_EXTRA = 1024, 32
SAE_ID = "layer_24_width_16k_l0_medium"  # pinné, cf. steering_fidelity_test.py
OUT_PATH = os.path.join("docs", "audit_delta_ce_results.json")


def get_sample_texts() -> list[str]:
    train_texts, train_labels, _, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED,
    )
    originals = [t for t, l in zip(train_texts, train_labels) if l == "original"]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(originals), size=min(N_DOCS, len(originals)), replace=False)
    return [originals[i] for i in idx]


def main():
    texts = get_sample_texts()
    print(f"[delta-ce] {len(texts)} mails originaux échantillonnés (seed={SEED}).")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=TORCH_DTYPE, device_map=DEVICE,
        low_cpu_mem_usage=True, token=HF_TOKEN, trust_remote_code=True, local_files_only=True,
    ).eval()

    sae_dir = os.path.join(LOCAL_SAE_ROOT, "snapshots", SAE_SNAPSHOT, HOOK_TYPE, SAE_ID)
    core = load_gemma_scope_sae(
        sae_dir=sae_dir, device=DEVICE, release_id=RELEASE_ID, sae_id=f"{HOOK_TYPE}/{SAE_ID}",
    ).to(DEVICE).eval()
    core.requires_grad_(False)

    print("[delta-ce] Condition CORE SEUL (GemmaScope, JumpReLU)...")
    res_core = ce_loss_increase(texts, model, tokenizer, core, layer=LAYER, device=DEVICE)
    print(f"  ce_clean={res_core['ce_clean']:.4f}  ce_patched={res_core['ce_patched']:.4f}  "
          f"delta_ce={res_core['delta_ce']:.4f}")

    print("[delta-ce] Chargement de l'extension entraînée (FrozenCoreResidualSAE)...")
    ext_sae = ExtendedSAE(core, d_extra=D_EXTRA, k_extra=K_EXTRA).to(DEVICE)
    ckpt_path = os.path.join(SAVE_DIR, f"p1_frozen_core_d{D_EXTRA}_k{K_EXTRA}.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    ext_sae.load_state_dict(ckpt["state_dict"], strict=False)
    ext_sae.eval()

    print("[delta-ce] Condition CORE+EXTENSION (FrozenCoreResidualSAE)...")
    res_ext = ce_loss_increase(texts, model, tokenizer, ext_sae, layer=LAYER, device=DEVICE)
    print(f"  ce_clean={res_ext['ce_clean']:.4f}  ce_patched={res_ext['ce_patched']:.4f}  "
          f"delta_ce={res_ext['delta_ce']:.4f}")

    results = {
        "n_docs": len(texts), "layer": LAYER, "sae_id_core": SAE_ID,
        "d_extra": D_EXTRA, "k_extra": K_EXTRA,
        "core_only": res_core, "core_plus_extension": res_ext,
        "delta_ce_increase_from_extension": res_ext["delta_ce"] - res_core["delta_ce"],
    }
    print(f"\n[delta-ce] Écart ΔCE (core+ext) - ΔCE (core seul) = "
          f"{results['delta_ce_increase_from_extension']:+.4f} "
          f"(positif = l'extension dégrade la fidélité fonctionnelle malgré une "
          f"meilleure FVE de reconstruction ; négatif = elle l'améliore)")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Écrit : {OUT_PATH}")


if __name__ == "__main__":
    main()
