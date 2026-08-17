"""
scripts/audit_2026_08_delta_ce_v2.py — Suite de audit_2026_08_delta_ce.py
(job 43273, résultat déjà intégré, RESULTS_TESTS.md §58), répond aux deux
demandes du round 3 de l'audit (§2, §3) :
  1. ΔCE PAR DOCUMENT (pas seulement l'agrégat) + test apparié (Wilcoxon
     signed-rank) entre core seul et core+extension sur les mêmes 60
     documents.
  2. Sanity check indépendant de `ce_loss_increase` (fraîchement corrigée,
     3 bugs) : patch par un SAE IDENTITÉ (encode=decode=identité) --
     ΔCE doit être ≈0 si le mécanisme de patch lui-même est correct,
     indépendamment de tout SAE réel.

Usage : sbatch slurm/validation/run_audit_delta_ce_v2.slurm
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from scipy.stats import wilcoxon
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
SAE_ID = "layer_24_width_16k_l0_medium"
OUT_PATH = os.path.join("docs", "audit_delta_ce_v2_results.json")


class IdentitySAE:
    """Sanity check : encode/decode = identité stricte. Patcher avec ceci ne
    doit rien changer -- ΔCE doit être ≈0 (à l'erreur d'arrondi bf16 près)."""
    def encode(self, x):
        return x

    def decode(self, x):
        return x


def get_sample_texts():
    train_texts, train_labels, _, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED,
    )
    originals = [t for t, l in zip(train_texts, train_labels) if l == "original"]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(originals), size=min(N_DOCS, len(originals)), replace=False)
    return [originals[i] for i in idx]


def main():
    texts = get_sample_texts()
    print(f"[delta-ce-v2] {len(texts)} mails originaux échantillonnés (seed={SEED}).")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=TORCH_DTYPE, device_map=DEVICE,
        low_cpu_mem_usage=True, token=HF_TOKEN, trust_remote_code=True, local_files_only=True,
    ).eval()

    # ── Sanity check : SAE identité ──────────────────────────────────────
    print("\n[delta-ce-v2] Sanity check : patch par SAE IDENTITÉ (doit donner ΔCE≈0)...")
    identity_sae = IdentitySAE()
    res_identity = ce_loss_increase(texts[:20], model, tokenizer, identity_sae, layer=LAYER, device=DEVICE)
    print(f"  ce_clean={res_identity['ce_clean']:.6f} ce_patched={res_identity['ce_patched']:.6f} "
          f"delta_ce={res_identity['delta_ce']:.6f}")
    identity_ok = abs(res_identity["delta_ce"]) < 0.01
    print(f"  Sanity check {'RÉUSSI' if identity_ok else 'ÉCHOUÉ'} "
          f"(seuil |delta_ce| < 0.01, obtenu {abs(res_identity['delta_ce']):.6f})")

    # ── Core seul, per-doc ────────────────────────────────────────────────
    sae_dir = os.path.join(LOCAL_SAE_ROOT, "snapshots", SAE_SNAPSHOT, HOOK_TYPE, SAE_ID)
    core = load_gemma_scope_sae(
        sae_dir=sae_dir, device=DEVICE, release_id=RELEASE_ID, sae_id=f"{HOOK_TYPE}/{SAE_ID}",
    ).to(DEVICE).eval()
    core.requires_grad_(False)

    print("\n[delta-ce-v2] Condition CORE SEUL, ΔCE par document...")
    res_core = ce_loss_increase(texts, model, tokenizer, core, layer=LAYER, device=DEVICE, return_per_doc=True)
    print(f"  ce_clean={res_core['ce_clean']:.4f} ce_patched={res_core['ce_patched']:.4f} "
          f"delta_ce={res_core['delta_ce']:.4f}")

    # ── Core + extension, per-doc ────────────────────────────────────────
    ext_sae = ExtendedSAE(core, d_extra=D_EXTRA, k_extra=K_EXTRA).to(DEVICE)
    ckpt = torch.load(os.path.join(SAVE_DIR, f"p1_frozen_core_d{D_EXTRA}_k{K_EXTRA}.pt"),
                       map_location=DEVICE, weights_only=False)
    ext_sae.load_state_dict(ckpt["state_dict"], strict=False)
    ext_sae.eval()

    print("\n[delta-ce-v2] Condition CORE+EXTENSION, ΔCE par document...")
    res_ext = ce_loss_increase(texts, model, tokenizer, ext_sae, layer=LAYER, device=DEVICE, return_per_doc=True)
    print(f"  ce_clean={res_ext['ce_clean']:.4f} ce_patched={res_ext['ce_patched']:.4f} "
          f"delta_ce={res_ext['delta_ce']:.4f}")

    # ── Test apparié (Wilcoxon signed-rank), même documents, même ordre ─
    delta_core = np.array([d["delta_ce"] for d in res_core["per_doc"]])
    delta_ext = np.array([d["delta_ce"] for d in res_ext["per_doc"]])
    stat, p = wilcoxon(delta_core, delta_ext, alternative="greater")  # H1: core seul dégrade plus
    n_ext_better = int((delta_ext < delta_core).sum())

    print(f"\n[delta-ce-v2] Wilcoxon signed-rank (H1: ΔCE core seul > ΔCE core+extension) : "
          f"stat={stat:.1f} p={p:.2e}")
    print(f"[delta-ce-v2] Extension fait mieux sur {n_ext_better}/{len(delta_core)} documents.")

    results = {
        "sanity_check_identity_sae": {**res_identity, "passed": identity_ok},
        "core_only": {k: v for k, v in res_core.items() if k != "per_doc"},
        "core_plus_extension": {k: v for k, v in res_ext.items() if k != "per_doc"},
        "per_doc_core": res_core["per_doc"],
        "per_doc_extension": res_ext["per_doc"],
        "wilcoxon_core_gt_extension": {"statistic": float(stat), "p": float(p)},
        "n_docs_extension_better": n_ext_better, "n_docs_total": len(delta_core),
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Écrit : {OUT_PATH}")


if __name__ == "__main__":
    main()
