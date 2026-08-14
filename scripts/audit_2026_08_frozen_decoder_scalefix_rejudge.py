"""
scripts/audit_2026_08_frozen_decoder_scalefix_rejudge.py — Suite de
audit_2026_08_frozen_decoder_scale_fix.py (job 43255, checkpoint deja
entraine) : rejuge les MEMES 150 features que le run principal /
FrozenDecoderExtendedSAE original, avec le decodeur ScaleCalibratedFrozenDecoderSAE
(decodeur aleatoire fige, input_scale calibre a 3993.48 au lieu de 1.0).

Isole l'effet du confond input_scale (A.3 point 3 de docs/AUDIT_2026-08.md) :
compare a 45,3% (entraine, echelle calibree) et 29,3% (FrozenDecoderExtendedSAE
original, echelle=1.0 non calibree, meme decodeur aleatoire dans les deux cas
sauf la calibration d'echelle).

Necessite de recalculer les activations doc-level (max-pool) et les fragments
token-level pour ce nouveau checkpoint -- contrairement aux scripts precedents
qui reutilisaient p1_all_doc_acts_ext_d1024.pt (produit par le checkpoint
ENTRAINE), ce checkpoint a un decodeur different -> les codes d'encodage
(partie encodeur seulement, W_enc_extra/b_enc_extra) sont specifiques a CE
checkpoint. Recalcule donc les activations sur les mails du run principal.

Usage : sbatch slurm/validation/run_audit_frozen_decoder_scalefix_rejudge.slurm
"""
from __future__ import annotations

import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import (
    MODEL_ID, HF_TOKEN, DTYPE, SAVE_DIR, LAYER, LOCAL_MAILS_PATH,
    LOCAL_AUGMENTED_MAILS_PATH, CORPUS_SPLIT_SEED, LOCAL_SAE_ROOT, SAE_SNAPSHOT,
    HOOK_TYPE, RELEASE_ID,
)
from src.data.preparation import build_email_train_test_corpus
from src.analysis.activations import extract_residual_acts, maxpool_sae_docs
from src.sae import load_gemma_scope_sae
from src.sae.frozen_core import FrozenDecoderExtendedSAE
from src.sae.judge import odd_one_out_judge

DEVICE = "cuda"
TORCH_DTYPE = torch.bfloat16 if DTYPE == "bf16" else torch.float16
D_EXTRA, K_EXTRA = 1024, 32
SAE_ID = "layer_24_width_16k_l0_medium"
CKPT_PATH = os.path.join(SAVE_DIR, "audit_2026_08_frozen_decoder_scalefix_d1024_k32.pt")
TOKEN_FRAGMENTS_DIR = os.path.join(SAVE_DIR, "cache", "p1_token_fragments")
ORIGINAL_JUDGE_CACHE = os.path.join(SAVE_DIR, "cache", "p1_judge_labels_extended.json")
OUT_PATH = os.path.join(SAVE_DIR, "cache", "audit_2026_08_frozen_decoder_scalefix_rejudge.json")


class ScaleCalibratedFrozenDecoderSAE(FrozenDecoderExtendedSAE):
    pass  # même classe que le script d'entraînement -- state_dict compatible


def main():
    print("[scalefix-rejudge] Chargement du SAE core...")
    sae_dir = os.path.join(LOCAL_SAE_ROOT, "snapshots", SAE_SNAPSHOT, HOOK_TYPE, SAE_ID)
    core = load_gemma_scope_sae(
        sae_dir=sae_dir, device=DEVICE, release_id=RELEASE_ID, sae_id=f"{HOOK_TYPE}/{SAE_ID}",
    ).to(DEVICE).eval()
    core.requires_grad_(False)

    ext_sae = ScaleCalibratedFrozenDecoderSAE(core, d_extra=D_EXTRA, k_extra=K_EXTRA).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    ext_sae.load_state_dict(ckpt["state_dict"], strict=False)
    ext_sae.eval()
    print(f"[scalefix-rejudge] input_scale chargé = {ext_sae.input_scale.item():.4f}")

    with open(ORIGINAL_JUDGE_CACHE, encoding="utf-8") as f:
        original_judge_data = json.load(f)
    feature_indices = [int(k) for k in original_judge_data.keys()]
    print(f"[scalefix-rejudge] {len(feature_indices)} features (mêmes que le run principal).")

    print("[scalefix-rejudge] Reconstruction du split train (déterministe)...")
    train_texts, _, _, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED,
    )
    n_train = len(train_texts)

    print(f"[scalefix-rejudge] Chargement du LLM {MODEL_ID} pour extraction...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=TORCH_DTYPE, device_map=DEVICE,
        low_cpu_mem_usage=True, token=HF_TOKEN, trust_remote_code=True, local_files_only=True,
    ).eval()

    print(f"[scalefix-rejudge] Extraction + encodage des {n_train} mails train (ce checkpoint)...")
    stream = extract_residual_acts(train_texts, model, tokenizer, layer=LAYER, device=DEVICE)

    def encode_fn(x):
        with torch.no_grad():
            return ext_sae.encode(x.to(DEVICE))

    train_doc_acts = maxpool_sae_docs(
        act_stream=stream, encode_fn=encode_fn, n_docs=n_train,
        d_sae=core.cfg.d_sae + D_EXTRA, device=DEVICE,
    )
    print(f"[scalefix-rejudge] train_doc_acts: {tuple(train_doc_acts.shape)}")

    # Libère le LLM utilisé pour l'extraction avant de le recharger comme juge
    # (même modèle, mais évite de garder deux instances en mémoire simultanément
    # n'est pas nécessaire ici -- même objet réutilisé directement pour le jugement).
    print("[scalefix-rejudge] Jugement odd-one-out (même protocole, même juge)...")
    results = odd_one_out_judge(
        model, tokenizer, feature_indices, TOKEN_FRAGMENTS_DIR, train_doc_acts,
        offset=0, n_pos=9,
    )

    n = len(results)
    n_interp = sum(1 for v in results.values() if v.get("interp_score") == 1)
    rate = n_interp / n
    n_orig_interp = sum(1 for v in original_judge_data.values() if v.get("interp_score") == 1)

    summary = {
        "n_tested": n, "n_interp": n_interp, "rate": rate,
        "input_scale": float(ext_sae.input_scale.item()),
        "reference_trained_calibrated": {"n": len(original_judge_data), "n_interp": n_orig_interp,
                                          "rate": n_orig_interp / len(original_judge_data)},
        "reference_frozen_decoder_uncalibrated_scale1": {"rate": 0.293, "note": "cf. RESULTS_TESTS.md §19"},
    }
    print("\n" + "=" * 70)
    print(" RÉSUMÉ — FROZEN DECODER, input_scale CALIBRÉ (vs 1.0 original)")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_feature": results}, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Écrit : {OUT_PATH}")


if __name__ == "__main__":
    main()
