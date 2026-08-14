"""
scripts/audit_2026_08_random_init_trained_decoder.py — A.1 de
`docs/AUDIT_2026-08.md` : baseline "Extended SAE (init aléatoire)" du papier
SAE Boost, adaptée à ce dépôt (pas d'architecture core-extension à ce
papier -- ici, l'équivalent le plus proche et directement implémentable sans
changement d'architecture est un `ExtendedSAE` standard, décodeur ENTRAINABLE,
mais SANS initialisation PCA -- comble la case manquante du plan 2x2
"décodeur figé/entraîné x init PCA/aléatoire" :

  |                  | décodeur figé          | décodeur entraîné       |
  |------------------|-------------------------|--------------------------|
  | init PCA         | (non testé, PCA+figé    | ExtendedSAE standard,    |
  |                  |  contredirait l'intention| 45,3% (référence)        |
  |                  |  du sanity check)        |                          |
  | init aléatoire   | FrozenDecoderExtendedSAE | CE SCRIPT (nouveau)      |
  |                  | scale-fixe, 16,0%        |                          |

Isole si le gain de 45,3% (PCA+entraîné) vs 16,0-29,3% (aléatoire+figé) vient
de l'ENTRAINEMENT du décodeur, de l'INIT PCA, ou des deux -- ce test répond à
la part de l'entraînement seul (décodeur entraîné, mais parti d'un point
aléatoire comme le sanity check, pas d'un point informé par les données).

Usage : sbatch slurm/validation/run_audit_random_init_trained_decoder.slurm
"""
from __future__ import annotations

import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import (
    MODEL_ID, HF_TOKEN, DTYPE, SAVE_DIR, LAYER, LOCAL_MAILS_PATH,
    LOCAL_AUGMENTED_MAILS_PATH, CORPUS_SPLIT_SEED, LOCAL_SAE_ROOT, SAE_SNAPSHOT,
    HOOK_TYPE, RELEASE_ID, EPOCHS_EXTRA, LR_EXTRA,
)
from src.data.preparation import build_email_train_test_corpus
from src.analysis.activations import extract_residual_acts, maxpool_sae_docs
from src.sae import load_gemma_scope_sae
from src.sae.frozen_core import ExtendedSAE
from src.sae.sae_shared import load_or_train_extended_sae
from src.sae.judge import odd_one_out_judge

DEVICE = "cuda"
TORCH_DTYPE = torch.bfloat16 if DTYPE == "bf16" else torch.float16
D_EXTRA, K_EXTRA = 1024, 32
SAE_ID = "layer_24_width_16k_l0_medium"
RESIDUALS_PATH = os.path.join(SAVE_DIR, "cache", "p1_raw_residuals.pt")
NEW_MODEL_NAME = f"audit_2026_08_random_init_trained_d{D_EXTRA}_k{K_EXTRA}"
TOKEN_FRAGMENTS_DIR = os.path.join(SAVE_DIR, "cache", "p1_token_fragments")
ORIGINAL_JUDGE_CACHE = os.path.join(SAVE_DIR, "cache", "p1_judge_labels_extended.json")
OUT_PATH = os.path.join(SAVE_DIR, "cache", "audit_2026_08_random_init_trained_results.json")


def main():
    print("[random-init-trained] Chargement du SAE core...")
    sae_dir = os.path.join(LOCAL_SAE_ROOT, "snapshots", SAE_SNAPSHOT, HOOK_TYPE, SAE_ID)
    core = load_gemma_scope_sae(
        sae_dir=sae_dir, device=DEVICE, release_id=RELEASE_ID, sae_id=f"{HOOK_TYPE}/{SAE_ID}",
    ).to(DEVICE).eval()
    core.requires_grad_(False)

    print(f"[random-init-trained] Réservoir de résidus : {RESIDUALS_PATH}")
    raw_residuals = torch.load(RESIDUALS_PATH, map_location="cpu", weights_only=True)

    # ExtendedSAE SANS domain_residuals -> décodeur reste à l'init aléatoire de
    # FrozenCoreResidualSAE (F.normalize(randn)), input_scale reste 1.0 par
    # défaut (cohérent avec le sanity check original, PAS le scale-fix de
    # A.3.3 -- ce test isole spécifiquement l'effet de l'entraînement du
    # décodeur à partir d'un point aléatoire, pas la calibration d'échelle).
    model = ExtendedSAE(core, d_extra=D_EXTRA, k_extra=K_EXTRA, domain_residuals=None).to(DEVICE)
    print(f"[random-init-trained] input_scale (non calibré, comme le sanity check original) "
          f"= {model.input_scale.item():.4f}")

    print(f"[random-init-trained] Entraînement (décodeur ENTRAINABLE, init aléatoire) : "
          f"{EPOCHS_EXTRA} epochs, lr={LR_EXTRA}...")
    model, history = load_or_train_extended_sae(
        model=model, model_name=NEW_MODEL_NAME, acts_train=raw_residuals,
        epochs=EPOCHS_EXTRA, lr=LR_EXTRA, save_dir=SAVE_DIR, device=DEVICE,
    )
    final_val_loss = history.get("val_loss", [None])[-1] if history.get("val_loss") else None
    print(f"[random-init-trained] val_loss final = {final_val_loss}")

    with open(ORIGINAL_JUDGE_CACHE, encoding="utf-8") as f:
        original_judge_data = json.load(f)
    feature_indices = [int(k) for k in original_judge_data.keys()]

    print(f"[random-init-trained] Chargement du LLM {MODEL_ID} pour extraction...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True, local_files_only=True)
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=TORCH_DTYPE, device_map=DEVICE,
        low_cpu_mem_usage=True, token=HF_TOKEN, trust_remote_code=True, local_files_only=True,
    ).eval()

    train_texts, _, _, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED,
    )
    n_train = len(train_texts)
    print(f"[random-init-trained] Extraction + encodage des {n_train} mails train...")
    stream = extract_residual_acts(train_texts, llm, tokenizer, layer=LAYER, device=DEVICE)

    def encode_fn(x):
        with torch.no_grad():
            return model.encode(x.to(DEVICE))

    train_doc_acts = maxpool_sae_docs(
        act_stream=stream, encode_fn=encode_fn, n_docs=n_train,
        d_sae=core.cfg.d_sae + D_EXTRA, device=DEVICE,
    )

    print("[random-init-trained] Jugement odd-one-out (même protocole, mêmes 150 features)...")
    results = odd_one_out_judge(
        llm, tokenizer, feature_indices, TOKEN_FRAGMENTS_DIR, train_doc_acts, offset=0, n_pos=9,
    )
    n = len(results)
    n_interp = sum(1 for v in results.values() if v.get("interp_score") == 1)
    rate = n_interp / n

    summary = {
        "n_tested": n, "n_interp": n_interp, "rate": rate,
        "final_val_loss": final_val_loss,
        "reference_pca_init_trained": 0.4533333333333333,
        "reference_random_init_frozen_scale1": 0.293,
        "reference_random_init_frozen_scale_calibrated": 0.16,
    }
    print("\n" + "=" * 70)
    print(" RÉSUMÉ — DÉCODEUR ENTRAINÉ, INIT ALÉATOIRE (pas PCA)")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_feature": results}, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Écrit : {OUT_PATH}")


if __name__ == "__main__":
    main()
