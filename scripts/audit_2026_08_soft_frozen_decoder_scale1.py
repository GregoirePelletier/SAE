"""
scripts/audit_2026_08_soft_frozen_decoder_scale1.py — variante scale=1 (non
calibrée) de `audit_2026_08_soft_frozen_decoder.py`, pour comparaison directe
avec les deux autres baselines scale=1 (Frozen Decoder pur §19, décodeur
entraîné/init aléatoire §61). Le Frozen Decoder pur a vu son score CHUTER
quand l'échelle est calibrée (29,3%→16,0%, A.3.3) : ce test vérifie si le
Soft-Frozen Decoder présente le même effet inverse-à-l'attendu.

scripts/audit_2026_08_soft_frozen_decoder.py — A.3 point 2 de
`docs/AUDIT_2026-08.md` : implémente le "Soft-Frozen Decoder" de Korznikov
et al. (Sanity Checks for Sparse Autoencoders), jamais reproduit dans ce
dépôt malgré être la baseline la plus informative de leur étude (0,88, quasi
égale au SAE entraîné 0,90 -- contrairement au Frozen Decoder pur, largement
distancé). Contrainte : les directions du décodeur restent à cosinus >= 0.8
de leur initialisation ALÉATOIRE, mais peuvent tourner DANS ce cône plutôt
que rester strictement figées (Frozen Decoder) ou totalement libres
(ExtendedSAE standard).

Ne modifie PAS `frozen_core.py` (le sanity check Frozen Decoder pur, 29,3%
puis 16,0% à échelle calibrée, est déjà un résultat publié) : sous-classe
locale, même principe que audit_2026_08_frozen_decoder_scale_fix.py.

`input_scale` calibrée dès le départ (leçon de A.3.3 : comparer à échelle
égale avec ExtendedSAE, pas à échelle unitaire).

Usage : sbatch slurm/validation/run_audit_soft_frozen_decoder.slurm
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import (
    MODEL_ID, HF_TOKEN, DTYPE, SAVE_DIR, LAYER, LOCAL_MAILS_PATH,
    LOCAL_AUGMENTED_MAILS_PATH, CORPUS_SPLIT_SEED, LOCAL_SAE_ROOT, SAE_SNAPSHOT,
    HOOK_TYPE, RELEASE_ID, EPOCHS_EXTRA, LR_EXTRA,
)
from src.data.preparation import build_email_train_test_corpus
from src.analysis.activations import extract_residual_acts, maxpool_sae_docs
from src.sae import load_gemma_scope_sae
from src.sae.frozen_core import FrozenCoreResidualSAE
from src.sae.sae_shared import load_or_train_extended_sae
from src.sae.judge import odd_one_out_judge

DEVICE = "cuda"
TORCH_DTYPE = torch.bfloat16 if DTYPE == "bf16" else torch.float16
D_EXTRA, K_EXTRA = 1024, 32
SAE_ID = "layer_24_width_16k_l0_medium"
COS_THRESHOLD = 0.8  # Korznikov et al., Soft-Frozen Decoder
RESIDUALS_PATH = os.path.join(SAVE_DIR, "cache", "p1_raw_residuals.pt")
NEW_MODEL_NAME = f"audit_2026_08_soft_frozen_decoder_scale1_d{D_EXTRA}_k{K_EXTRA}"
TOKEN_FRAGMENTS_DIR = os.path.join(SAVE_DIR, "cache", "p1_token_fragments")
ORIGINAL_JUDGE_CACHE = os.path.join(SAVE_DIR, "cache", "p1_judge_labels_extended.json")
OUT_PATH = os.path.join(SAVE_DIR, "cache", "audit_2026_08_soft_frozen_decoder_scale1_results.json")


class SoftFrozenDecoderSAE(FrozenCoreResidualSAE):
    """Décodeur ENTRAÎNABLE mais contraint : après chaque step, chaque
    direction est reprojetée sur le bord du cône de demi-angle
    arccos(COS_THRESHOLD) autour de sa position D'INITIALISATION si elle en
    est sortie -- projection exacte sur la calotte sphérique (pas une
    approximation par interpolation linéaire)."""

    def __init__(self, core_sae, d_extra=1024, k_extra=32):
        super().__init__(core_sae, d_extra, k_extra)
        self.register_buffer("W_dec_extra_init", self.W_dec_extra.data.clone())

    def calibrate_scale(self, residuals_sample: torch.Tensor) -> None:
        sample = residuals_sample[:min(8192, len(residuals_sample))].float()
        with torch.no_grad():
            self.input_scale.copy_(sample.norm(dim=-1).median().to(self.input_scale.dtype))

    @torch.no_grad()
    def normalize_decoder(self):
        """Projection du gradient parallèle (comme la classe de base), puis
        renormalisation, PUIS projection sur la calotte sphérique autour de
        l'init si le cosinus est descendu sous COS_THRESHOLD."""
        if self.W_dec_extra.grad is not None:
            parallel = (self.W_dec_extra.grad * self.W_dec_extra.data).sum(-1, keepdim=True) \
                       * self.W_dec_extra.data
            self.W_dec_extra.grad -= parallel
        w = F.normalize(self.W_dec_extra.data, dim=1)
        w0 = self.W_dec_extra_init  # déjà unitaire (F.normalize à la construction)

        cos = (w * w0).sum(dim=1, keepdim=True)
        out_of_cone = (cos < COS_THRESHOLD).squeeze(-1)
        if out_of_cone.any():
            w_perp = w - cos * w0
            w_perp_hat = F.normalize(w_perp, dim=1)
            sin_t = (1 - COS_THRESHOLD ** 2) ** 0.5
            w_boundary = COS_THRESHOLD * w0 + sin_t * w_perp_hat
            w = torch.where(out_of_cone.unsqueeze(-1), w_boundary, w)
        self.W_dec_extra.data = F.normalize(w, dim=1)
        # W_enc_extra reste libre (non contraint par Korznikov et al. -- seul
        # le décodeur porte la contrainte de proximité à l'init).


def main():
    print("[soft-frozen] Chargement du SAE core...")
    sae_dir = os.path.join(LOCAL_SAE_ROOT, "snapshots", SAE_SNAPSHOT, HOOK_TYPE, SAE_ID)
    core = load_gemma_scope_sae(
        sae_dir=sae_dir, device=DEVICE, release_id=RELEASE_ID, sae_id=f"{HOOK_TYPE}/{SAE_ID}",
    ).to(DEVICE).eval()
    core.requires_grad_(False)

    raw_residuals = torch.load(RESIDUALS_PATH, map_location="cpu", weights_only=True)
    model = SoftFrozenDecoderSAE(core, d_extra=D_EXTRA, k_extra=K_EXTRA).to(DEVICE)
    # input_scale reste à 1.0 (défaut de FrozenCoreResidualSAE.__init__) --
    # variante scale=1, pas de calibrate_scale() ici, cf. docstring.
    print(f"[soft-frozen-scale1] input_scale = {model.input_scale.item():.4f} "
          f"(non calibrée), cos_threshold={COS_THRESHOLD}")

    print(f"[soft-frozen] Entraînement (décodeur contraint au cône, encodeur libre) : "
          f"{EPOCHS_EXTRA} epochs, lr={LR_EXTRA}...")
    model, history = load_or_train_extended_sae(
        model=model, model_name=NEW_MODEL_NAME, acts_train=raw_residuals,
        epochs=EPOCHS_EXTRA, lr=LR_EXTRA, save_dir=SAVE_DIR, device=DEVICE,
    )
    final_val_loss = history.get("val_loss", [None])[-1] if history.get("val_loss") else None
    print(f"[soft-frozen] val_loss final = {final_val_loss}")

    with torch.no_grad():
        w = F.normalize(model.W_dec_extra.data, dim=1)
        cos_final = (w * model.W_dec_extra_init).sum(dim=1)
        print(f"[soft-frozen] cosinus final à l'init : min={cos_final.min():.4f} "
              f"moyenne={cos_final.mean():.4f} (contrainte >= {COS_THRESHOLD})")

    with open(ORIGINAL_JUDGE_CACHE, encoding="utf-8") as f:
        original_judge_data = json.load(f)
    feature_indices = [int(k) for k in original_judge_data.keys()]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True, local_files_only=True)
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=TORCH_DTYPE, device_map=DEVICE,
        low_cpu_mem_usage=True, token=HF_TOKEN, trust_remote_code=True, local_files_only=True,
    ).eval()

    train_texts, _, _, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED,
    )
    n_train = len(train_texts)
    print(f"[soft-frozen] Extraction + encodage des {n_train} mails train...")
    stream = extract_residual_acts(train_texts, llm, tokenizer, layer=LAYER, device=DEVICE)

    def encode_fn(x):
        with torch.no_grad():
            return model.encode(x.to(DEVICE))

    train_doc_acts = maxpool_sae_docs(
        act_stream=stream, encode_fn=encode_fn, n_docs=n_train,
        d_sae=core.cfg.d_sae + D_EXTRA, device=DEVICE,
    )

    print("[soft-frozen] Jugement odd-one-out...")
    results = odd_one_out_judge(
        llm, tokenizer, feature_indices, TOKEN_FRAGMENTS_DIR, train_doc_acts, offset=0, n_pos=9,
    )
    n = len(results)
    n_interp = sum(1 for v in results.values() if v.get("interp_score") == 1)
    rate = n_interp / n

    summary = {
        "n_tested": n, "n_interp": n_interp, "rate": rate,
        "cos_threshold": COS_THRESHOLD,
        "cos_final_min": float(cos_final.min()), "cos_final_mean": float(cos_final.mean()),
        "input_scale": float(model.input_scale.item()), "final_val_loss": final_val_loss,
        "reference_trained_pca_calibrated": 0.4533333333333333,
        "reference_frozen_random_scale1": 0.293,
        "reference_frozen_random_scale_calibrated": 0.16,
        "reference_trained_random_init_scale1": 0.30666666666666664,
        "reference_soft_frozen_scale_calibrated": 0.26,
    }
    print("\n" + "=" * 70)
    print(" RÉSUMÉ — SOFT-FROZEN DECODER (Korznikov et al.), cos>=0.8, échelle=1 (non calibrée)")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_feature": results}, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Écrit : {OUT_PATH}")


if __name__ == "__main__":
    main()
