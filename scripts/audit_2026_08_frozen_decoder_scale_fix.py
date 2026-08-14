"""
scripts/audit_2026_08_frozen_decoder_scale_fix.py — A.3 point 3 de
`docs/AUDIT_2026-08.md` (confirmé par lecture statique, ce script retrain pour
mesurer l'effet) : `FrozenDecoderExtendedSAE` (sanity check Korznikov et al.,
decodeur aleatoire fige) herite `input_scale=1.0` jamais calibre, alors que
`ExtendedSAE` (le SAE "entraine" auquel on le compare, 45,3%) a une echelle
calibree sur la mediane des normes du residu. La comparaison publiee 45,3%
vs 29,3% oppose donc "decodeur entraine + echelle calibree" a "decodeur fige
+ echelle unitaire" -- deux differences simultanees.

Ce script NE MODIFIE PAS `src/sae/frozen_core.py` (le sanity check actuel,
29,3%, est deja un resultat publie, RESULTS_TESTS.md §19) : sous-classe
locale qui calibre UNIQUEMENT `input_scale` (mediane des normes du residu,
meme formule que `ExtendedSAE._init_from_residual_pca`) sans jamais toucher
W_dec_extra/W_enc_extra (le decodeur reste aleatoire fige, intention du
sanity check preservee). Reutilise le reservoir de residus deja en cache
(`p1_raw_residuals.pt`, 500k tokens) -- aucun nouveau forward LM.

Usage : sbatch slurm/validation/run_audit_frozen_decoder_scale_fix.slurm
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F

from src.config import (
    LOCAL_SAE_ROOT, SAE_SNAPSHOT, HOOK_TYPE, RELEASE_ID, SAVE_DIR,
    EPOCHS_EXTRA, LR_EXTRA,
)
from src.sae import load_gemma_scope_sae
from src.sae.frozen_core import FrozenDecoderExtendedSAE
from src.sae.sae_shared import load_or_train_extended_sae

DEVICE = "cuda"
D_EXTRA, K_EXTRA = 1024, 32
SAE_ID = "layer_24_width_16k_l0_medium"
RESIDUALS_PATH = os.path.join(SAVE_DIR, "cache", "p1_raw_residuals.pt")
NEW_MODEL_NAME = f"audit_2026_08_frozen_decoder_scalefix_d{D_EXTRA}_k{K_EXTRA}"
# Nom de checkpoint DEDIE (cf. B.8 de l'audit) -- ne collisionne avec aucun
# checkpoint existant, ne depend d'aucun cache partage.


class ScaleCalibratedFrozenDecoderSAE(FrozenDecoderExtendedSAE):
    """FrozenDecoderExtendedSAE (decodeur aleatoire fige) + input_scale
    calibre sur la mediane des normes du residu (meme formule que
    ExtendedSAE._init_from_residual_pca), SANS toucher au decodeur/encodeur
    (restent aleatoires, intention du sanity check Korznikov et al.
    preservee -- seule l'echelle scalaire est calibree, pas une direction)."""

    def calibrate_scale_only(self, residuals_sample: torch.Tensor) -> None:
        sample = residuals_sample[:min(8192, len(residuals_sample))].float()
        with torch.no_grad():
            self.input_scale.copy_(sample.norm(dim=-1).median().to(self.input_scale.dtype))


def main():
    print("[scale-fix] Chargement du SAE core GemmaScope...")
    sae_dir = os.path.join(LOCAL_SAE_ROOT, "snapshots", SAE_SNAPSHOT, HOOK_TYPE, SAE_ID)
    core = load_gemma_scope_sae(
        sae_dir=sae_dir, device=DEVICE, release_id=RELEASE_ID, sae_id=f"{HOOK_TYPE}/{SAE_ID}",
    ).to(DEVICE).eval()
    core.requires_grad_(False)

    print(f"[scale-fix] Chargement du reservoir de residus : {RESIDUALS_PATH}")
    raw_residuals = torch.load(RESIDUALS_PATH, map_location="cpu", weights_only=True)
    print(f"[scale-fix] Reservoir : {tuple(raw_residuals.shape)}")

    model = ScaleCalibratedFrozenDecoderSAE(core, d_extra=D_EXTRA, k_extra=K_EXTRA).to(DEVICE)

    print("[scale-fix] Calibration de input_scale (mediane des normes du residu x-core_out)...")
    with torch.no_grad():
        sample = raw_residuals[:8192].to(DEVICE).to(next(core.parameters()).dtype)
        core_acts = core.encode(sample)
        core_out = core.decode(core_acts)
        domain_residuals_cpu = (sample - core_out).cpu().float()
    model.calibrate_scale_only(domain_residuals_cpu)
    print(f"[scale-fix] input_scale calibre : {model.input_scale.item():.4f} "
          f"(etait 1.0 par defaut dans FrozenDecoderExtendedSAE non corrige)")

    print(f"[scale-fix] Entrainement (encodeur seulement, decodeur fige) : "
          f"{EPOCHS_EXTRA} epochs, lr={LR_EXTRA}...")
    save_dir = os.path.join(SAVE_DIR)
    model, history = load_or_train_extended_sae(
        model=model, model_name=NEW_MODEL_NAME, acts_train=raw_residuals,
        epochs=EPOCHS_EXTRA, lr=LR_EXTRA, save_dir=save_dir, device=DEVICE,
    )

    final_val_loss = history.get("val_loss", [None])[-1] if history.get("val_loss") else None
    summary = {
        "input_scale_calibrated": float(model.input_scale.item()),
        "d_extra": D_EXTRA, "k_extra": K_EXTRA,
        "epochs": EPOCHS_EXTRA, "lr": LR_EXTRA,
        "final_val_loss": final_val_loss,
        "checkpoint": os.path.join(save_dir, f"{NEW_MODEL_NAME}.pt"),
        "note": "Decodeur/encodeur restent ceux de FrozenDecoderExtendedSAE (aleatoires, "
                "figes) -- seul input_scale differe de la baseline publiee (29,3%, "
                "input_scale=1.0). Rejuger ce checkpoint avec le protocole odd-one-out "
                "standard (meme juge, memes 150 features) pour mesurer si le taux "
                "d'interpretabilite change une fois ce confond retire -- etape separee, "
                "non faite dans ce script (necessite le juge LLM complet, cout different).",
    }
    out_path = os.path.join(save_dir, "cache", "audit_2026_08_frozen_decoder_scalefix_summary.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Ecrit : {out_path}")
    print(f"[+] Checkpoint : {summary['checkpoint']}")


if __name__ == "__main__":
    main()
