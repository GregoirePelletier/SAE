"""
scripts/saelens_numeric_comparison.py — Comparaison CHIFFRÉE (pas seulement de
formule, cf. docs/references.md) entre notre `compute_metrics` (src/analysis/metrics.py)
et les formules de variance expliquée maintenues par `sae_lens.evals` elles-mêmes
("legacy" et "corrigée"), calculées sur le MÊME SAE natif sae-lens (chargé via
`load_gemma_scope_sae`, qui appelle déjà `sae_lens.SAE.load_from_disk`/`from_pretrained`
en interne -- ce n'est pas un SAE réimplémenté) et les MÊMES activations (déjà en
cache, tokens réels d'emails, `p1_eval_raw_tokens.pt`).

Ne nécessite PAS de faire passer le chargement du modèle par
`transformer_lens.HookedTransformer` + `sae_lens.ActivationsStore` (l'API attendue par
`sae_lens.evals.run_evals`/`get_sparsity_and_variance_metrics`, qui re-calcule les
activations elle-même) : on réplique directement leurs formules de variance expliquée
sur des activations déjà extraites, ce qui suffit à répondre à la question "nos deux
formules sont-elles numériquement cohérentes sur les mêmes données ?". Tourne sur CPU
(SAE de taille modeste, 4096 tokens) -- aucun job GPU nécessaire.

Usage :
    PYTHONPATH=. .venv/bin/python scripts/saelens_numeric_comparison.py
"""
from __future__ import annotations

import os

import torch

from src.sae.gemma_scope_loader import load_gemma_scope_sae
from src.config import LOCAL_SAE_ROOT, SAE_SNAPSHOT, HOOK_TYPE, SAE_ID, RELEASE_ID

DEVICE = "cpu"
EVAL_TOKENS_PATH = "./results_v10_emails_main/cache/p1_eval_raw_tokens.pt"


def sae_lens_explained_variance_legacy(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Réplique sae_lens.evals.get_sparsity_and_variance_metrics : moyenne PAR TOKEN
    de (1 - résidu²/variance_batch), variance centrée par dimension sur la moyenne du
    batch en cours (une seule "batch" ici = tout le tenseur d'éval)."""
    resid_sum_of_squares = (x - x_hat).pow(2).sum(dim=-1)
    batched_variance_sum = (x - x.mean(dim=0, keepdim=True)).pow(2).sum(dim=-1)
    explained_variance_legacy = 1 - resid_sum_of_squares / batched_variance_sum
    return explained_variance_legacy.mean().item()


def sae_lens_explained_variance_corrected(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Réplique la formule "corrigée" de sae_lens.evals (agrégation globale avant le
    ratio, pas une moyenne de ratios par token). Dans leur code, `mean_act_per_dimension`
    est d'abord un vecteur [d_model] par batch (`.mean(dim=0)`), puis `torch.cat(...).
    mean(dim=0)` sur la liste des batches l'effondre en un SCALAIRE (concatène tous les
    batches sur le même axe puis moyenne globale) -- reproduit ici directement avec un
    seul "batch" (tout `x`), d'où le double `.mean()` équivalent à `x.pow(2).mean()`.
    """
    mean_sum_of_squares = x.pow(2).sum(dim=-1).mean(dim=0)          # scalaire
    mean_act_per_dimension = x.pow(2).mean(dim=0).mean()            # scalaire (cf. docstring)
    total_variance = mean_sum_of_squares - mean_act_per_dimension ** 2
    resid_sum_of_squares = (x - x_hat).pow(2).sum(dim=-1)
    residual_variance = resid_sum_of_squares.mean(dim=0)
    return (1 - residual_variance / total_variance).item()


def our_compute_metrics_fve(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Réplique EXACTEMENT src/analysis/metrics.py::compute_metrics (partie FVE),
    sans repasser par le forward complet du SAE (déjà fait une fois, réutilisé ici)."""
    mse = torch.mean((x - x_hat).pow(2))
    variance = torch.mean((x - x.mean(dim=0, keepdim=True)).pow(2)) + 1e-8
    nmse = mse / variance
    return (1.0 - nmse).item()


def main():
    print("[compare] Chargement du SAE GemmaScope natif sae-lens (CPU)...")
    sae_dir = os.path.join(LOCAL_SAE_ROOT, "snapshots", SAE_SNAPSHOT, HOOK_TYPE, SAE_ID)
    sae = load_gemma_scope_sae(
        sae_dir=sae_dir, device=DEVICE, release_id=RELEASE_ID, sae_id=f"{HOOK_TYPE}/{SAE_ID}",
    )
    sae = sae.to(DEVICE).eval()
    print(f"[compare] SAE chargé : d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}, "
          f"architecture={sae.cfg.architecture}")

    print(f"[compare] Chargement des activations réelles (emails, cache existant) : {EVAL_TOKENS_PATH}")
    x = torch.load(EVAL_TOKENS_PATH, map_location=DEVICE, weights_only=True).float()
    print(f"[compare] {x.shape[0]} tokens réels, d_in={x.shape[1]}")

    with torch.no_grad():
        x_bf16 = x.to(sae.W_enc.dtype)
        codes = sae.encode(x_bf16)
        x_hat = sae.decode(codes).float()

    fve_ours = our_compute_metrics_fve(x, x_hat)
    ev_legacy = sae_lens_explained_variance_legacy(x, x_hat)
    ev_corrected = sae_lens_explained_variance_corrected(x, x_hat)
    l0_ours = (codes.abs() > 1e-6).float().sum(dim=-1).mean().item()

    print("\n" + "=" * 70)
    print(" COMPARAISON CHIFFRÉE — même SAE natif sae-lens, mêmes activations réelles")
    print("=" * 70)
    print(f"  FVE (notre compute_metrics)                         : {fve_ours:.6f}")
    print(f"  explained_variance_legacy (formule sae_lens.evals)  : {ev_legacy:.6f}")
    print(f"  explained_variance (formule 'corrigée' sae_lens)    : {ev_corrected:.6f}")
    print(f"  Écart notre formule vs legacy sae_lens               : {abs(fve_ours - ev_legacy):.6f}")
    print(f"  Écart notre formule vs corrigée sae_lens             : {abs(fve_ours - ev_corrected):.6f}")
    print(f"  L0 (notre formule, > seuil 1e-6)                     : {l0_ours:.2f}")

    import json
    out = {
        "n_tokens": x.shape[0], "d_in": x.shape[1], "d_sae": sae.cfg.d_sae,
        "fve_ours": fve_ours, "explained_variance_legacy_saelens": ev_legacy,
        "explained_variance_corrected_saelens": ev_corrected, "l0_ours": l0_ours,
    }
    out_path = "./results_v10_emails_main/cache/saelens_numeric_comparison.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[+] Écrit : {out_path}")


if __name__ == "__main__":
    main()
