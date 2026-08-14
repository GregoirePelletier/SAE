"""
scripts/audit_2026_08_bf16_fp32_diagnostic.py — Diagnostic B.21 de
`docs/AUDIT_2026-08.md` : le résidu `x - core_out` de `FrozenCoreResidualSAE`
est-il affecté par une annulation catastrophique en bf16 ?

Compare, sur un petit échantillon de contrôle (mails originaux déjà utilisés
par les autres scripts d'audit), l'activation résiduelle `x` extraite via un
forward LM en bf16 (production actuelle) à celle extraite via un forward LM
en fp32 (le seul point du pipeline où `x` existe en pleine précision), sur
les MÊMES tokens. Stratifie par norme de `x` (2-4σ vs <2σ, `norm_outlier_mask`
désactivé ici pour observer toute la plage plutôt que la couper).

Ne modifie aucun cache/checkpoint existant — script d'audit autonome, écrit en
dur sur un échantillon indépendant.

Coût : charge Gemma-3-12b-it deux fois (bf16 puis fp32) séquentiellement,
libération mémoire explicite entre les deux — fp32 (~48 Go de poids) exige un
GPU à VRAM large (H100 80 Go), d'où la soumission dédiée sur cette partition.

Usage : sbatch slurm/validation/run_audit_bf16_fp32_diagnostic.slurm
"""
from __future__ import annotations

import gc
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import (
    MODEL_ID, HF_TOKEN, LAYER, LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH,
    CORPUS_SPLIT_SEED, LOCAL_SAE_ROOT, SAE_SNAPSHOT, HOOK_TYPE, RELEASE_ID,
)
from src.data.preparation import build_email_train_test_corpus
from src.analysis.activations import valid_token_mask
from src.sae import load_gemma_scope_sae

DEVICE = "cuda"
N_DOCS = 40
SEED = 42
OUT_PATH = os.path.join("docs", "audit_bf16_fp32_diagnostic_results.json")
SAE_ID = "layer_24_width_16k_l0_medium"  # pinné : cf. steering_fidelity_test.py,
# ne pas dépendre du défaut ambiant de src.config (65k) -- doit matcher LAYER=24.


def get_sample_texts() -> list[str]:
    train_texts, train_labels, _, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED,
    )
    originals = [t for t, l in zip(train_texts, train_labels) if l == "original"]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(originals), size=min(N_DOCS, len(originals)), replace=False)
    return [originals[i] for i in idx]


@torch.no_grad()
def extract_layer_x(texts: list[str], tokenizer, dtype: torch.dtype, batch_size: int = 4):
    """Forward LM en dtype donné, extrait hidden_states[LAYER] token-level,
    masqué (special tokens + 1er token de contenu), SANS norm_outlier_mask
    (on veut observer toute la plage de normes, pas la couper)."""
    print(f"  [diag] Chargement {MODEL_ID} en {dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, device_map=DEVICE,
        low_cpu_mem_usage=True, token=HF_TOKEN, trust_remote_code=True, local_files_only=True,
    ).eval()

    all_x, all_doc_ids, all_tok_pos = [], [], []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = enc["input_ids"].to(DEVICE)
        attn = enc["attention_mask"].to(DEVICE)
        out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, logits_to_keep=1)
        resid = out.hidden_states[LAYER].clone()
        del out
        mask = valid_token_mask(input_ids, attn, tokenizer, skip_first_content_token=True)
        b_idx, t_idx = mask.nonzero(as_tuple=True)
        all_x.append(resid[b_idx, t_idx].float().cpu())
        all_doc_ids.append((b_idx + start).cpu())
        all_tok_pos.append(t_idx.cpu())

    x = torch.cat(all_x, dim=0)
    doc_ids = torch.cat(all_doc_ids, dim=0)
    tok_pos = torch.cat(all_tok_pos, dim=0)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return x, doc_ids, tok_pos


def main():
    texts = get_sample_texts()
    print(f"[diag] {len(texts)} mails originaux échantillonnés (seed={SEED}).")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True, local_files_only=True)

    x_bf16, doc_bf16, pos_bf16 = extract_layer_x(texts, tokenizer, torch.bfloat16)
    print(f"[diag] bf16 : {x_bf16.shape[0]} tokens valides extraits.")

    x_fp32, doc_fp32, pos_fp32 = extract_layer_x(texts, tokenizer, torch.float32)
    print(f"[diag] fp32 : {x_fp32.shape[0]} tokens valides extraits.")

    # Alignement par (doc_id, tok_pos) -- le masquage (special tokens, 1er
    # token de contenu) est déterministe et indépendant du dtype (dépend de
    # input_ids/attention_mask, identiques dans les deux passes), donc les
    # deux ensembles de (doc_id, tok_pos) doivent coïncider exactement.
    key_bf16 = [(int(d), int(p)) for d, p in zip(doc_bf16, pos_bf16)]
    key_fp32 = [(int(d), int(p)) for d, p in zip(doc_fp32, pos_fp32)]
    assert key_bf16 == key_fp32, (
        f"Désalignement token-à-token entre les deux passes ({len(key_bf16)} vs "
        f"{len(key_fp32)}) -- le masquage a dû diverger entre bf16/fp32, à diagnostiquer "
        f"avant de faire confiance à la suite (ne devrait pas arriver : le masque ne "
        f"dépend que de input_ids/attention_mask, identiques dans les deux passes)."
    )
    n = len(key_bf16)
    print(f"[diag] {n} tokens alignés entre les deux passes.")

    # Charge le SAE core (bf16, comme en production) pour calculer le résidu
    # tel qu'utilisé par FrozenCoreResidualSAE.
    sae_dir = os.path.join(LOCAL_SAE_ROOT, "snapshots", SAE_SNAPSHOT, HOOK_TYPE, SAE_ID)
    core = load_gemma_scope_sae(
        sae_dir=sae_dir, device=DEVICE, release_id=RELEASE_ID, sae_id=f"{HOOK_TYPE}/{SAE_ID}",
    ).to(DEVICE).eval()
    core.requires_grad_(False)

    with torch.no_grad():
        core_out_bf16 = core.decode(core.encode(x_bf16.to(DEVICE).to(torch.bfloat16))).float().cpu()
        core_out_fp32_input = core.decode(core.encode(x_fp32.to(DEVICE).to(torch.bfloat16))).float().cpu()

    # Production actuelle : soustraction EN bf16 puis élargie (frozen_core.py::forward).
    residual_bf16_asis = (x_bf16.to(torch.bfloat16) - core_out_bf16.to(torch.bfloat16)).float()
    # Correctif proposé : x extrait via un forward LM fp32, soustraction en fp32.
    residual_fp32_fix = x_fp32.float() - core_out_fp32_input.float()

    x_bf16_norms = x_bf16.norm(dim=-1)
    mu, sd = x_bf16_norms.mean().item(), x_bf16_norms.std().item()
    strata = {
        "sous_2sigma": (x_bf16_norms < mu + 2 * sd),
        "entre_2_4sigma": (x_bf16_norms >= mu + 2 * sd) & (x_bf16_norms < mu + 4 * sd),
        "au_dela_4sigma": (x_bf16_norms >= mu + 4 * sd),
    }

    delta_x = (x_bf16 - x_fp32).norm(dim=-1)          # divergence bf16 vs fp32 sur x lui-même
    delta_residual = (residual_bf16_asis - residual_fp32_fix).norm(dim=-1)
    residual_norm_asis = residual_bf16_asis.norm(dim=-1)

    results = {"n_tokens": n, "mu_norm_x_bf16": mu, "sd_norm_x_bf16": sd, "strata": {}}
    print("\n" + "=" * 78)
    print(" B.21 — Divergence bf16 vs fp32 du résidu, par strate de norme de x")
    print("=" * 78)
    for name, m in strata.items():
        n_s = int(m.sum())
        if n_s == 0:
            print(f"  {name}: 0 tokens, ignoré.")
            continue
        dx = delta_x[m]
        dr = delta_residual[m]
        rn = residual_norm_asis[m]
        rel_dx = (dx / (x_bf16[m].norm(dim=-1) + 1e-8))
        rel_dr_vs_residual = (dr / (rn + 1e-8))          # le ratio qui compte : erreur / signal
        results["strata"][name] = {
            "n": n_s,
            "mean_abs_delta_x": float(dx.mean()),
            "mean_rel_delta_x": float(rel_dx.mean()),
            "mean_residual_norm_asis": float(rn.mean()),
            "mean_abs_delta_residual": float(dr.mean()),
            "mean_rel_delta_residual_vs_residual_norm": float(rel_dr_vs_residual.mean()),
            "median_rel_delta_residual_vs_residual_norm": float(rel_dr_vs_residual.median()),
            "frac_delta_residual_gt_half_residual_norm": float((dr > 0.5 * rn).float().mean()),
        }
        print(f"  {name} (n={n_s}): |Δx|/|x|={rel_dx.mean():.4f} | "
              f"|résidu| moyen={rn.mean():.2f} | |Δrésidu|={dr.mean():.4f} | "
              f"|Δrésidu|/|résidu|={rel_dr_vs_residual.mean():.4f} "
              f"(médiane={rel_dr_vs_residual.median():.4f}) | "
              f"frac(|Δrésidu|>0.5|résidu|)={(dr > 0.5*rn).float().mean():.3f}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Écrit : {OUT_PATH}")


if __name__ == "__main__":
    main()
