"""
scripts/audit_2026_08_bf16_fp32_diagnostic_v2.py — Suite de
audit_2026_08_bf16_fp32_diagnostic.py (job 43256, résultat déjà intégré,
RESULTS_TESTS.md/AUDIT_2026-08.md B.21). Round 3 de l'audit (§1) a repéré
que la strate >4σ était VIDE dans le premier passage -- pas par hasard, mais
parce que `skip_first_content_token=True` exclut précisément le 1er token de
contenu, l'un des principaux sites documentés d'activations massives
(attention sink, cf. docstring `activations.py`). Le diagnostic n'avait donc
jamais pu mesurer la population qu'il visait.

Corrige en NE PAS excluant le 1er token de contenu ici (uniquement les
tokens spéciaux BOS/EOS/PAD, qui n'ont de toute façon pas de texte à
comparer) -- cible délibérément les positions où l'annulation catastrophique
serait la plus sévère, plutôt qu'un tirage aléatoire de documents complets.
Ajoute un intervalle de confiance (bootstrap) sur les moyennes par strate,
répondant aussi à round 3 §2.

Usage : sbatch slurm/validation/run_audit_bf16_fp32_diagnostic_v2.slurm
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
from src.sae import load_gemma_scope_sae

DEVICE = "cuda"
N_DOCS = 150  # échantillon élargi (round 3 : "augmenter drastiquement")
SEED = 42
OUT_PATH = os.path.join("docs", "audit_bf16_fp32_diagnostic_v2_results.json")
SAE_ID = "layer_24_width_16k_l0_medium"
N_BOOT = 2000


def get_sample_texts() -> list[str]:
    train_texts, train_labels, _, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED,
    )
    originals = [t for t, l in zip(train_texts, train_labels) if l == "original"]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(originals), size=min(N_DOCS, len(originals)), replace=False)
    return [originals[i] for i in idx]


def valid_token_mask_include_first(input_ids, attention_mask, tokenizer):
    """Comme activations.py::valid_token_mask mais SANS exclure le 1er token
    de contenu -- on veut délibérément l'inclure ici (cible du diagnostic)."""
    mask = attention_mask.bool()
    special = torch.zeros_like(mask)
    for tid in tokenizer.all_special_ids:
        special |= input_ids == tid
    mask &= ~special
    return mask


@torch.no_grad()
def extract_layer_x(texts: list[str], tokenizer, dtype: torch.dtype, batch_size: int = 4):
    print(f"  [diag-v2] Chargement {MODEL_ID} en {dtype}...")
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
        mask = valid_token_mask_include_first(input_ids, attn, tokenizer)
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


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    if len(values) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)
    ])
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def main():
    texts = get_sample_texts()
    print(f"[diag-v2] {len(texts)} mails originaux échantillonnés (seed={SEED}, "
          f"élargi vs 40 dans le premier passage).")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True, local_files_only=True)

    x_bf16, doc_bf16, pos_bf16 = extract_layer_x(texts, tokenizer, torch.bfloat16)
    print(f"[diag-v2] bf16 : {x_bf16.shape[0]} tokens valides extraits (1er token de contenu INCLUS).")

    x_fp32, doc_fp32, pos_fp32 = extract_layer_x(texts, tokenizer, torch.float32)
    print(f"[diag-v2] fp32 : {x_fp32.shape[0]} tokens valides extraits.")

    key_bf16 = [(int(d), int(p)) for d, p in zip(doc_bf16, pos_bf16)]
    key_fp32 = [(int(d), int(p)) for d, p in zip(doc_fp32, pos_fp32)]
    assert key_bf16 == key_fp32, "Désalignement token-à-token entre les deux passes."
    n = len(key_bf16)
    print(f"[diag-v2] {n} tokens alignés.")

    sae_dir = os.path.join(LOCAL_SAE_ROOT, "snapshots", SAE_SNAPSHOT, HOOK_TYPE, SAE_ID)
    core = load_gemma_scope_sae(
        sae_dir=sae_dir, device=DEVICE, release_id=RELEASE_ID, sae_id=f"{HOOK_TYPE}/{SAE_ID}",
    ).to(DEVICE).eval()
    core.requires_grad_(False)

    with torch.no_grad():
        core_out_bf16 = core.decode(core.encode(x_bf16.to(DEVICE).to(torch.bfloat16))).float().cpu()
        core_out_fp32_input = core.decode(core.encode(x_fp32.to(DEVICE).to(torch.bfloat16))).float().cpu()

    residual_bf16_asis = (x_bf16.to(torch.bfloat16) - core_out_bf16.to(torch.bfloat16)).float()
    residual_fp32_fix = x_fp32.float() - core_out_fp32_input.float()

    x_bf16_norms = x_bf16.norm(dim=-1)
    mu, sd = x_bf16_norms.mean().item(), x_bf16_norms.std().item()
    strata = {
        "sous_2sigma": (x_bf16_norms < mu + 2 * sd),
        "entre_2_4sigma": (x_bf16_norms >= mu + 2 * sd) & (x_bf16_norms < mu + 4 * sd),
        "au_dela_4sigma": (x_bf16_norms >= mu + 4 * sd),
    }

    delta_x = (x_bf16 - x_fp32).norm(dim=-1)
    delta_residual = (residual_bf16_asis - residual_fp32_fix).norm(dim=-1)
    residual_norm_asis = residual_bf16_asis.norm(dim=-1)

    results = {"n_tokens": n, "mu_norm_x_bf16": mu, "sd_norm_x_bf16": sd,
               "n_docs_sampled": len(texts), "first_content_token_included": True, "strata": {}}
    print("\n" + "=" * 78)
    print(" B.21 v2 — Divergence bf16 vs fp32, 1er token de contenu INCLUS, IC bootstrap")
    print("=" * 78)
    for name, m in strata.items():
        n_s = int(m.sum())
        if n_s == 0:
            print(f"  {name}: 0 tokens -- même après avoir inclus le 1er token de contenu et "
                  f"élargi l'échantillon à {len(texts)} documents. À documenter comme "
                  f"population difficile d'accès plutôt que silencieusement absente.")
            results["strata"][name] = {"n": 0}
            continue
        dx = delta_x[m]
        dr = delta_residual[m]
        rn = residual_norm_asis[m]
        rel_dx = (dx / (x_bf16[m].norm(dim=-1) + 1e-8))
        rel_dr_vs_residual = (dr / (rn + 1e-8))
        rel_dr_np = rel_dr_vs_residual.numpy()
        ci_low, ci_high = bootstrap_ci(rel_dr_np)
        results["strata"][name] = {
            "n": n_s,
            "mean_rel_delta_x": float(rel_dx.mean()),
            "mean_residual_norm_asis": float(rn.mean()),
            "mean_rel_delta_residual_vs_residual_norm": float(rel_dr_vs_residual.mean()),
            "median_rel_delta_residual_vs_residual_norm": float(rel_dr_vs_residual.median()),
            "ci95_bootstrap_mean_rel_delta_residual": [ci_low, ci_high],
            "frac_delta_residual_gt_half_residual_norm": float((dr > 0.5 * rn).float().mean()),
        }
        print(f"  {name} (n={n_s}): |Δx|/|x|={rel_dx.mean():.4f} | "
              f"|Δrésidu|/|résidu|={rel_dr_vs_residual.mean():.4f} "
              f"(IC95% bootstrap=[{ci_low:.4f}, {ci_high:.4f}]) | "
              f"frac>0.5={(dr > 0.5*rn).float().mean():.3f}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Écrit : {OUT_PATH}")


if __name__ == "__main__":
    main()
