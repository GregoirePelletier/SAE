"""
scripts/steering_fidelity_test_r0.py — Fidélité du steering (`steer_and_decode`),
sous R0. RESULTS_TESTS.md §24/§68/§69 mesurent ce protocole exclusivement sur
le checkpoint historique `p1_frozen_core_d1024_k32.pt` (K_EXTRA=32, layer 24,
`results_v10_emails_main`), jamais sur R0 (K_EXTRA=5, layer 31). Ce script
reprend `scripts/steering_fidelity_test.py` à l'identique (le correctif
`random_state=SEED` de §69 est déjà en place, `INTENT_KEYWORDS_FR` déjà
corrigé en production) et ne change que les trois constantes qui pointaient
sur le checkpoint historique (D_EXTRA/K_EXTRA/SAE_ID) et SAVE_DIR.

Coût : charge le SAE core GemmaScope (petit, CPU possible) + le checkpoint
p1_frozen_core_d1024_k5.pt déjà entraîné (R0) -- pas de LLM Gemma-3-12B, pas
de ré-extraction (réutilise p1_all_doc_acts_ext_d1024.pt déjà en cache pour R0).

Usage : PYTHONPATH=. .venv/bin/python scripts/steering_fidelity_test_r0.py
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from src.config import (
    LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, SAVE_DIR,
    RELEASE_ID, HOOK_TYPE, LOCAL_SAE_ROOT, SAE_SNAPSHOT, CORPUS_SPLIT_SEED,
)
from src.data.dataset import load_mails_tsv
from src.data.preparation import build_email_train_test_corpus
from src.sae import load_gemma_scope_sae
from src.sae.frozen_core import SAEBoostResidualSAE
from src.sae.sae_shared import steer_and_decode, load_all_doc_acts

CACHE_DIR = os.path.join(SAVE_DIR, "cache")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
TOP_K = 10
N_DOCS_SAMPLE = 200
D_EXTRA, K_EXTRA = 1024, 5   # R0 (au lieu de 1024/32 historique)
SAE_ID = "layer_31_width_16k_l0_medium"  # R0 (au lieu de layer_24 historique)
rng_np = np.random.default_rng(SEED)


def replicate_load_and_clean_emails_with_index(tsv_path: str):
    df = load_mails_tsv(tsv_path).rename(columns={"text": "document"})
    kept_row_indices = []
    for row_idx, row in df.iterrows():
        if "document" not in row or pd.isna(row["document"]):
            continue
        raw_text = str(row["document"])
        clean_text = re.sub(r'^\s*(?:Objet|Subject)\s*:\s*[^\n]+\n*', '', raw_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'\[\s*\{\s*"start".*?\}\s*\]', '', clean_text, flags=re.DOTALL).strip()
        if clean_text:
            kept_row_indices.append(row_idx)
    return kept_row_indices, df


def load_real_email_acts_and_intents():
    kept_row_indices, df_full = replicate_load_and_clean_emails_with_index(LOCAL_MAILS_PATH)
    n_real = len(kept_row_indices)
    rng = np.random.default_rng(SEED)
    test_mask = rng.random(n_real) < float(os.environ.get("EMAIL_TEST_SPLIT", "0.05"))
    train_positions = [i for i in range(n_real) if not test_mask[i]]
    k_train_original = len(train_positions)

    train_texts, train_labels, _, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED,
    )
    n_original_train = sum(1 for l in train_labels if l == "original")
    assert n_original_train == k_train_original, "Incohérence de correspondance -- cf. intent_urgency_probe.py"

    acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts_ext_d1024.pt")
    if not os.path.exists(acts_path):
        acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts.pt")
    all_doc_acts = load_all_doc_acts(acts_path)
    original_train_acts = all_doc_acts[:k_train_original].float().numpy()

    row_indices_for_train_original = [kept_row_indices[i] for i in train_positions]
    intent_cols = [c for c in df_full.columns if c.startswith("intent_")]
    labels_df = df_full.loc[row_indices_for_train_original, intent_cols].reset_index(drop=True)
    return original_train_acts, labels_df


def load_frozen_core_sae() -> SAEBoostResidualSAE:
    sae_dir = os.path.join(LOCAL_SAE_ROOT, "snapshots", SAE_SNAPSHOT, HOOK_TYPE, SAE_ID)
    core_sae = load_gemma_scope_sae(
        sae_dir=sae_dir, device=DEVICE, release_id=RELEASE_ID, sae_id=f"{HOOK_TYPE}/{SAE_ID}",
    )
    ext_sae = SAEBoostResidualSAE(core_sae, d_extra=D_EXTRA, k_extra=K_EXTRA).to(DEVICE)
    ckpt_path = os.path.join(SAVE_DIR, f"p1_frozen_core_d{D_EXTRA}_k{K_EXTRA}.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    load_result = ext_sae.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"[steering-fidelity-r0] load_state_dict : missing={list(load_result.missing_keys)} "
          f"unexpected={list(load_result.unexpected_keys)}")
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(f"Chargement incomplet de {ckpt_path}")
    ext_sae.eval()
    return ext_sae


def probe_before_after(clf: LogisticRegression, x_before: np.ndarray, x_after: np.ndarray) -> tuple[float, float]:
    p_before = clf.predict_proba(x_before.reshape(1, -1))[0, 1]
    p_after = clf.predict_proba(x_after.reshape(1, -1))[0, 1]
    return float(p_before - p_after), float(p_before)


def main():
    print("[steering-fidelity-r0] Chargement des activations et labels d'intention...")
    acts, labels_df = load_real_email_acts_and_intents()
    print(f"[steering-fidelity-r0] {acts.shape[0]} mails originaux, {acts.shape[1]} dims SAE.")

    print("[steering-fidelity-r0] Chargement du FrozenCoreResidualSAE R0 (core GemmaScope + extension entraînée K5/layer31)...")
    ext_sae = load_frozen_core_sae()

    results = {}
    for col in labels_df.columns:
        y = labels_df[col].to_numpy().astype(int)
        n_pos = int(y.sum())
        if n_pos < 30 or n_pos > len(y) - 30:
            continue

        clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear", random_state=SEED)
        clf.fit(acts, y)
        coef = clf.coef_[0]
        probs = clf.predict_proba(acts)[:, 1]
        candidate_idx = np.where((y == 1) & (probs > 0.7))[0]
        if len(candidate_idx) == 0:
            continue
        sample_idx = rng_np.choice(candidate_idx, size=min(N_DOCS_SAMPLE, len(candidate_idx)), replace=False)

        drops_inplace, drops_roundtrip, residual_leak = [], [], []
        for i in sample_idx:
            x = acts[i]
            active = np.where(x > 1e-6)[0]
            if len(active) < TOP_K * 2:
                continue
            contributions = coef[active] * x[active]
            order = np.argsort(contributions)[::-1]
            top_feats = active[order[:TOP_K]]

            x_inplace = x.copy()
            x_inplace[top_feats] = 0.0
            d_inplace, p0 = probe_before_after(clf, x, x_inplace)

            with torch.no_grad():
                x_t = torch.from_numpy(x).unsqueeze(0).to(DEVICE)
                amplifications = {int(f): 0.0 for f in top_feats}
                decoded = steer_and_decode(x_t, amplifications, ext_sae)
                reencoded = ext_sae.encode(decoded).squeeze(0).float().cpu().numpy()
            d_roundtrip, _ = probe_before_after(clf, x, reencoded)
            leak = float(np.mean(reencoded[top_feats]) / max(np.mean(x[top_feats]), 1e-6))

            drops_inplace.append(d_inplace)
            drops_roundtrip.append(d_roundtrip)
            residual_leak.append(leak)

        if not drops_inplace:
            continue

        results[col] = {
            "n_docs_tested": len(drops_inplace),
            "mean_drop_inplace": float(np.mean(drops_inplace)),
            "mean_drop_roundtrip_steer_and_decode": float(np.mean(drops_roundtrip)),
            "ratio_roundtrip_vs_inplace": float(np.mean(drops_roundtrip) / max(np.mean(drops_inplace), 1e-6)),
            "mean_residual_leak_fraction": float(np.mean(residual_leak)),
        }
        print(f"[steering-fidelity-r0] {col} (n={len(drops_inplace)}): "
              f"chute en place={np.mean(drops_inplace):.4f} | "
              f"chute steer_and_decode={np.mean(drops_roundtrip):.4f} | "
              f"ratio={np.mean(drops_roundtrip)/max(np.mean(drops_inplace),1e-6):.2f}x | "
              f"fuite résiduelle={np.mean(residual_leak):.3f}")

    out_path = os.path.join(CACHE_DIR, "steering_fidelity_r0_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Écrit : {out_path}")


if __name__ == "__main__":
    main()
