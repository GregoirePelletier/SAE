"""
scripts/b2_stratified_selection_rejudge.py -- Sélection stratifiée par
fréquence (`feature_selection_stratified_by_frequency`, AUDIT_SAE_2026-08.md
item B.2) vs sélection par magnitude (défaut, référence 45,3%/68/150) sur les
MÊMES caches d'activations et le MÊME juge -- aucun réentraînement, aucune
nouvelle extraction, seul l'échantillon de 150 features change.

Usage :
    SAVE_DIR=./results_v10_emails_main/ PYTHONPATH=. \
      .venv/bin/python scripts/b2_stratified_selection_rejudge.py
"""
from __future__ import annotations

import json
import os
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "sae"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import HF_TOKEN, DTYPE, SAVE_DIR, MODEL_ID, CORPUS_SPLIT_SEED, D_EXTRA, N_FEATURES_TO_LABEL, LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH
from src.data.preparation import build_email_train_test_corpus
from src.sae.judge import feature_selection_stratified_by_frequency, odd_one_out_judge

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if DTYPE == "bf16" else torch.float16
SEED = int(os.environ.get("SEED", "42"))

CACHE_DIR = os.path.join(SAVE_DIR, "cache")
REF_JUDGE_CACHE = os.path.join(CACHE_DIR, "p1_judge_labels_extended.json")
TOKEN_FRAGMENTS_DIR = os.path.join(CACHE_DIR, "p1_token_fragments")
OUT_PATH = os.path.join(CACHE_DIR, "b2_stratified_selection_rejudge.json")


def main() -> None:
    random.seed(SEED)
    with open(REF_JUDGE_CACHE, encoding="utf-8") as f:
        reference = json.load(f)
    ref_indices = [int(k) for k in reference.keys()]
    d_core, d_total = min(ref_indices), max(ref_indices) + 1  # plage extension observée dans le cache de référence
    print(f"[b2-rejudge] plage extension [{d_core}, {d_total}) déduite du cache de référence")

    train_texts, _, _, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED,
    )
    n_train = len(train_texts)
    print(f"[b2-rejudge] n_train={n_train}")

    top_ext_indices = feature_selection_stratified_by_frequency(
        TOKEN_FRAGMENTS_DIR, list(range(n_train)), d_total, N_FEATURES_TO_LABEL,
        lo=d_core, hi=d_total, seed=SEED,
    )
    print(f"[b2-rejudge] {len(top_ext_indices)} features sélectionnées par fréquence "
          f"(vs {len(ref_indices)} par magnitude dans la référence)")
    overlap = set(top_ext_indices) & set(ref_indices)
    print(f"[b2-rejudge] chevauchement avec l'échantillon de référence : {len(overlap)}/{len(ref_indices)}")

    all_doc_acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts_ext_d1024.pt")
    if not os.path.exists(all_doc_acts_path):
        all_doc_acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts.pt")
    all_doc_sae_acts = torch.load(all_doc_acts_path, map_location="cpu", weights_only=True)
    train_acts = all_doc_sae_acts[:n_train]

    print(f"[b2-rejudge] Chargement du juge {MODEL_ID} (même modèle que la référence)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=TORCH_DTYPE, device_map=DEVICE,
        low_cpu_mem_usage=True, token=HF_TOKEN, trust_remote_code=True, local_files_only=True,
    ).eval()

    results = odd_one_out_judge(
        model, tokenizer, top_ext_indices, TOKEN_FRAGMENTS_DIR, train_acts, offset=0, n_pos=9,
    )

    n = len(results)
    n_interp = sum(1 for v in results.values() if v.get("interp_score") == 1)
    n_dead = sum(1 for v in results.values() if v.get("label") == "dead_feature")
    rate = n_interp / n if n else float("nan")
    print(f"\n[b2-rejudge] Stratifié par fréquence : {n_interp}/{n} = {rate:.4f} interprétable ({n_dead} dead/insuffisant)")

    n_ref_interp = sum(1 for v in reference.values() if v.get("interp_score") == 1)
    print(f"[b2-rejudge] Référence (magnitude) : {n_ref_interp}/{len(reference)} = {n_ref_interp/len(reference):.4f}")

    json.dump(
        {"selected_features": top_ext_indices, "overlap_with_magnitude_selection": len(overlap), "results": results},
        open(OUT_PATH, "w"), indent=2, ensure_ascii=False,
    )
    print(f"[b2-rejudge] Sauvé : {OUT_PATH}")


if __name__ == "__main__":
    main()
