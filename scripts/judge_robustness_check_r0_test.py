"""
scripts/judge_robustness_check_r0_test.py — Robustesse au réordonnancement
du protocole odd-one-out, sous R0 (RESULTS_TESTS.md §13.1 réutilisait
`results_v10_emails_main`, sélection par magnitude, et surtout rechargeait
MODEL_ID (Gemma-3-12b-it, l'extracteur) comme juge -- un biais d'auto-préférence
que CLAUDE.md interdit explicitement de réintroduire. Ce script corrige les
trois écarts : SAVE_DIR pointé sur R0 (résultats_v27...), juge = Qwen3.8-27B
(`load_judge_model`, JUDGE_MODEL_ID découplé de MODEL_ID), et `doc_groups=train_groups`
(déduplication par mail parent, comme l'appel officiel dans saev5.py::odd_one_out_judge).

Zéro réextraction : réutilise les activations et fragments déjà en cache
(`p1_all_doc_acts_ext_d1024.pt`, `p1_token_fragments_ext`) et les 300 features
déjà jugées de R0 (`p1_judge_labels_extended.json`). Pour chaque feature,
répète la question odd-one-out N_REPEATS fois avec un ordre de mélange
différent (mêmes exemples), calcule le vote majoritaire.

Usage (SLURM, 1 GPU) :
    SAVE_DIR=./results_v27_ablation_classic_setup_k5_25m_layer31/ \
      PYTHONPATH=. .venv/bin/python scripts/judge_robustness_check_r0_test.py
"""
from __future__ import annotations

import json
import os
import random
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "sae"))

from src.config import SAVE_DIR, CORPUS_SPLIT_SEED, LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH
from src.sae.judge import build_feature_examples_with_control, _apply_chat_and_extract, load_judge_model
from src.data.preparation import build_email_train_test_corpus
from src.sae.sae_shared import load_all_doc_acts
from src.storage.fragment_store import resolve_extension_fragments_dir

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_REPEATS = int(os.environ.get("N_REPEATS", "5"))
SEED = int(os.environ.get("SEED", "42"))

CACHE_DIR = os.path.join(SAVE_DIR, "cache")
JUDGE_CACHE = os.path.join(CACHE_DIR, "p1_judge_labels_extended.json")
TOKEN_FRAGMENTS_DIR = resolve_extension_fragments_dir(CACHE_DIR)
OUT_PATH = os.path.join(CACHE_DIR, "p1_judge_robustness_r0.json")


def repeated_odd_one_out(model, tokenizer, pos_examples, neg_example, n_repeats):
    all_examples = pos_examples + ([neg_example] if neg_example else [])
    neg_position = len(all_examples) - 1 if neg_example else None

    corrects = []
    predicted_positions = []
    for _ in range(n_repeats):
        indices = list(range(len(all_examples)))
        random.shuffle(indices)
        shuffled = [all_examples[i] for i in indices]
        correct_answer = indices.index(neg_position) + 1 if neg_example else None

        examples_text = "\n".join(f"{i+1}. {ex}" for i, ex in enumerate(shuffled))
        prompt_ood = (
            "Voici des exemples de textes où une feature neuronale est fortement activée "
            "(sauf un, qui est un contrôle négatif).\n\n"
            f"{examples_text}\n\n"
            "Quel numéro est l'intrus (celui qui ne partage pas le concept commun des autres) ? "
            "Réponds uniquement avec le numéro."
        )
        inputs = _apply_chat_and_extract(
            tokenizer, [{"role": "user", "content": prompt_ood}],
            device=model.device, add_generation_prompt=True, return_tensors="pt",
            enable_thinking=False,
        )
        with torch.no_grad():
            out = model.generate(input_ids=inputs, max_new_tokens=8, do_sample=False)
            resp = tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()
        try:
            predicted = int(re.search(r"\d+", resp).group())
        except Exception:
            predicted = -1
        predicted_positions.append(predicted)
        corrects.append(predicted == correct_answer)
    return corrects, predicted_positions


def main():
    random.seed(SEED)
    with open(JUDGE_CACHE, encoding="utf-8") as f:
        judge_data = json.load(f)
    feature_indices = [int(k) for k in judge_data.keys()]
    print(f"[robustness-r0] {len(feature_indices)} features R0 (déjà jugées, single-shot) à retester.")

    print("[robustness-r0] Reconstruction du split train/test + groupes mail parent...")
    train_texts, _, _, _, train_groups, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED, return_groups=True,
    )
    n_train = len(train_texts)
    print(f"[robustness-r0] n_train = {n_train}")

    all_doc_acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts_ext_d1024.pt")
    if not os.path.exists(all_doc_acts_path):
        all_doc_acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts.pt")
    print(f"[robustness-r0] Chargement des activations : {all_doc_acts_path}")
    all_doc_sae_acts = load_all_doc_acts(all_doc_acts_path)
    train_doc_acts = all_doc_sae_acts[:n_train]

    print("[robustness-r0] Chargement du juge R0 (JUDGE_MODEL_ID, Qwen3.8-27B — découplé de MODEL_ID)...")
    model, tokenizer = load_judge_model(device=DEVICE)
    model.eval()

    results = {}
    n_flipped_to_interp = 0
    n_flipped_to_noninterp = 0
    n_majority_interp = 0
    agreement_rates = []

    for i, f_idx in enumerate(feature_indices):
        pos_examples, neg_example = build_feature_examples_with_control(
            f_idx, TOKEN_FRAGMENTS_DIR, train_doc_acts, offset=0, n_pos=9, doc_groups=train_groups,
        )
        if len(pos_examples) < 3 or neg_example is None:
            continue

        corrects, predicted_positions = repeated_odd_one_out(
            model, tokenizer, pos_examples, neg_example, N_REPEATS
        )
        n_correct = sum(corrects)
        majority_interp = int(n_correct > N_REPEATS / 2)
        agreement = max(n_correct, N_REPEATS - n_correct) / N_REPEATS

        original_interp = judge_data[str(f_idx)].get("interp_score", 0)
        if original_interp == 0 and majority_interp == 1:
            n_flipped_to_interp += 1
        elif original_interp == 1 and majority_interp == 0:
            n_flipped_to_noninterp += 1
        n_majority_interp += majority_interp
        agreement_rates.append(agreement)

        results[f_idx] = {
            "original_interp_score": original_interp,
            "majority_interp_score": majority_interp,
            "n_correct_of_n_repeats": f"{n_correct}/{N_REPEATS}",
            "agreement_rate": agreement,
            "predicted_positions": predicted_positions,
        }
        if (i + 1) % 25 == 0:
            print(f"[robustness-r0] {i+1}/{len(feature_indices)} features retestées...")

    n_tested = len(results)
    single_shot_rate = sum(v["original_interp_score"] for v in results.values()) / n_tested
    majority_rate = n_majority_interp / n_tested
    unanimous = sum(1 for v in results.values() if v["agreement_rate"] == 1.0)
    mean_agreement = float(np.mean(agreement_rates))

    summary = {
        "n_tested": n_tested,
        "n_repeats": N_REPEATS,
        "judge_model_id": "JUDGE_MODEL_ID (Qwen3.8-27B, découplé)",
        "single_shot_interp_rate": single_shot_rate,
        "majority_vote_interp_rate": majority_rate,
        "n_flipped_0_to_1": n_flipped_to_interp,
        "n_flipped_1_to_0": n_flipped_to_noninterp,
        "n_flipped_total": n_flipped_to_interp + n_flipped_to_noninterp,
        "n_unanimous": unanimous,
        "pct_unanimous": unanimous / n_tested,
        "mean_agreement_rate": mean_agreement,
    }
    print("\n" + "=" * 70)
    print(" RÉSUMÉ — ROBUSTESSE DU PROTOCOLE ODD-ONE-OUT SOUS R0")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_feature": results}, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Écrit : {OUT_PATH}")


if __name__ == "__main__":
    main()
