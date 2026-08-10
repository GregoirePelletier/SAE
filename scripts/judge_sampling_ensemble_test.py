"""
scripts/judge_sampling_ensemble_test.py — robustesse du juge par ÉCHANTILLONNAGE
(temperature>0), distinct de judge_robustness_check.py (réordonnancement des
exemples, décision toujours greedy do_sample=False). Context.md flague le vote
majoritaire sur générations comme piste non testée -- judge_robustness_check.py
ne le fait pas : il ne varie que l'ordre de présentation, jamais la génération
elle-même. Ce script isole la variance pure de génération : MÊME ordre
d'exemples à chaque répétition, seul do_sample=True/temperature change.

Usage :
    SAVE_DIR=./results_v10_emails_main/ PYTHONPATH=. \
      .venv/bin/python scripts/judge_sampling_ensemble_test.py
"""
from __future__ import annotations

import json
import os
import random
import re
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "sae"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import MODEL_ID, HF_TOKEN, DTYPE, SAVE_DIR, CORPUS_SPLIT_SEED
from src.sae.judge import build_feature_examples_with_control, _apply_chat_and_extract

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if DTYPE == "bf16" else torch.float16
N_REPEATS = int(os.environ.get("N_REPEATS", "5"))
TEMPERATURE = float(os.environ.get("JUDGE_TEMPERATURE", "0.7"))
SEED = int(os.environ.get("SEED", "42"))

CACHE_DIR = os.path.join(SAVE_DIR, "cache")
JUDGE_CACHE = os.path.join(CACHE_DIR, "p1_judge_labels_extended.json")
TOKEN_FRAGMENTS_DIR = os.path.join(CACHE_DIR, "p1_token_fragments")
OUT_PATH = os.path.join(CACHE_DIR, "p1_judge_sampling_ensemble.json")


def sampled_odd_one_out(model, tokenizer, pos_examples, neg_example, correct_answer, examples_text):
    prompt = (
        "Voici des exemples de textes où une feature neuronale est fortement activée "
        "(sauf un, qui est un contrôle négatif).\n\n"
        f"{examples_text}\n\n"
        "Quel numéro est l'intrus (celui qui ne partage pas le concept commun des autres) ? "
        "Réponds uniquement avec le numéro."
    )
    inputs = _apply_chat_and_extract(
        tokenizer, [{"role": "user", "content": prompt}],
        device=model.device, add_generation_prompt=True, return_tensors="pt",
    )
    with torch.no_grad():
        out = model.generate(input_ids=inputs, max_new_tokens=8, do_sample=True,
                              temperature=TEMPERATURE, top_p=0.95)
        resp = tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()
    try:
        predicted = int(re.search(r"\d+", resp).group())
    except Exception:
        predicted = -1
    return predicted == correct_answer, predicted


def main():
    random.seed(SEED)
    with open(JUDGE_CACHE, encoding="utf-8") as f:
        judge_data = json.load(f)
    feature_indices = [int(k) for k in judge_data.keys()]
    print(f"[sampling-ensemble] {len(feature_indices)} features, N_REPEATS={N_REPEATS}, T={TEMPERATURE}")

    all_doc_acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts_ext_d1024.pt")
    if not os.path.exists(all_doc_acts_path):
        all_doc_acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts.pt")
    all_doc_sae_acts = torch.load(all_doc_acts_path, map_location="cpu", weights_only=True)

    from src.data.preparation import build_email_train_test_corpus
    from src.config import LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH
    train_texts, _, _, _ = build_email_train_test_corpus(LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED)
    train_doc_acts = all_doc_sae_acts[:len(train_texts)]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=TORCH_DTYPE, device_map=DEVICE,
        low_cpu_mem_usage=True, token=HF_TOKEN, trust_remote_code=True, local_files_only=True,
    ).eval()

    results = {}
    n_majority_interp = 0
    agreement_rates = []
    n_flipped_to_interp = n_flipped_to_noninterp = 0

    for i, f_idx in enumerate(feature_indices):
        pos_examples, neg_example = build_feature_examples_with_control(
            f_idx, TOKEN_FRAGMENTS_DIR, train_doc_acts, offset=0, n_pos=9,
        )
        if len(pos_examples) < 3 or neg_example is None:
            continue

        # Ordre FIXE (pas de réordonnancement -- isole la variance de génération,
        # cf. judge_robustness_check.py pour la variance d'ordre déjà mesurée).
        all_examples = pos_examples + [neg_example]
        correct_answer = len(all_examples)  # négatif toujours en dernière position ici
        examples_text = "\n".join(f"{j+1}. {ex}" for j, ex in enumerate(all_examples))

        corrects, predicted = [], []
        for _ in range(N_REPEATS):
            c, p = sampled_odd_one_out(model, tokenizer, pos_examples, neg_example, correct_answer, examples_text)
            corrects.append(c)
            predicted.append(p)

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
            "predicted": predicted,
        }
        if (i + 1) % 25 == 0:
            print(f"[sampling-ensemble] {i+1}/{len(feature_indices)}...")

    n_tested = len(results)
    summary = {
        "n_tested": n_tested,
        "n_repeats": N_REPEATS,
        "temperature": TEMPERATURE,
        "single_shot_interp_rate": sum(v["original_interp_score"] for v in results.values()) / n_tested,
        "majority_vote_interp_rate": n_majority_interp / n_tested,
        "n_flipped_0_to_1": n_flipped_to_interp,
        "n_flipped_1_to_0": n_flipped_to_noninterp,
        "mean_agreement_rate": float(np.mean(agreement_rates)),
    }
    print("\n" + "=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_feature": results}, f, indent=2, ensure_ascii=False)
    print(f"\n[+] {OUT_PATH}")


if __name__ == "__main__":
    main()
