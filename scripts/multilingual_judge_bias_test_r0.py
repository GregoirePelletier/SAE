"""
scripts/multilingual_judge_bias_test_r0.py — Biais multilingue du juge, sous
R0. RESULTS_TESTS.md §22 réutilisait `results_v10_emails_main` (sélection par
magnitude) et rechargeait MODEL_ID (Gemma-3-12b-it, l'extracteur) comme juge
ET traducteur -- un biais d'auto-préférence que CLAUDE.md interdit explicitement
de réintroduire. Corrige les mêmes écarts que
`judge_robustness_check_r0_test.py` : SAVE_DIR sur R0, juge/traducteur =
Qwen3.8-27B (`load_judge_model`), `doc_groups=train_groups` (déduplication
mail parent, comme l'appel officiel dans saev5.py::odd_one_out_judge).

Zéro réextraction : réutilise les activations/fragments déjà en cache et les
300 features déjà jugées de R0.

Usage (SLURM, 1 GPU) :
    SAVE_DIR=./results_v27_ablation_classic_setup_k5_25m_layer31/ \
      PYTHONPATH=. .venv/bin/python scripts/multilingual_judge_bias_test_r0.py
"""
from __future__ import annotations

import json
import os
import random
import re
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "sae"))

from src.config import SAVE_DIR, LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, CORPUS_SPLIT_SEED
from src.sae.judge import build_feature_examples_with_control, _apply_chat_and_extract, load_judge_model
from src.data.preparation import build_email_train_test_corpus
from src.sae.sae_shared import load_all_doc_acts
from src.storage.fragment_store import resolve_extension_fragments_dir

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = int(os.environ.get("SEED", "42"))

CACHE_DIR = os.path.join(SAVE_DIR, "cache")
JUDGE_CACHE = os.path.join(CACHE_DIR, "p1_judge_labels_extended.json")
TOKEN_FRAGMENTS_DIR = resolve_extension_fragments_dir(CACHE_DIR)
OUT_PATH = os.path.join(CACHE_DIR, "multilingual_judge_bias_r0_results.json")


def translate_examples_to_english(examples: list[str], model, tokenizer) -> list[str]:
    numbered = "\n".join(f"{i+1}. {ex}" for i, ex in enumerate(examples))
    prompt = (
        "Translate each of the following French text snippets into English. "
        "Each snippet may contain a marker <<word>> around one word or short phrase — "
        "keep the << >> marker in your translation, around the corresponding "
        "translated word/phrase (not necessarily the same position in the sentence).\n\n"
        f"{numbered}\n\n"
        'Respond in strict JSON: {"translations": ["...", "...", ...]} '
        f"with exactly {len(examples)} entries, in the same order."
    )
    inputs = _apply_chat_and_extract(
        tokenizer, [{"role": "user", "content": prompt}],
        device=model.device, add_generation_prompt=True, return_tensors="pt",
        enable_thinking=False,
    )
    with torch.no_grad():
        out = model.generate(input_ids=inputs, max_new_tokens=100 * len(examples), do_sample=False)
        resp = tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
    try:
        translations = json.loads(re.search(r"\{.*\}", resp, re.DOTALL).group())["translations"]
        if len(translations) != len(examples):
            raise ValueError("longueur incohérente")
        return translations
    except Exception:
        return examples


def odd_one_out_english(pos_examples_en: list[str], neg_example_en: str, model, tokenizer) -> tuple[int, int | None]:
    all_examples = pos_examples_en + [neg_example_en]
    neg_position = len(all_examples) - 1
    indices = list(range(len(all_examples)))
    random.shuffle(indices)
    shuffled = [all_examples[i] for i in indices]
    correct_answer = indices.index(neg_position) + 1

    examples_text = "\n".join(f"{i+1}. {ex}" for i, ex in enumerate(shuffled))
    prompt_ood = (
        "Here are text examples where a neural feature is strongly activated "
        "(except one, which is a negative control).\n\n"
        f"{examples_text}\n\n"
        "Which number is the odd one out (the one that does NOT share the common concept "
        "of the others)? Answer with only the number."
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
    return int(predicted == correct_answer), predicted


def main():
    random.seed(SEED)
    with open(JUDGE_CACHE, encoding="utf-8") as f:
        original_judge_data = json.load(f)
    feature_indices = [int(k) for k in original_judge_data.keys()]
    print(f"[multilingual-r0] {len(feature_indices)} features R0 (mêmes que le run odd-one-out original).")

    print("[multilingual-r0] Reconstruction du split train/test + groupes mail parent...")
    train_texts, _, _, _, train_groups, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED, return_groups=True,
    )
    n_train = len(train_texts)

    acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts_ext_d1024.pt")
    if not os.path.exists(acts_path):
        acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts.pt")
    all_doc_acts = load_all_doc_acts(acts_path)
    train_doc_acts = all_doc_acts[:n_train]

    print("[multilingual-r0] Chargement du juge R0 (JUDGE_MODEL_ID, Qwen3.8-27B — découplé de MODEL_ID)...")
    model, tokenizer = load_judge_model(device=DEVICE)
    model.eval()

    results = {}
    n_en_interp, n_translation_failed, n_fr_to_en_flip, n_en_to_fr_flip = 0, 0, 0, 0
    n_tested = 0

    for i, f_idx in enumerate(feature_indices):
        original = original_judge_data[str(f_idx)]
        original_interp = int(original.get("interp_score", 0) or 0)

        pos_examples, neg_example = build_feature_examples_with_control(
            f_idx, TOKEN_FRAGMENTS_DIR, train_doc_acts, offset=0, n_pos=9, doc_groups=train_groups,
        )
        if len(pos_examples) < 3 or not neg_example:
            continue

        all_fr = pos_examples + [neg_example]
        all_en = translate_examples_to_english(all_fr, model, tokenizer)
        if all_en == all_fr:
            n_translation_failed += 1
            continue
        pos_en, neg_en = all_en[:-1], all_en[-1]

        interp_en, predicted = odd_one_out_english(pos_en, neg_en, model, tokenizer)
        n_tested += 1
        n_en_interp += interp_en
        if original_interp == 0 and interp_en == 1:
            n_fr_to_en_flip += 1
        if original_interp == 1 and interp_en == 0:
            n_en_to_fr_flip += 1

        results[f_idx] = {
            "interp_score_fr_original": original_interp,
            "interp_score_en_translated": interp_en,
        }

        if (i + 1) % 25 == 0:
            print(f"[multilingual-r0] {i+1}/{len(feature_indices)} — "
                  f"EN interp courant : {n_en_interp}/{n_tested} = {100*n_en_interp/max(n_tested,1):.1f}%")

    n_fr_interp = sum(v["interp_score_fr_original"] for v in results.values())
    summary = {
        "n_tested": n_tested,
        "n_translation_failed": n_translation_failed,
        "judge_model_id": "JUDGE_MODEL_ID (Qwen3.8-27B, découplé)",
        "interp_rate_fr_original": n_fr_interp / max(n_tested, 1),
        "interp_rate_en_translated": n_en_interp / max(n_tested, 1),
        "n_features_flip_fr_noninterp_to_en_interp": n_fr_to_en_flip,
        "n_features_flip_fr_interp_to_en_noninterp": n_en_to_fr_flip,
        "n_features_flip_total": n_fr_to_en_flip + n_en_to_fr_flip,
    }
    print("\n" + "=" * 70)
    print(" RÉSUMÉ — BIAIS MULTILINGUE DU JUGE SOUS R0 (FR original vs EN traduit)")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_feature": results}, f, indent=2, ensure_ascii=False)
    print(f"\n[multilingual-r0] Résultats sauvegardés : {OUT_PATH}")


if __name__ == "__main__":
    main()
