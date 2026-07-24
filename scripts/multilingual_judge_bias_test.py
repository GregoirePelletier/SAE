"""
scripts/multilingual_judge_bias_test.py — Teste si le juge (Gemma-3-12B-it)
interprète MIEUX les mêmes features quand les exemples et le prompt sont en
ANGLAIS plutôt qu'en français, conformément à l'hypothèse documentée dans la
littérature multilingue (Resck et al. 2025 ; Sparse Autoencoders Can Capture
Language-Specific Concepts Across Diverse Languages, arXiv:2507.11230) : les LLM
représenteraient souvent les concepts multilingues via une structure interne
dominée par l'anglais, ce qui pourrait dégrader la qualité de l'auto-interprétation
sur un corpus non-anglophone comme le nôtre (mails en français).

Protocole : réutilise EXACTEMENT les mêmes 150 features déjà jugées (même
sélection top-N par magnitude, mêmes activations, même fragments en cache --
aucune réextraction Gemma-3, aucun réentraînement de SAE). Pour chaque feature :
  1. Reconstruit les mêmes pos_examples/neg_example français que le run original
     (build_feature_examples_with_control, src/sae/judge.py, inchangée).
  2. Traduit ces exemples en anglais en UN appel (JSON in/JSON out), en préservant
     les marqueurs <<mot>>.
  3. Rejoue le protocole odd-one-out (prompt + décision) intégralement en anglais.
  4. Compare le taux d'interprétabilité anglais au taux français déjà connu
     (résultat du run principal, RESULTS_TESTS.md §5.1/§12), sur les MÊMES features.

Limite assumée : la traduction elle-même est faite par le même modèle (Gemma-3),
introduisant un bruit de traduction propre (contrairement à un re-entraînement
complet sur corpus anglais natif, qui testerait une hypothèse légèrement
différente -- cf. RESULTS_TESTS.md pour la discussion complète).

Usage :
    SAVE_DIR=./results_v10_emails_main/ MODEL_SIZE=12b \
      PYTHONPATH=. .venv/bin/python scripts/multilingual_judge_bias_test.py
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

from src.config import MODEL_ID, HF_TOKEN, DTYPE, SAVE_DIR, LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH
from src.sae.judge import build_feature_examples_with_control, _apply_chat_and_extract
from src.data.preparation import build_email_train_test_corpus

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if DTYPE == "bf16" else torch.float16
SEED = int(os.environ.get("SEED", "42"))

CACHE_DIR = os.path.join(SAVE_DIR, "cache")
JUDGE_CACHE = os.path.join(CACHE_DIR, "p1_judge_labels_extended.json")
TOKEN_FRAGMENTS_DIR = os.path.join(CACHE_DIR, "p1_token_fragments")
OUT_PATH = os.path.join(CACHE_DIR, "multilingual_judge_bias_results.json")


def translate_examples_to_english(examples: list[str], model, tokenizer) -> list[str]:
    """Traduit une liste de courts extraits FR->EN en un seul appel, en préservant
    les marqueurs <<mot>> (déplacés sur le mot traduit correspondant)."""
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
        return examples  # échec de parsing -> pas de traduction, feature exclue en aval


def odd_one_out_english(pos_examples_en: list[str], neg_example_en: str, model, tokenizer) -> tuple[int, int | None]:
    """Réplique odd_one_out_judge (src/sae/judge.py) intégralement en anglais.
    Retourne (interp_score, predicted)."""
    all_examples = pos_examples_en + [neg_example_en]
    neg_position = len(all_examples) - 1
    indices = list(range(len(all_examples)))
    random.shuffle(indices)
    shuffled = [all_examples[i] for i in indices]
    correct_answer = indices.index(neg_position) + 1  # 1-based

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
    print(f"[multilingual] {len(feature_indices)} features (mêmes que le run odd-one-out original).")

    print("[multilingual] Reconstruction du split train/test (déterministe, pas de GPU)...")
    train_texts, _, _, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=SEED,
    )
    n_train = len(train_texts)

    acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts_ext_d1024.pt")
    if not os.path.exists(acts_path):
        acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts.pt")
    all_doc_acts = torch.load(acts_path, map_location="cpu", weights_only=True)
    train_doc_acts = all_doc_acts[:n_train]

    print(f"[multilingual] Chargement du juge : {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=TORCH_DTYPE, device_map=DEVICE,
        low_cpu_mem_usage=True, token=HF_TOKEN, trust_remote_code=True, local_files_only=True,
    ).eval()

    results = {}
    n_en_interp, n_translation_failed, n_fr_to_en_flip, n_en_to_fr_flip = 0, 0, 0, 0
    n_tested = 0

    for i, f_idx in enumerate(feature_indices):
        original = original_judge_data[str(f_idx)]
        original_interp = int(original.get("interp_score", 0) or 0)

        pos_examples, neg_example = build_feature_examples_with_control(
            f_idx, TOKEN_FRAGMENTS_DIR, train_doc_acts, offset=0, n_pos=9,
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
            print(f"[multilingual] {i+1}/{len(feature_indices)} — "
                  f"EN interp courant : {n_en_interp}/{n_tested} = {100*n_en_interp/max(n_tested,1):.1f}%")

    n_fr_interp = sum(v["interp_score_fr_original"] for v in results.values())
    summary = {
        "n_tested": n_tested,
        "n_translation_failed": n_translation_failed,
        "interp_rate_fr_original": n_fr_interp / max(n_tested, 1),
        "interp_rate_en_translated": n_en_interp / max(n_tested, 1),
        "n_features_flip_fr_noninterp_to_en_interp": n_fr_to_en_flip,
        "n_features_flip_fr_interp_to_en_noninterp": n_en_to_fr_flip,
    }
    print("\n" + "=" * 70)
    print(" RÉSUMÉ — BIAIS MULTILINGUE DU JUGE (FR original vs EN traduit)")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_feature": results}, f, indent=2, ensure_ascii=False)
    print(f"\n[multilingual] Résultats sauvegardés : {OUT_PATH}")


if __name__ == "__main__":
    main()
