"""
scripts/judge_robustness_check.py — Teste la robustesse du protocole odd-one-out
(src/sae/judge.py::odd_one_out_judge) face au biais de position/ordre des exemples.

Contexte (cf. RESULTS_TESTS.md §12, report/04_limites_et_perspectives.md) : après
correction du corpus d'entraînement (emails dominants), le taux d'interprétabilité
des features d'extension reste à ~41-45% (n=150), sans effet du volume de tokens
(testé 100k/500k/2M). Piste non testée : le protocole ne fait qu'UNE SEULE décision
greedy par feature -- si le juge est sensible à la position arbitraire de l'intrus
dans la liste mélangée (biais de position connu des LLM en QCM), une partie du taux
d'échec pourrait être due au protocole de jugement lui-même, pas à la feature.

Ce script réutilise les activations et fragments DÉJÀ calculés par un run complet
(aucune réextraction Gemma-3, juste un rechargement du modèle pour le rôle de juge) :
pour chaque feature déjà jugée, répète la question odd-one-out N_REPEATS fois avec un
mélange aléatoire différent à chaque fois (même jeu d'exemples, ordre différent),
calcule le vote majoritaire et le taux d'accord, et compare au résultat single-shot
déjà en cache.

Usage :
    SAVE_DIR=./results_v10_emails_main/ MODEL_SIZE=12b \
      PYTHONPATH=. .venv/bin/python scripts/judge_robustness_check.py
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "sae"))

from src.config import MODEL_ID, HF_TOKEN, DTYPE, SAVE_DIR
from src.sae.judge import build_feature_examples_with_control, _apply_chat_and_extract
from src.data.preparation import build_email_train_test_corpus
from src.config import LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if DTYPE == "bf16" else torch.float16
N_REPEATS = int(os.environ.get("N_REPEATS", "5"))
SEED = int(os.environ.get("SEED", "42"))

CACHE_DIR = os.path.join(SAVE_DIR, "cache")
JUDGE_CACHE = os.path.join(CACHE_DIR, "p1_judge_labels_extended.json")
TOKEN_FRAGMENTS_DIR = os.path.join(CACHE_DIR, "p1_token_fragments")
OUT_PATH = os.path.join(CACHE_DIR, "p1_judge_robustness.json")


def repeated_odd_one_out(model, tokenizer, pos_examples, neg_example, n_repeats):
    """Répète la question odd-one-out n_repeats fois (même exemples, ordre différent
    à chaque fois). Retourne (votes, corrects) : listes booléennes par répétition."""
    all_examples = pos_examples + ([neg_example] if neg_example else [])
    neg_position = len(all_examples) - 1 if neg_example else None

    corrects = []
    predicted_positions = []  # position (1-based) prédite, pour diagnostiquer le biais
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
        )
        with torch.no_grad():
            out = model.generate(input_ids=inputs, max_new_tokens=8, do_sample=False)
            resp = tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()
        try:
            import re
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
    print(f"[robustness] {len(feature_indices)} features (déjà jugées, single-shot) à retester.")

    print("[robustness] Reconstruction du split train/test (déterministe, SEED fixe, pas de GPU)...")
    train_texts, _, _, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=SEED,
    )
    n_train = len(train_texts)
    print(f"[robustness] n_train = {n_train}")

    all_doc_acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts_ext_d1024.pt")
    if not os.path.exists(all_doc_acts_path):
        all_doc_acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts.pt")
    print(f"[robustness] Chargement des activations : {all_doc_acts_path}")
    all_doc_sae_acts = torch.load(all_doc_acts_path, map_location="cpu", weights_only=True)
    train_doc_acts = all_doc_sae_acts[:n_train]

    print(f"[robustness] Chargement du juge : {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=TORCH_DTYPE, device_map=DEVICE,
        low_cpu_mem_usage=True, token=HF_TOKEN, trust_remote_code=True, local_files_only=True,
    ).eval()

    results = {}
    n_flipped_to_interp = 0
    n_flipped_to_noninterp = 0
    n_majority_interp = 0
    agreement_rates = []

    for i, f_idx in enumerate(feature_indices):
        pos_examples, neg_example = build_feature_examples_with_control(
            f_idx, TOKEN_FRAGMENTS_DIR, train_doc_acts, offset=0, n_pos=9,
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
            print(f"[robustness] {i+1}/{len(feature_indices)} features retestées...")

    n_tested = len(results)
    single_shot_rate = sum(v["original_interp_score"] for v in results.values()) / n_tested
    majority_rate = n_majority_interp / n_tested
    mean_agreement = float(np.mean(agreement_rates))

    summary = {
        "n_tested": n_tested,
        "n_repeats": N_REPEATS,
        "single_shot_interp_rate": single_shot_rate,
        "majority_vote_interp_rate": majority_rate,
        "n_flipped_0_to_1": n_flipped_to_interp,
        "n_flipped_1_to_0": n_flipped_to_noninterp,
        "mean_agreement_rate": mean_agreement,
    }
    print("\n" + "=" * 70)
    print(" RÉSUMÉ — ROBUSTESSE DU PROTOCOLE ODD-ONE-OUT")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_feature": results}, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Écrit : {OUT_PATH}")


if __name__ == "__main__":
    main()
