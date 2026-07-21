"""
scripts/explanation_plausibility_test.py — Plausibilité de l'explication document-level,
par choix forcé (comparatif, pas d'auto-évaluation de confiance -- cf. RESULTS_TESTS.md
§13.1/§15.4 : le jugement comparatif odd-one-out est plus fiable qu'un score de
confiance auto-rapporté, systématiquement trop optimiste).

Question : pour un mail donné, l'ensemble des concepts/labels SAE les plus actifs
constitue-t-il une MEILLEURE explication de son contenu qu'un ensemble de concepts
choisis au hasard dans le même vocabulaire de labels ?

Protocole (choix forcé, comme odd_one_out_judge mais au niveau document) :
  1. Pour chaque document échantillonné : calcule le top-K réel (features les plus
     actives, labels Neuronpedia/juge) et un décoy de K labels tirés au hasard dans le
     même vocabulaire de labels (features NON parmi les plus actives pour ce document).
  2. Présente le mail + les deux ensembles A/B (ordre mélangé aléatoirement) au juge
     (Gemma-3-12B-it), demande lequel explique le mieux le contenu du mail.
  3. Taux de succès = fréquence à laquelle le juge choisit l'ensemble réel -- comparé
     au hasard (50%).

Nécessite le juge LLM (GPU) mais réutilise les activations déjà en cache -- aucune
réextraction Gemma-3.

Usage : PYTHONPATH=. .venv/bin/python scripts/explanation_plausibility_test.py
"""
from __future__ import annotations

import json
import os
import random
import re
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "sae"))

from src.config import (
    LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, SAVE_DIR, MODEL_ID, HF_TOKEN, DTYPE,
)
from src.data.dataset import load_mails_tsv
from src.data.preparation import build_email_train_test_corpus
from src.sae.judge import _apply_chat_and_extract

CACHE_DIR = os.path.join(SAVE_DIR, "cache")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if DTYPE == "bf16" else torch.float16
SEED = 42
TOP_K = 8
N_DOCS = int(os.environ.get("N_DOCS_PLAUSIBILITY", "60"))
random.seed(SEED)
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


def load_real_emails_and_acts():
    kept_row_indices, df_full = replicate_load_and_clean_emails_with_index(LOCAL_MAILS_PATH)
    n_real = len(kept_row_indices)
    rng = np.random.default_rng(SEED)
    test_mask = rng.random(n_real) < float(os.environ.get("EMAIL_TEST_SPLIT", "0.05"))
    train_positions = [i for i in range(n_real) if not test_mask[i]]
    k_train_original = len(train_positions)

    train_texts, train_labels, _, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=SEED,
    )
    n_original_train = sum(1 for l in train_labels if l == "original")
    assert n_original_train == k_train_original, "Incohérence de correspondance"

    acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts_ext_d1024.pt")
    if not os.path.exists(acts_path):
        acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts.pt")
    all_doc_acts = torch.load(acts_path, map_location="cpu", weights_only=True)
    original_train_acts = all_doc_acts[:k_train_original].float()
    real_texts = train_texts[:k_train_original]
    return original_train_acts, real_texts


def load_label_map():
    with open("local_data/neuronpedia_labels/neuronpedia_labels_24-gemmascope-2-res-16k.json") as f:
        labels_core = {int(k): v for k, v in json.load(f).items()}
    judge_path = os.path.join(CACHE_DIR, "p1_judge_labels_extended.json")
    label_map = dict(labels_core)
    if os.path.exists(judge_path):
        with open(judge_path) as f:
            judge_ext = json.load(f)
        for k, v in judge_ext.items():
            label_map[int(k)] = "[EXT] " + v.get("label", f"F{k}")
    return label_map


def clean_labels(indices, label_map):
    out = []
    for idx in indices:
        lbl = label_map.get(int(idx))
        if lbl and not re.fullmatch(r"(\[EXT\]\s*)?F\d+", lbl.strip()):
            out.append(lbl)
    return out


def build_prompt(text: str, set_a: list[str], set_b: list[str]) -> str:
    a_txt = "\n".join(f"- {l}" for l in set_a)
    b_txt = "\n".join(f"- {l}" for l in set_b)
    return (
        "Voici un mail client, et deux listes de concepts censés l'expliquer.\n\n"
        f"MAIL :\n{text[:1500]}\n\n"
        f"LISTE A :\n{a_txt}\n\n"
        f"LISTE B :\n{b_txt}\n\n"
        "Laquelle des deux listes explique le MIEUX le contenu réel de ce mail ? "
        "Réponds uniquement par 'A' ou 'B'."
    )


def main():
    print("[plausibility] Chargement des activations et labels...")
    acts, texts = load_real_emails_and_acts()
    label_map = load_label_map()
    all_labeled_features = [f for f in range(acts.shape[1])
                             if clean_labels([f], label_map)]
    print(f"[plausibility] {acts.shape[0]} mails réels, {len(all_labeled_features)} features labellisées.")

    doc_indices = rng_np.choice(acts.shape[0], size=min(N_DOCS, acts.shape[0]), replace=False)

    print(f"[plausibility] Chargement du juge : {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=TORCH_DTYPE, device_map=DEVICE,
        low_cpu_mem_usage=True, token=HF_TOKEN, trust_remote_code=True, local_files_only=True,
    ).eval()

    n_correct = 0
    n_tested = 0
    results = []
    for i, doc_idx in enumerate(doc_indices):
        x = acts[doc_idx].numpy()
        active_order = np.argsort(x)[::-1]
        real_top = [int(f) for f in active_order[:TOP_K * 3] if clean_labels([f], label_map)][:TOP_K]
        if len(real_top) < TOP_K // 2:
            continue
        real_labels = clean_labels(real_top, label_map)

        decoy_pool = [f for f in all_labeled_features if f not in set(real_top)]
        decoy = list(rng_np.choice(decoy_pool, size=len(real_labels), replace=False))
        decoy_labels = clean_labels(decoy, label_map)

        is_real_a = random.random() < 0.5
        set_a, set_b = (real_labels, decoy_labels) if is_real_a else (decoy_labels, real_labels)
        prompt = build_prompt(texts[doc_idx], set_a, set_b)

        inputs = _apply_chat_and_extract(
            tokenizer, [{"role": "user", "content": prompt}],
            device=model.device, add_generation_prompt=True, return_tensors="pt",
        )
        with torch.no_grad():
            out = model.generate(input_ids=inputs, max_new_tokens=4, do_sample=False)
            resp = tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip().upper()

        picked_real = ("A" in resp and is_real_a) or ("B" in resp and not is_real_a)
        n_tested += 1
        n_correct += int(picked_real)
        results.append({
            "doc_idx": int(doc_idx), "real_labels": real_labels, "decoy_labels": decoy_labels,
            "is_real_a": is_real_a, "judge_response": resp, "picked_real": picked_real,
        })
        if (i + 1) % 10 == 0:
            print(f"[plausibility] {i+1}/{len(doc_indices)} documents testés "
                  f"(taux de succès courant : {n_correct}/{n_tested} = {100*n_correct/max(n_tested,1):.1f}%)")

    summary = {
        "n_tested": n_tested,
        "n_correct": n_correct,
        "success_rate": n_correct / max(n_tested, 1),
        "chance_level": 0.5,
    }
    print("\n" + "=" * 70)
    print(" RÉSUMÉ — PLAUSIBILITÉ DE L'EXPLICATION (choix forcé réel vs aléatoire)")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out_path = os.path.join(CACHE_DIR, "explanation_plausibility_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "examples": results}, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Écrit : {out_path}")


if __name__ == "__main__":
    main()
