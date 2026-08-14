"""
scripts/c2_original_only_rejudge.py — Test le plus direct de la boucle
auto-référentielle juge/générateur (`RESULTS_TESTS.md` §44/§48/§49) :
Gemma-3-12b-it génère ~92% du corpus d'entraînement ET sert de juge
d'auto-interprétation.

Réutilise le SAE d'extension DÉJÀ ENTRAÎNÉ (aucun réentraînement) et le MÊME
juge (Gemma-3-12b-it, aucun nouveau modèle téléchargé) sur les 150 features
déjà jugées de `results_v10_emails_main` -- seule la sélection des exemples
positifs change : restreinte aux mails ORIGINAUX uniquement (jamais de texte
généré par Gemma). §50 a confirmé la faisabilité : 150/150 features ont
>=9 candidats positifs provenant exclusivement de mails originaux.

Si le taux d'interprétabilité reste proche de 45,3% (référence, exemples
mixtes) avec des exemples 100% originaux, c'est la preuve directe la plus
forte possible contre C2 : le juge n'a alors JAMAIS vu de texte qu'il a
lui-même généré pour ces 150 jugements, et pourtant le taux ne bouge pas.

Usage :
    SAVE_DIR=./results_v10_emails_main/ PYTHONPATH=. \
      .venv/bin/python scripts/c2_original_only_rejudge.py
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import HF_TOKEN, DTYPE, SAVE_DIR, MODEL_ID, CORPUS_SPLIT_SEED, LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH
from src.data.preparation import build_email_train_test_corpus
import src.sae.judge as judge_mod
from src.sae.judge import (
    fragment_exists, load_fragment, feature_column, extract_causal_context,
    odd_one_out_judge,
)
import re

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if DTYPE == "bf16" else torch.float16
# SEED ajouté (audit 2026-08 round 2, §1, B.17) : ce script rebâtit ses
# exemples via `build_feature_examples_with_control` (`random.shuffle` sur
# le pool négatif), sans jamais poser de graine -- non reproductible d'un
# run à l'autre avant ce correctif. SEED=42 par défaut = comportement le
# plus proche du run historique non seedé (job 42878, §52) ; SEED!=42 sert
# à mesurer la variance inter-graine demandée par le round 2.
SEED = int(os.environ.get("SEED", "42"))

CACHE_DIR = os.path.join(SAVE_DIR, "cache")
JUDGE_CACHE = os.path.join(CACHE_DIR, "p1_judge_labels_extended.json")
TOKEN_FRAGMENTS_DIR = os.path.join(CACHE_DIR, "p1_token_fragments")
OUT_PATH = os.path.join(CACHE_DIR, f"c2_original_only_rejudge_seed{SEED}.json")


def main() -> None:
    random.seed(SEED)
    with open(JUDGE_CACHE, encoding="utf-8") as f:
        original = json.load(f)
    feature_indices = [int(k) for k in original.keys()]
    print(f"[c2-rejudge] {len(feature_indices)} features, exemples restreints aux mails originaux")

    train_texts, train_labels, _, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED,
    )
    orig_indices = {i for i, l in enumerate(train_labels) if l == "original"}
    print(f"[c2-rejudge] {len(orig_indices)}/{len(train_labels)} documents train = mail original")

    all_doc_acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts_ext_d1024.pt")
    if not os.path.exists(all_doc_acts_path):
        all_doc_acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts.pt")
    all_doc_sae_acts = torch.load(all_doc_acts_path, map_location="cpu", weights_only=True)
    train_acts = all_doc_sae_acts[: len(train_texts)]

    # Monkey-patch : même logique déterministe que build_feature_examples_with_control
    # (judge.py), restreinte aux indices de documents "original" uniquement. Ne modifie
    # PAS judge.py (module partagé) -- isolé à ce script d'ablation.
    _orig_fn = judge_mod.build_feature_examples_with_control

    def _restricted_build_examples(f_idx, token_fragments_dir, acts, offset=0, n_pos=9, neg_quantile=0.05):
        f_acts = acts[:, f_idx].detach().float().numpy()
        threshold_pos = 1e-6
        threshold_neg = float(np.quantile(f_acts, neg_quantile))
        sorted_desc = np.argsort(f_acts)[::-1]
        pos_examples = []
        seen_target_words = set()
        for d_idx in sorted_desc:
            if f_acts[d_idx] <= threshold_pos:
                break
            if int(d_idx) not in orig_indices:
                continue
            if not fragment_exists(token_fragments_dir, int(d_idx + offset)):
                continue
            doc_data = load_fragment(token_fragments_dir, int(d_idx + offset))
            token_acts = feature_column(doc_data, f_idx)
            max_act = token_acts.max()
            if max_act <= threshold_pos:
                continue
            target_idx = int(token_acts.argmax())
            ctx = extract_causal_context(doc_data["token_strings"], target_idx)
            m = re.search(r"<<(.+?)>>", ctx)
            target_word = m.group(1).strip().lower() if m else ctx.strip().lower()
            if target_word in seen_target_words:
                continue
            seen_target_words.add(target_word)
            pos_examples.append(ctx)
            if len(pos_examples) >= n_pos:
                break

        neg_pool = np.where(f_acts <= threshold_neg)[0].tolist()
        import random
        random.shuffle(neg_pool)
        neg_example = None
        for d_idx in neg_pool[:20]:
            if not fragment_exists(token_fragments_dir, int(d_idx + offset)):
                continue
            doc_data = load_fragment(token_fragments_dir, int(d_idx + offset))
            toks = doc_data["token_strings"]
            mid = len(toks) // 2
            neg_example = extract_causal_context(toks, mid)
            break
        return pos_examples, neg_example

    judge_mod.build_feature_examples_with_control = _restricted_build_examples

    print(f"[c2-rejudge] Chargement du juge {MODEL_ID} (même modèle que l'original)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=TORCH_DTYPE, device_map=DEVICE,
        low_cpu_mem_usage=True, token=HF_TOKEN, trust_remote_code=True, local_files_only=True,
    ).eval()

    results = odd_one_out_judge(
        model, tokenizer, feature_indices, TOKEN_FRAGMENTS_DIR, train_acts, offset=0, n_pos=9,
    )
    judge_mod.build_feature_examples_with_control = _orig_fn  # restauration (hygiène, script mono-usage)

    n = len(results)
    n_interp = sum(1 for v in results.values() if v.get("interp_score") == 1)
    n_dead = sum(1 for v in results.values() if v.get("label") == "dead_feature")
    rate = n_interp / n if n else float("nan")
    print(f"\n[c2-rejudge] Original-only : {n_interp}/{n} = {rate:.4f} interprétable ({n_dead} dead/insuffisant)")

    n_orig_ref = sum(1 for v in original.values() if v.get("interp_score") == 1)
    print(f"[c2-rejudge] Référence (exemples mixtes, run principal) : {n_orig_ref}/{len(original)} = {n_orig_ref/len(original):.4f}")

    json.dump(results, open(OUT_PATH, "w"), indent=2, ensure_ascii=False)
    print(f"[c2-rejudge] Sauvé : {OUT_PATH}")


if __name__ == "__main__":
    main()
