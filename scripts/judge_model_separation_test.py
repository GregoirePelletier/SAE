"""
scripts/judge_model_separation_test.py — jusqu'à l'introduction de
JUDGE_MODEL_ID (src/config.py), le juge d'auto-interprétation et le modèle
dont on extrait le residual stream étaient le même checkpoint rechargé deux
fois (saev5.py) : risque de biais d'auto-préférence jamais mesuré. Rejuge les
mêmes 150 features déjà en cache avec ALT_JUDGE_MODEL_ID au lieu du juge par
défaut, même protocole odd-one-out, mêmes exemples -- isole l'effet du CHOIX
du modèle juge, à activations extraites identiques. `load_judge_model`
(chargement partagé avec le pipeline principal) gère aussi bien un juge même
famille que famille différente (Qwen3.8-27B-FP8, VLM natif + quantization_config
e4m3) sans changement de ce script.

Usage :
    SAVE_DIR=./results_v10_emails_main/ PYTHONPATH=. \
      .venv/bin/python scripts/judge_model_separation_test.py
"""
from __future__ import annotations

import json
import os
import random
import sys

import torch
from src.sae.sae_shared import load_all_doc_acts

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "sae"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import SAVE_DIR, CORPUS_SPLIT_SEED, LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH
from src.sae.judge import odd_one_out_judge, load_judge_model
from src.data.preparation import build_email_train_test_corpus
from src.storage.fragment_store import resolve_extension_fragments_dir

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALT_JUDGE_MODEL_ID = os.environ.get("ALT_JUDGE_MODEL_ID", "./models/gemma-3-4b-it")
# SEED ajouté (audit 2026-08 round 2, §1, B.17) : cf. c2_original_only_rejudge.py
# pour la justification -- ce script n'était jamais seedé avant ce correctif.
SEED = int(os.environ.get("SEED", "42"))

CACHE_DIR = os.path.join(SAVE_DIR, "cache")
JUDGE_CACHE = os.path.join(CACHE_DIR, "p1_judge_labels_extended.json")
TOKEN_FRAGMENTS_DIR = resolve_extension_fragments_dir(CACHE_DIR)  # features EXTENSION uniquement (N1, AUDIT_SAE_2026-08.md §8) -- p1_token_fragments_ext si présent (post-N1), repli p1_token_fragments sinon (legacy).
# Tag dérivé mécaniquement d'ALT_JUDGE_MODEL_ID (R5) -- sans lui, deux juges
# alternatifs différents lancés à la même SEED écrasent le même fichier de
# sortie (OUT_PATH n'était keyé QUE par SEED avant ce correctif, jamais un
# problème tant qu'un seul juge alternatif -- gemma-3-4b-it -- avait jamais
# été testé).
_ALT_JUDGE_TAG = os.path.basename(ALT_JUDGE_MODEL_ID.rstrip("/"))
OUT_PATH = os.path.join(CACHE_DIR, f"p1_judge_model_separation_{_ALT_JUDGE_TAG}_seed{SEED}.json")


def main() -> None:
    random.seed(SEED)
    with open(JUDGE_CACHE, encoding="utf-8") as f:
        original = json.load(f)
    feature_indices = [int(k) for k in original.keys()]
    print(f"[separation] {len(feature_indices)} features, juge alternatif={ALT_JUDGE_MODEL_ID}")

    all_doc_acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts_ext_d1024.pt")
    if not os.path.exists(all_doc_acts_path):
        all_doc_acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts.pt")
    all_doc_sae_acts = load_all_doc_acts(all_doc_acts_path)
    train_texts, _, _, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED,
    )
    train_doc_acts = all_doc_sae_acts[:len(train_texts)]

    model, tokenizer = load_judge_model(judge_model_id=ALT_JUDGE_MODEL_ID, device=DEVICE)

    alt_results = odd_one_out_judge(
        model=model, tokenizer=tokenizer, feature_indices=feature_indices,
        token_fragments_dir=TOKEN_FRAGMENTS_DIR, acts=train_doc_acts, offset=0,
    )

    n_orig_interp = sum(1 for v in original.values() if v.get("interp_score") == 1)
    n_alt_interp = sum(1 for v in alt_results.values() if v.get("interp_score") == 1)
    n_flip_0_1 = n_flip_1_0 = n_agree = 0
    for f_idx in feature_indices:
        o = original[str(f_idx)].get("interp_score", 0)
        a = alt_results[f_idx].get("interp_score", 0)
        if o == a:
            n_agree += 1
        elif o == 0 and a == 1:
            n_flip_0_1 += 1
        elif o == 1 and a == 0:
            n_flip_1_0 += 1

    summary = {
        "n_tested": len(feature_indices),
        "judge_original": f"juge ayant produit {JUDGE_CACHE} (cf. src.config.JUDGE_MODEL_ID pour le juge par défaut actuel)",
        "judge_alternative": ALT_JUDGE_MODEL_ID,
        "interp_rate_original": n_orig_interp / len(feature_indices),
        "interp_rate_alternative": n_alt_interp / len(feature_indices),
        "n_agree": n_agree,
        "agreement_rate": n_agree / len(feature_indices),
        "n_flipped_0_to_1": n_flip_0_1,
        "n_flipped_1_to_0": n_flip_1_0,
    }
    print("\n" + "=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "alt_per_feature": alt_results}, f, indent=2, ensure_ascii=False)
    print(f"\n[+] {OUT_PATH}")


if __name__ == "__main__":
    main()
