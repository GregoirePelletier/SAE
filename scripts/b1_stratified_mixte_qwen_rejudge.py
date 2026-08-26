"""
scripts/b1_stratified_mixte_qwen_rejudge.py -- N4 (AUDIT_SAE_2026-08.md §8) :
§81 (B.1) conclut "résolu négativement" sur un écart -7,3 pts (89,3% mixte vs
82,0% originaux+filler, tous deux stratifiés, p=0,070) sans pouvoir distinguer
"pas de contamination" de "le juge Gemma est complaisant avec du texte généré
par Gemma" -- §48/§50/§52 ont testé juge et corpus séparément, jamais leur
interaction. Rejuge les 150 features de l'arme MIXTE (stratifié, §79,
`b2_stratified_selection_rejudge.json`, déjà en cache dans
`results_v10_emails_main/`) avec Qwen3.8-27B au lieu de gemma-3-12b-it --
même patron que `judge_model_separation_test.py`, mêmes exemples, seul le
juge change.

L'arme "originaux+filler" (§81, `results_v26_validation_layer24_v12_originals_
filler_matched_n150_h100/`) N'A PAS pu être rejugée de la même façon : ses
fragments token-level ont été supprimés par le nettoyage disque de cette
session (`docs/archived_runs_manifest.md`, seul `results_v10_emails_main/`
gardé complet) -- rejuger cette arme demanderait une extraction complète
fraîche, pas "quelques minutes de GPU" comme prévu par N4. Ce script mesure
donc seulement si Qwen est systématiquement plus/moins généreux que Gemma sur
l'arme mixte, pas l'interaction complète juge×corpus.

Usage :
    SAVE_DIR=./results_v10_emails_main/ PYTHONPATH=. \
      .venv/bin/python scripts/b1_stratified_mixte_qwen_rejudge.py
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
ALT_JUDGE_MODEL_ID = os.environ.get("ALT_JUDGE_MODEL_ID", "/home/h21486/SAE/models/Qwen3.8-27B")
SEED = int(os.environ.get("SEED", "42"))

CACHE_DIR = os.path.join(SAVE_DIR, "cache")
# Arme MIXTE stratifiée (§79) -- PAS p1_judge_labels_extended.json (arme
# magnitude/référence, §43/§63/§65/§83, déjà rejugée par judge_model_separation_test.py).
B2_CACHE = os.path.join(CACHE_DIR, "b2_stratified_selection_rejudge.json")
TOKEN_FRAGMENTS_DIR = resolve_extension_fragments_dir(CACHE_DIR)  # features EXTENSION uniquement (N1, AUDIT_SAE_2026-08.md §8) -- p1_token_fragments_ext si présent (post-N1), repli p1_token_fragments sinon (legacy).
_ALT_JUDGE_TAG = os.path.basename(ALT_JUDGE_MODEL_ID.rstrip("/"))
OUT_PATH = os.path.join(CACHE_DIR, f"b1_stratified_mixte_qwen_rejudge_{_ALT_JUDGE_TAG}_seed{SEED}.json")


def main() -> None:
    random.seed(SEED)
    with open(B2_CACHE, encoding="utf-8") as f:
        b2 = json.load(f)
    original = b2["results"]
    feature_indices = [int(k) for k in original.keys()]
    print(f"[b1-mixte-qwen] {len(feature_indices)} features (arme mixte stratifiée, §79), "
          f"juge alternatif={ALT_JUDGE_MODEL_ID}")

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
        "judge_original": f"gemma-3-12b-it (juge ayant produit {B2_CACHE}, arme mixte stratifiée §79)",
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
