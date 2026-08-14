"""
scripts/audit_2026_08_b26_round2_fix.py — Corrige le bug du bug : le
correctif appliqué par `audit_2026_08_palier1_batch.py`
(`v[:-3] + r"\\w*)\\b"`) n'ajoutait `\\w*` qu'à la TOUTE DERNIÈRE alternative
de chaque groupe OR, pas à chacune -- toutes les alternatives précédentes
(les radicaux réellement cassés : `contest`, `r[ée]clamation`, `r[ée]sili`,
`clôtur`, `rembours`, `renseign`, `imm[ée]diat`) restaient non corrigées.
D'où les flips quasi nuls (0-3 documents) observés dans le premier passage --
ils ne mesuraient l'effet que sur la dernière alternative de chaque groupe.

Correctif réécrit à la main, alternative par alternative (pas de
transformation générique) : `\\w*` ajouté à chaque radical À UN SEUL MOT,
les alternatives-phrases (avec espaces, déjà spécifiques) laissées
inchangées, conformément à la recommandation explicite du round 2 de
l'audit.

Échantillon manuel (40 occurrences de "résili" avec contexte, lu directement
par l'agent avant ce script) : 40/40 sont des mentions authentiques d'intention
de résiliation (aucun faux positif de bruit détecté) -- soutient l'hypothèse
que le sous-comptage d'origine était bien sévère, pas que le radical capte du
bruit.

Usage : sbatch slurm/validation/run_audit_b26_round2_fix.slurm
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd
import torch

import src.data.dataset as dataset_mod
from src.config import LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, SAVE_DIR, CORPUS_SPLIT_SEED

ORIG_PATTERNS = dict(dataset_mod.INTENT_KEYWORDS_FR)

# Correctif écrit à la main, alternative par alternative (round 2, §0 point 1).
FIXED_PATTERNS_V2 = {
    "reclamation": r"\b(r[ée]clamation\w*|contest\w*|inadmissible\w*|scandaleux\w*|erreur de facturation)\b",
    "resiliation": r"\b(r[ée]sili\w*|clôtur\w*|mettre fin au contrat)\b",
    "remboursement": r"\b(rembours\w*|trop[- ]perçu|avoir\w*)\b",
    "information": r"\b(renseign\w*|information\w*|pourriez[- ]vous m'indiquer|comment (faire|proc[ée]der))\b",
    "urgence": r"\b(urgent\w*|imm[ée]diat\w*|sans d[ée]lai|coupure\w*)\b",
}

assert set(FIXED_PATTERNS_V2) == set(ORIG_PATTERNS)

OUT_PATH = os.path.join("docs", "audit_b26_round2_fix_results.json")


def main():
    print("=" * 78)
    print(" B.26 (round 2) — impact du correctif RÉÉCRIT (chaque alternative, pas la dernière seule)")
    print("=" * 78)

    df_orig = dataset_mod.load_mails_tsv(LOCAL_MAILS_PATH)
    corpus_impact = {}
    for intent in ORIG_PATTERNS:
        orig_mask = df_orig["text"].str.contains(ORIG_PATTERNS[intent], flags=re.I, regex=True)
        fixed_mask = df_orig["text"].str.contains(FIXED_PATTERNS_V2[intent], flags=re.I, regex=True)
        flip_on = int((fixed_mask & ~orig_mask).sum())
        flip_off = int((orig_mask & ~fixed_mask).sum())
        corpus_impact[intent] = {
            "n_pos_original": int(orig_mask.sum()),
            "n_pos_corrige_v2": int(fixed_mask.sum()),
            "flip_on_document_level": flip_on,
            "flip_off_document_level": flip_off,
        }
        print(f"  {intent:15s} n_orig={orig_mask.sum():5d} n_fixed_v2={fixed_mask.sum():5d} "
              f"flip_on={flip_on} flip_off={flip_off}")
    print()

    # Rejugement complet avec les labels V2 (même protocole que le batch précédent).
    dataset_mod.INTENT_KEYWORDS_FR = FIXED_PATTERNS_V2
    from src.data.preparation import build_email_train_test_corpus
    from src.analysis.metrics import downstream_classification
    from src.analysis.stats import two_proportion_test

    SEED = 42

    def replicate_load_and_clean_emails_with_index(tsv_path):
        df = dataset_mod.load_mails_tsv(tsv_path).rename(columns={"text": "document"})
        kept = []
        for row_idx, row in df.iterrows():
            if "document" not in row or pd.isna(row["document"]):
                continue
            raw_text = str(row["document"])
            clean_text = re.sub(r'^\s*(?:Objet|Subject)\s*:\s*[^\n]+\n*', '', raw_text, flags=re.IGNORECASE)
            clean_text = re.sub(r'\[\s*\{\s*"start".*?\}\s*\]', '', clean_text, flags=re.DOTALL).strip()
            if clean_text:
                kept.append(row_idx)
        return kept, df

    kept_row_indices, df_full = replicate_load_and_clean_emails_with_index(LOCAL_MAILS_PATH)
    n_real = len(kept_row_indices)
    rng = np.random.default_rng(SEED)
    test_mask = rng.random(n_real) < float(os.environ.get("EMAIL_TEST_SPLIT", "0.05"))
    train_positions = [i for i in range(n_real) if not test_mask[i]]
    k_train_original = len(train_positions)

    train_texts, train_labels, _, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED,
    )
    n_original_train = sum(1 for l in train_labels if l == "original")
    assert n_original_train == k_train_original

    PROBE_SAVE_DIR = "./results_v10_emails_main"
    acts_path = os.path.join(PROBE_SAVE_DIR, "cache", "p1_all_doc_acts_ext_d1024.pt")
    all_doc_acts = torch.load(acts_path, map_location="cpu", weights_only=True)
    original_train_acts = all_doc_acts[:k_train_original]

    row_indices_for_train_original = [kept_row_indices[i] for i in train_positions]
    intent_cols = [c for c in df_full.columns if c.startswith("intent_")]
    labels_df = df_full.loc[row_indices_for_train_original, intent_cols].reset_index(drop=True)

    # Références publiées (labels originaux buggés, RESULTS_TESTS.md §13.2)
    REFERENCE = {
        "intent_reclamation": {"n_pos": 1819, "acc_sae": 0.9773, "majority_baseline": 0.5512},
        "intent_remboursement": {"n_pos": 479, "acc_sae": 0.8455, "majority_baseline": 0.8548},
        "intent_information": {"n_pos": 599, "acc_sae": 0.8785, "majority_baseline": 0.8185},
        "intent_urgence": {"n_pos": 968, "acc_sae": 0.9773, "majority_baseline": 0.7067},
    }

    probe_results = {}
    print("Rejugement (labels V2, correctif complet) :")
    for col in intent_cols:
        y = labels_df[col].to_numpy()
        n_pos = int(y.sum())
        if n_pos < 10 or n_pos > len(y) - 10:
            print(f"  {col}: {n_pos}/{len(y)} -- ignoré.")
            continue
        pos_mask = torch.from_numpy(y.astype(bool))
        clf = downstream_classification(acts_by_label={
            "positif": original_train_acts[pos_mask],
            "negatif": original_train_acts[~pos_mask],
        })
        majority_baseline = max(n_pos, len(y) - n_pos) / len(y)
        entry = {
            "n_pos": n_pos, "n_total": len(y),
            "acc_sae": clf["acc_sae"], "majority_baseline": majority_baseline,
            "delta_pts": 100 * (clf["acc_sae"] - majority_baseline),
        }
        ref = REFERENCE.get(col)
        if ref:
            two_prop = two_proportion_test(n_pos, len(y), ref["n_pos"], len(y))
            entry["n_pos_vs_reference"] = {
                "n_pos_reference": ref["n_pos"], "diff": two_prop.diff,
                "z": two_prop.z, "p": two_prop.p,
            }
        probe_results[col] = entry
        print(f"  {col}: n_pos={n_pos}/{len(y)} ({100*n_pos/len(y):.1f}%) acc_SAE={clf['acc_sae']:.4f} "
              f"baseline={majority_baseline:.4f} delta={100*(clf['acc_sae']-majority_baseline):+.1f}pts")

    dataset_mod.INTENT_KEYWORDS_FR = ORIG_PATTERNS

    out = {"corpus_level_impact_v2": corpus_impact, "probe_results_v2": probe_results}
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Écrit : {OUT_PATH}")


if __name__ == "__main__":
    main()
