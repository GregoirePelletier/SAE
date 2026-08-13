"""
scripts/intent_urgency_probe.py — Les codes latents du SAE prédisent-ils l'urgence et
l'intention (labels faibles par regex, src/data/dataset.py::INTENT_KEYWORDS_FR) sur les
MAILS ORIGINAUX (pas les variantes augmentées) ?

Objectif du projet : "détection d'urgence", "détection d'intentions". Le
probe de classification déjà ajouté (RESULTS_TESTS.md §12) mesure la
séparabilité des AXES D'AUGMENTATION synthétiques (colère, urgence simulée,
etc.) -- ce script mesure la même chose mais sur les intentions/urgence
RÉELLES des mails originaux (détectées par regex sur le texte brut,
indépendamment de toute perturbation artificielle), un test plus proche du
cas d'usage final.

Zéro calcul GPU : réutilise les activations déjà mises en cache
(p1_all_doc_acts_ext_d1024.pt).

Usage :
    PYTHONPATH=. .venv/bin/python scripts/intent_urgency_probe.py
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import torch

from src.config import LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, SAVE_DIR, CORPUS_SPLIT_SEED
from src.data.dataset import load_mails_tsv
from src.data.preparation import build_email_train_test_corpus
from src.analysis.metrics import downstream_classification

CACHE_DIR = os.path.join(SAVE_DIR, "cache")
SEED = 42


def replicate_load_and_clean_emails_with_index(tsv_path: str) -> list[int]:
    """Reproduit EXACTEMENT le filtrage de src/data/preparation.py::load_and_clean_emails
    (même regex, même ordre) mais retourne les index de ligne du DataFrame source
    plutôt que les textes -- nécessaire pour rejoindre les colonnes intent_* de
    load_mails_tsv (qui ne survivent pas au renommage/filtrage de load_and_clean_emails)."""
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


def main():
    print("[intent-probe] Reconstruction de la correspondance texte réel <-> ligne Mails.tsv...")
    kept_row_indices, df_full = replicate_load_and_clean_emails_with_index(LOCAL_MAILS_PATH)
    n_real = len(kept_row_indices)
    print(f"[intent-probe] n_real (mails originaux survivant au nettoyage) = {n_real}")

    print("[intent-probe] Reconstruction du split train/test (déterministe, SEED fixe)...")
    rng = np.random.default_rng(SEED)
    test_mask = rng.random(n_real) < float(os.environ.get("EMAIL_TEST_SPLIT", "0.05"))
    train_positions = [i for i in range(n_real) if not test_mask[i]]  # ordre croissant, cf. build_email_train_test_corpus
    k_train_original = len(train_positions)
    print(f"[intent-probe] {k_train_original} mails originaux dans le bloc 'original' de train_texts.")

    # Vérification croisée : build_email_train_test_corpus doit annoncer le même compte.
    train_texts, train_labels, _, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED,
    )
    n_original_train = sum(1 for l in train_labels if l == "original")
    assert n_original_train == k_train_original, (
        f"Incohérence : {n_original_train} (build_email_train_test_corpus) != "
        f"{k_train_original} (reconstruction locale) -- ne pas poursuivre, la "
        f"correspondance activations<->labels serait fausse."
    )
    print("[intent-probe] Correspondance vérifiée (comptes identiques).")

    acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts_ext_d1024.pt")
    if not os.path.exists(acts_path):
        acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts.pt")
    print(f"[intent-probe] Chargement des activations : {acts_path}")
    all_doc_acts = torch.load(acts_path, map_location="cpu", weights_only=True)
    original_train_acts = all_doc_acts[:k_train_original]  # cf. justification dans le docstring du module

    # Labels intent_* alignés : train_positions[j] est l'index dans real_texts (0..n_real-1)
    # du j-ième mail 'original' de train ; kept_row_indices[.] convertit vers la ligne Mails.tsv.
    row_indices_for_train_original = [kept_row_indices[i] for i in train_positions]
    intent_cols = [c for c in df_full.columns if c.startswith("intent_")]
    labels_df = df_full.loc[row_indices_for_train_original, intent_cols].reset_index(drop=True)

    print(f"\n[intent-probe] {len(intent_cols)} intentions testées sur {k_train_original} mails originaux "
          f"(activations SAE Pipeline 1, {original_train_acts.shape[1]} dims) :\n")

    results = {}
    for col in intent_cols:
        y = labels_df[col].to_numpy()
        n_pos = int(y.sum())
        if n_pos < 10 or n_pos > len(y) - 10:
            print(f"  {col}: {n_pos}/{len(y)} positifs -- ignoré (classe trop déséquilibrée pour un probe fiable).")
            continue
        pos_mask = torch.from_numpy(y.astype(bool))
        try:
            clf = downstream_classification(acts_by_label={
                "positif": original_train_acts[pos_mask],
                "negatif": original_train_acts[~pos_mask],
            })
        except Exception as e:
            print(f"  {col}: WARN échec ({e})")
            continue
        majority_baseline = max(n_pos, len(y) - n_pos) / len(y)
        results[col] = {
            "n_pos": n_pos, "n_total": len(y),
            "acc_sae": clf["acc_sae"], "majority_baseline": majority_baseline,
        }
        print(f"  {col}: {n_pos}/{len(y)} positifs ({100*n_pos/len(y):.1f}%) | "
              f"acc_SAE={clf['acc_sae']:.4f} | baseline (classe majoritaire)={majority_baseline:.4f} | "
              f"Δ={clf['acc_sae']-majority_baseline:+.4f}")

    import json
    out_path = os.path.join(CACHE_DIR, "intent_urgency_probe_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Écrit : {out_path}")


if __name__ == "__main__":
    main()
