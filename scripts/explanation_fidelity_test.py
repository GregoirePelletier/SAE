"""
scripts/explanation_fidelity_test.py — Fidélité de l'explication document-level, par
ablation (deletion metric, classique en explicabilité : Samek et al. 2017, "Evaluating
the Visualization of What a Deep Neural Network Has Learned").

Question : pour un mail donné, les features SAE désignées comme "expliquant" sa
classification (ex. urgence) sont-elles vraiment celles qui pilotent la décision de la
sonde, ou juste des labels plausibles sans lien causal réel ?

Protocole :
  1. Ajuste UNE sonde logistique finale (pas la CV de downstream_classification, qui ne
     réexpose pas les coefficients) sur les activations SAE des mails réels ("original",
     cf. scripts/intent_urgency_probe.py pour la correspondance) pour chaque intention
     testée (urgence, réclamation...).
  2. Pour un échantillon de documents correctement classés positifs avec confiance
     élevée : calcule la contribution de chaque feature à la décision
     (coef_i * activation_i), identifie le top-K des features qui poussent le plus vers
     "positif" -- c'est l'"explication" que le pipeline produirait pour ce document.
  3. Ablate (met à zéro) ces top-K features, mesure la chute de probabilité prédite.
  4. Compare à deux témoins : ablation de K features actives choisies au hasard, et
     ablation des K features actives les MOINS contributives (bottom-K). Si
     l'explication est fidèle, l'ablation du top-K doit faire chuter la probabilité
     nettement plus que les deux témoins.

Zéro calcul GPU : réutilise les activations déjà en cache (résultats_v10_emails_main).

Usage : PYTHONPATH=. .venv/bin/python scripts/explanation_fidelity_test.py
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from src.config import LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, SAVE_DIR, NEURONPEDIA_LABELS_PATH
from src.data.dataset import load_mails_tsv
from src.data.preparation import build_email_train_test_corpus

CACHE_DIR = os.path.join(SAVE_DIR, "cache")
SEED = 42
TOP_K = 10          # nombre de features "expliquant" la décision
N_DOCS_SAMPLE = 200  # nombre de documents positifs testés par intention
rng_np = np.random.default_rng(SEED)


def replicate_load_and_clean_emails_with_index(tsv_path: str):
    """Identique à scripts/intent_urgency_probe.py -- dupliqué ici pour garder ce
    script autonome (pas de dépendance croisée entre scripts de diagnostic)."""
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


def load_real_email_acts_and_intents():
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
    assert n_original_train == k_train_original, "Incohérence de correspondance -- cf. intent_urgency_probe.py"

    acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts_ext_d1024.pt")
    if not os.path.exists(acts_path):
        acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts.pt")
    all_doc_acts = torch.load(acts_path, map_location="cpu", weights_only=True)
    original_train_acts = all_doc_acts[:k_train_original].float().numpy()

    row_indices_for_train_original = [kept_row_indices[i] for i in train_positions]
    intent_cols = [c for c in df_full.columns if c.startswith("intent_")]
    labels_df = df_full.loc[row_indices_for_train_original, intent_cols].reset_index(drop=True)
    real_texts = [train_texts[i] for i in range(k_train_original)]  # bloc 'original' = préfixe de train_texts
    return original_train_acts, labels_df, real_texts


def load_label_map():
    # Dérivé de SAE_ID (cf. commentaire équivalent dans explanation_plausibility_test.py) :
    # ici les labels ne servent qu'à l'annotation cosmétique des features dans les
    # exemples exportés (le calcul de fidélité ablate par INDEX, indépendant du texte
    # du label), mais un mauvais mapping produirait quand même des labels d'exemple
    # trompeurs pour un run n'utilisant pas la largeur 16k.
    with open(NEURONPEDIA_LABELS_PATH) as f:
        labels_core = {int(k): v for k, v in json.load(f).items()}
    judge_path = os.path.join(CACHE_DIR, "p1_judge_labels_extended.json")
    label_map = dict(labels_core)
    if os.path.exists(judge_path):
        with open(judge_path) as f:
            judge_ext = json.load(f)
        for k, v in judge_ext.items():
            label_map[int(k)] = "[EXT] " + v.get("label", f"F{k}")
    return label_map


def ablation_drop(clf: LogisticRegression, x: np.ndarray, feature_idx: np.ndarray) -> float:
    """Probabilité prédite avant/après mise à zéro des features données."""
    p_before = clf.predict_proba(x.reshape(1, -1))[0, 1]
    x_ablated = x.copy()
    x_ablated[feature_idx] = 0.0
    p_after = clf.predict_proba(x_ablated.reshape(1, -1))[0, 1]
    return float(p_before - p_after), float(p_before)


def main():
    print("[fidelity] Chargement des activations et labels d'intention (mails réels)...")
    acts, labels_df, texts = load_real_email_acts_and_intents()
    label_map = load_label_map()
    print(f"[fidelity] {acts.shape[0]} mails réels, {acts.shape[1]} dims SAE.")

    results = {}
    for col in labels_df.columns:
        y = labels_df[col].to_numpy().astype(int)
        n_pos = int(y.sum())
        if n_pos < 30 or n_pos > len(y) - 30:
            print(f"[fidelity] {col}: {n_pos}/{len(y)} positifs -- ignoré (trop déséquilibré).")
            continue

        clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")
        clf.fit(acts, y)
        coef = clf.coef_[0]  # (d_sae,) contribution par unité d'activation

        probs = clf.predict_proba(acts)[:, 1]
        # Échantillon : documents réellement positifs, bien classés (proba > 0.7)
        candidate_idx = np.where((y == 1) & (probs > 0.7))[0]
        if len(candidate_idx) == 0:
            print(f"[fidelity] {col}: aucun document positif classé avec confiance -- ignoré.")
            continue
        sample_idx = rng_np.choice(candidate_idx, size=min(N_DOCS_SAMPLE, len(candidate_idx)), replace=False)

        drops_top, drops_random, drops_bottom = [], [], []
        example_explanations = []
        for i in sample_idx:
            x = acts[i]
            active = np.where(x > 1e-6)[0]
            if len(active) < TOP_K * 2:
                continue
            contributions = coef[active] * x[active]  # contribution positive = pousse vers "positif"
            order = np.argsort(contributions)[::-1]
            top_feats = active[order[:TOP_K]]
            bottom_feats = active[order[-TOP_K:]]
            random_feats = rng_np.choice(active, size=TOP_K, replace=False)

            d_top, p0 = ablation_drop(clf, x, top_feats)
            d_random, _ = ablation_drop(clf, x, random_feats)
            d_bottom, _ = ablation_drop(clf, x, bottom_feats)
            drops_top.append(d_top)
            drops_random.append(d_random)
            drops_bottom.append(d_bottom)

            if len(example_explanations) < 5:
                example_explanations.append({
                    "doc_idx": int(i),
                    "text_preview": texts[i][:150],
                    "p_before": p0,
                    "drop_top_k": d_top,
                    "drop_random_k": d_random,
                    "top_features": [{"f": int(f), "label": label_map.get(int(f), f"F{f}")} for f in top_feats],
                })

        if not drops_top:
            print(f"[fidelity] {col}: aucun document exploitable (pas assez de features actives).")
            continue

        results[col] = {
            "n_docs_tested": len(drops_top),
            "mean_drop_top_k": float(np.mean(drops_top)),
            "mean_drop_random_k": float(np.mean(drops_random)),
            "mean_drop_bottom_k": float(np.mean(drops_bottom)),
            "fidelity_ratio_top_vs_random": float(np.mean(drops_top) / max(np.mean(drops_random), 1e-6)),
            "examples": example_explanations,
        }
        print(f"[fidelity] {col} (n={len(drops_top)}): "
              f"chute top-{TOP_K}={np.mean(drops_top):.4f} | "
              f"chute random-{TOP_K}={np.mean(drops_random):.4f} | "
              f"chute bottom-{TOP_K}={np.mean(drops_bottom):.4f} | "
              f"ratio top/random={np.mean(drops_top)/max(np.mean(drops_random),1e-6):.2f}x")

    out_path = os.path.join(CACHE_DIR, "explanation_fidelity_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Écrit : {out_path}")


if __name__ == "__main__":
    main()
