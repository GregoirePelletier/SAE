"""
scripts/diffing_structured_hypotheses_test.py — App D.2 (arXiv:2512.10092v2,
docs/PDF_APPENDICES_EXTRACT.md lignes 212-266) : génère des hypothèses de
diffing STRUCTURÉES (JSON, prompt verbatim) à partir des features SAE les
plus discriminantes déjà calculées (`p1_diff_energy_sports.csv`,
`corpus_diff_stats`), sélectionnées par différence de FRÉQUENCE (App D.2 :
top 200, seuil 0.03 -- `select_top_diff_features_by_frequency`, distinct du
tri par log-odds-ratio utilisé ailleurs). Vérifie ensuite ces hypothèses
structurées (leur champ `description`) par juge LLM sur un corpus
energy/sports frais (`hypothesis_verifier.verify_hypotheses`, App K.1) --
mesure `verification_rate`/`coverage` sur des hypothèses GÉNÉRÉES selon le
protocole du papier, pas sur les labels de features bruts comme
`scripts/diffing_hypothesis_verification_test.py` (§84, RESULTS_TESTS.md) :
comble le dernier écart identifié entre les deux ("§84 a mesuré
verification_rate/coverage mais sur des hypothèses SAE non structurées, pas
sur la génération App D.2 elle-même").

Écart de protocole documenté (R6) : pas de ré-extraction de fragments
token-level pour illustrer chaque feature d'un exemple positif/négatif marqué
(<< >>) -- seul le label déjà produit par le juge (odd-one-out, App C) est
transmis au prompt de génération. Le champ `description` de chaque hypothèse
structurée reste vérifiable indépendamment (App K.1), l'absence d'exemples
illustrés dans le prompt de génération affecte potentiellement la PRÉCISION
des hypothèses générées, pas la validité de leur vérification en aval.

Usage :
    SAVE_DIR=./results_v10_emails_main/ PYTHONPATH=. \
      .venv/bin/python scripts/diffing_structured_hypotheses_test.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import SAVE_DIR, LOCAL_DATASET_PATH
from src.data.keywords import ENERGY_KEYWORDS, SPORTS_KEYWORDS, ENERGY_URL_PATTERNS, SPORTS_URL_PATTERNS
from src.data.preparation import prepare_domain_dataset
from src.analysis.cooccurrence import select_top_diff_features_by_frequency
from src.analysis.diff_hypothesis_generator import generate_structured_diff_hypotheses
from src.analysis.hypothesis_verifier import verify_hypotheses, compute_verification_metrics
from src.sae.judge import load_judge_model

FREQ_DIFF_THRESHOLD = float(os.environ.get("FREQ_DIFF_THRESHOLD", "0.03"))   # App D.2
TOP_N_FEATURES = int(os.environ.get("TOP_N_FEATURES", "200"))                # App D.2
NUM_HYPOTHESES = int(os.environ.get("NUM_HYPOTHESES", "10"))
N_DOCS_PER_DOMAIN = int(os.environ.get("N_DOCS_PER_DOMAIN", "40"))
VERIFICATION_THRESHOLD = float(os.environ.get("VERIFICATION_THRESHOLD", "0.01"))
USE_FINEWEB2 = True

CACHE_DIR = os.path.join(SAVE_DIR, "cache")
DIFF_CSV = os.path.join(SAVE_DIR, "p1_diff_energy_sports.csv")
OUT_PATH = os.path.join(CACHE_DIR, "diffing_structured_hypotheses.json")


def main():
    diff_df = pd.read_csv(DIFF_CSV)
    selected = select_top_diff_features_by_frequency(diff_df, threshold=FREQ_DIFF_THRESHOLD, top_n=TOP_N_FEATURES)
    print(f"[diff-structured] {len(selected)}/{len(diff_df)} features au-dessus du seuil "
          f"de différence de fréquence {FREQ_DIFF_THRESHOLD} (App D.2, top {TOP_N_FEATURES}).", flush=True)
    if len(selected) == 0:
        print("[diff-structured] Aucune feature au-dessus du seuil -- rien à générer.", flush=True)
        return

    # target=energie (freq_A), other=sports (freq_B) -- même convention que
    # corpus_diff_stats(group_mask=energy_mask) dans saev5.py (section diffing).
    features = [
        {
            "feature_id": int(row["feature_id"]),
            "label": row["label"],
            "percentage_difference": float(row["freq_diff"]),
        }
        for _, row in selected.iterrows()
    ]

    print("[diff-structured] Chargement du juge...", flush=True)
    model, tokenizer = load_judge_model()

    print(f"[diff-structured] Génération d'au plus {NUM_HYPOTHESES} hypothèses structurées "
          f"(App D.2, un seul appel LLM sur {len(features)} features)...", flush=True)
    hypotheses = generate_structured_diff_hypotheses(
        model, tokenizer, features,
        query="What distinguishes the energy domain (target) from the sports domain (other)?",
        num_hypotheses=NUM_HYPOTHESES,
    )
    print(f"[diff-structured] {len(hypotheses)} hypothèses structurées générées.", flush=True)
    for h in hypotheses:
        print(f"    - [{h['dataset']}, diff={h['percentage_difference']:+.2f}, "
              f"conf={h['confidence']:.2f}] {h['description']}", flush=True)

    if not hypotheses:
        print("[diff-structured] Aucune hypothèse valide parsée -- arrêt avant vérification.", flush=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump({"features_considered": features, "hypotheses": [], "verification": None}, f,
                      indent=2, ensure_ascii=False)
        return

    print("[diff-structured] Reconstruction corpus energy/sports frais (prepare_domain_dataset)...", flush=True)
    energy_texts = prepare_domain_dataset(
        ENERGY_KEYWORDS, "energy", N_DOCS_PER_DOMAIN,
        chunk_length=1024, max_chunks=20, url_patterns=ENERGY_URL_PATTERNS,
        local_dataset_path=LOCAL_DATASET_PATH, use_fineweb2=USE_FINEWEB2,
    )
    sports_texts = prepare_domain_dataset(
        SPORTS_KEYWORDS, "sports", N_DOCS_PER_DOMAIN,
        chunk_length=1024, max_chunks=20, url_patterns=SPORTS_URL_PATTERNS,
        local_dataset_path=LOCAL_DATASET_PATH, use_fineweb2=USE_FINEWEB2,
    )
    documents = energy_texts + sports_texts
    group_mask = np.array([True] * len(energy_texts) + [False] * len(sports_texts))
    print(f"[diff-structured] {len(energy_texts)} docs energy, {len(sports_texts)} docs sports", flush=True)

    # Seul le champ "description" (l'énoncé vérifiable de l'hypothèse, App D.2)
    # est passé à verify_hypotheses -- pas l'objet JSON complet.
    hypothesis_texts = [h["description"] for h in hypotheses]
    print(f"[diff-structured] Vérification {len(hypothesis_texts)}x{len(documents)}...", flush=True)
    matrix = verify_hypotheses(model, tokenizer, hypothesis_texts, documents)

    per_hyp, summary = compute_verification_metrics(
        matrix, group_mask, hypothesis_labels=hypothesis_texts, threshold=VERIFICATION_THRESHOLD,
    )
    print("\n" + "=" * 60, flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v}", flush=True)
    print(per_hyp.to_string(), flush=True)

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "features_considered": features,
            "hypotheses": hypotheses,
            "verification": {
                "summary": summary,
                "per_hypothesis": per_hyp.to_dict(orient="records"),
            },
            "documents": documents,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[+] {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
