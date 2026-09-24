"""
scripts/diffing_hypothesis_verification_test.py — App K.1 (arXiv:2512.10092v2,
docs/archive/references/PDF_APPENDICES_EXTRACT.md lignes 340, 740-775) : vérifie par juge LLM
local (Qwen3.8-27B-FP8, JUDGE_MODEL_ID) les hypothèses de diffing SAE déjà
produites (`p1_diff_energy_sports.csv`, labels des features les plus
discriminantes énergie vs sports, `corpus_diff_stats`) contre un corpus frais
energy/sports (`prepare_domain_dataset`, même construction que la section
diffing de `saev5.py` -- pas besoin des MÊMES documents que ceux ayant produit
les labels, seulement du MÊME domaine). Calcule `verification_rate`/`coverage`
(`src/analysis/hypothesis_verifier.py`) -- première mesure de fidélité App K.1
de ce dépôt (rien n'était comparable au papier pour le diffing avant, cf.
docs/archive/audits/AUDIT_SAE_2026-08.md §7).

Usage :
    SAVE_DIR=./results_v10_emails_main/ PYTHONPATH=. \
      .venv/bin/python scripts/diffing_hypothesis_verification_test.py
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
from src.analysis.hypothesis_verifier import verify_hypotheses, compute_verification_metrics
from src.sae.judge import load_judge_model

# Corpus frais, volontairement plus petit que N_TOTAL_ENERGY/SPORTS de saev5.py
# (300 par défaut, dimensionné pour l'entraînement) -- ici seulement pour la
# vérification (coût = n_hypothèses x n_documents appels de génération).
USE_FINEWEB2 = True
N_DOCS_PER_DOMAIN = int(os.environ.get("N_DOCS_PER_DOMAIN", "40"))
TOP_K_HYPOTHESES = int(os.environ.get("TOP_K_HYPOTHESES", "10"))
VERIFICATION_THRESHOLD = float(os.environ.get("VERIFICATION_THRESHOLD", "0.01"))

CACHE_DIR = os.path.join(SAVE_DIR, "cache")
DIFF_CSV = os.path.join(SAVE_DIR, "p1_diff_energy_sports.csv")
OUT_PATH = os.path.join(CACHE_DIR, "diffing_hypothesis_verification.json")


def main():
    diff_df = pd.read_csv(DIFF_CSV)
    # corpus_diff_stats(group_mask=energy_mask) -> freq_A=énergie, freq_B=sports
    # (saev5.py, section diffing cross-domaine) : ne garder que le sens
    # "plus fréquent en énergie", significatif (BH), trié par q croissant.
    energy_hyp_df = diff_df[(diff_df["significant"] == True) & (diff_df["freq_A"] > diff_df["freq_B"])] \
        .sort_values("q").head(TOP_K_HYPOTHESES)
    hypotheses = energy_hyp_df["label"].tolist()
    print(f"[diffing-verif] {len(hypotheses)} hypothèses (top {TOP_K_HYPOTHESES} par q, sens énergie>sports) "
          f"depuis {DIFF_CSV}", flush=True)
    for h in hypotheses:
        print(f"    - {h}", flush=True)

    print("[diffing-verif] Reconstruction corpus energy/sports frais (prepare_domain_dataset)...", flush=True)
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
    print(f"[diffing-verif] {len(energy_texts)} docs energy, {len(sports_texts)} docs sports", flush=True)

    print("[diffing-verif] Chargement du juge...", flush=True)
    model, tokenizer = load_judge_model()

    print(f"[diffing-verif] Vérification {len(hypotheses)}x{len(documents)}...", flush=True)
    matrix = verify_hypotheses(model, tokenizer, hypotheses, documents)

    per_hyp, summary = compute_verification_metrics(
        matrix, group_mask, hypothesis_labels=hypotheses, threshold=VERIFICATION_THRESHOLD,
    )
    print("\n" + "=" * 60, flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v}", flush=True)
    print(per_hyp.to_string(), flush=True)

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "per_hypothesis": per_hyp.to_dict(orient="records"),
            "matrix": matrix.tolist(),
            "documents": documents,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[+] {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
