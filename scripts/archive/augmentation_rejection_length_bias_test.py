"""Teste l'hypothèse B.9 (AUDIT_SAE_2026-08.md) : le prompt d'augmentation
tronqué à 2048 tokens ferait perdre des faits aux mails longs (comparaison
faite par `validate()` contre le parent COMPLET), sous-représentant les
mails longs parmi les variantes acceptées. Corrèle longueur du mail parent
et rejet (global, puis restreint à `facts_lost` sur les axes non-orthographe,
seuls concernés par ce garde-fou) sur les manifests d'augmentation déjà
produits -- aucun calcul GPU, lecture de fichiers existants uniquement.

Convention CLAUDE.md : produit un JSON -> section numérotée RESULTS_TESTS.md.
"""
import glob
import json
import os

import pandas as pd
from scipy.stats import spearmanr

from src.data.preparation import load_and_clean_emails

MAILS_PATH = os.environ.get("LOCAL_MAILS_PATH", "local_data/emails/Mails.tsv")
MANIFEST_GLOB = os.environ.get(
    "AUGMENTED_MANIFEST_GLOB", "local_data/emails/archive/augmented_mails_shard*of8_manifest.parquet"
)
EXTRA_MANIFEST = os.environ.get(
    "AUGMENTED_TEST_MANIFEST", "local_data/emails/archive/augmented_mails_test_manifest.parquet"
)
OUT_PATH = os.environ.get(
    "OUT_JSON", "local_data/augmentation_rejection_length_bias_results.json"
)
# ~4 caractères/token en français (approximation grossière, cohérente avec le
# reste du dépôt) -> 2048 tokens ~ 8192 caractères, seuil de troncature du prompt.
CHARS_PER_TOKEN_APPROX = 4
PROMPT_TRUNCATION_TOKENS = 2048


def main():
    paths = sorted(glob.glob(MANIFEST_GLOB))
    if os.path.exists(EXTRA_MANIFEST):
        paths.append(EXTRA_MANIFEST)
    if not paths:
        raise FileNotFoundError(f"Aucun manifest trouvé ({MANIFEST_GLOB!r})")

    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df["is_rejected"] = df["rejected"].notna()
    df["is_facts_lost"] = df["rejected"].fillna("").str.startswith("facts_lost")

    real_texts, _ = load_and_clean_emails(MAILS_PATH)
    parent_len = {i: len(t) for i, t in enumerate(real_texts)}
    df["parent_idx"] = df["parent_id"].astype(int)
    df = df[df["parent_idx"] < len(real_texts)].copy()
    df["parent_len_chars"] = df["parent_idx"].map(parent_len)

    truncation_chars = PROMPT_TRUNCATION_TOKENS * CHARS_PER_TOKEN_APPROX
    df["exceeds_truncation_approx"] = df["parent_len_chars"] > truncation_chars

    rho_all, p_all = spearmanr(df["parent_len_chars"], df["is_rejected"].astype(int))
    quartile_rates = (
        pd.qcut(df["parent_len_chars"], 4, labels=["Q1_court", "Q2", "Q3", "Q4_long"])
        .pipe(lambda q: df.groupby(q, observed=True)["is_rejected"].mean())
        .to_dict()
    )

    non_ortho = df[df["axis"] != "orthographe"]
    rho_facts, p_facts = spearmanr(non_ortho["parent_len_chars"], non_ortho["is_facts_lost"].astype(int))
    n_mails_exceeding = int(non_ortho.groupby("parent_idx")["exceeds_truncation_approx"].first().sum())
    n_mails_total = int(non_ortho["parent_idx"].nunique())
    facts_lost_by_truncation = (
        non_ortho.groupby("exceeds_truncation_approx")["is_facts_lost"].agg(["mean", "count"]).to_dict()
    )

    results = {
        "n_variants_total": int(len(df)),
        "n_parents": int(df["parent_idx"].nunique()),
        "spearman_parent_len_vs_rejection": {"rho": float(rho_all), "p": float(p_all), "n": int(len(df))},
        "rejection_rate_by_length_quartile": {str(k): float(v) for k, v in quartile_rates.items()},
        "spearman_parent_len_vs_facts_lost_non_orthographe": {
            "rho": float(rho_facts), "p": float(p_facts), "n": int(len(non_ortho)),
        },
        "n_parent_mails_exceeding_prompt_truncation_approx": n_mails_exceeding,
        "n_parent_mails_total": n_mails_total,
        "facts_lost_rate_by_truncation_bucket": facts_lost_by_truncation,
        "truncation_threshold_chars_approx": truncation_chars,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nÉcrit : {OUT_PATH}")


if __name__ == "__main__":
    main()
