"""
Application du pipeline interp_embed EXACT (Dataset + GoodfireSAE,
Llama-3.1-8B-Instruct + Goodfire SAE-l19) sur le corpus mails EDF, dans le
prolongement du test movie-genre (même modèle/SAE). Objectif : isoler la
dégradation FR/industriel -- le SAE est entraîné sur LMSYS-Chat-1M (majoritairement
anglais, conversationnel) ; les mails sont en français, domaine facturation/
énergie. Pas de ground-truth ici (contrairement au test genre) -- exploration
descriptive uniquement, via LEUR API (`dataset.latents`, `dataset.feature_labels`,
`dataset.top_documents_for_feature`), aucun juge/scoring nécessaire.

Écarts documentés :
- Échantillon de N_EMAILS mails (pas les 72757 lignes -- borne le coût GPU
  pour un premier passage descriptif).
- Aucune comparaison ground-truth (P1 du plan original : "isole la
  dégradation", pas une éval chiffrée) -- rapporte simplement la fraction de
  labels connus qui s'activent, et des exemples top-documents pour inspection
  humaine.

Sortie : JSON dans local_data/email_interp_embed/results.json.
"""
import sys
import csv
import json
import random
from pathlib import Path

sys.path.insert(0, "/home/h21486/SAE/external/interp_embed")

import numpy as np
import pandas as pd
import torch

OUT_DIR = Path("/home/h21486/SAE/local_data/email_interp_embed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_EMAILS = 1500
TOP_K_FEATURES_TO_INSPECT = 15
TOP_DOCS_PER_FEATURE = 5
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def load_email_sample() -> pd.DataFrame:
    rows = []
    with open("/home/h21486/SAE/local_data/emails/Mails.tsv", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = (row.get("document") or "").strip()
            if text:
                rows.append({"text": text})
    print(f"Total non-empty emails available: {len(rows)}", flush=True)
    sampled = random.sample(rows, min(N_EMAILS, len(rows)))
    return pd.DataFrame(sampled)


def main():
    from interp_embed import Dataset
    from interp_embed.sae.local_sae import GoodfireSAE

    print("Loading email sample...", flush=True)
    df = load_email_sample()
    print(f"Sampled {len(df)} emails.", flush=True)

    print("Loading Llama-3.1-8B-Instruct + Goodfire SAE-l19...", flush=True)
    sae = GoodfireSAE(variant_name="Llama-3.1-8B-Instruct-SAE-l19", device="cuda:0")
    ds = Dataset(
        data=df, sae=sae, field="text",
        dataset_description="edf_emails_interp_embed_encode_test",
        save_path=str(OUT_DIR / "encoded_emails.pkl"),
        batch_size=16,
    )
    torch.cuda.empty_cache()
    n_before = len(ds)
    # Dataset.latents() remplit les lignes échouées (None, ex. edge case
    # d'encodage) avec csr_matrix(np.full(d_sae, np.nan)) -- scipy stocke NaN
    # comme entrée explicite "non-nulle" dans TOUTES les colonnes, donc
    # .sum(axis=0) sur le résultat dense propage NaN à CHAQUE feature, pas
    # seulement aux lignes en échec (bug rencontré : 0/65536 "actives" alors
    # que les activations brutes étaient saines -- diagnostiqué via
    # scripts/diagnose_email_zero_activations.py). filter_na_rows() est leur
    # méthode dédiée pour exactement ce cas.
    ds = ds.filter_na_rows()
    n_dropped = n_before - len(ds)
    print(f"Encoded {n_before} emails, {n_dropped} failed to encode (dropped via filter_na_rows).", flush=True)

    binarized = ds.latents("binarize")  # (n_docs, d_sae), leur API telle quelle
    freq = binarized.sum(axis=0) / binarized.shape[0]
    labels = ds.feature_labels()
    d_sae = binarized.shape[1]

    active_feature_ids = np.where(freq > 0)[0]
    labeled_active = [f for f in active_feature_ids if f in labels]
    l0_per_doc = binarized.sum(axis=1)

    print(f"\n=== STATS DESCRIPTIVES ===", flush=True)
    print(f"d_sae = {d_sae}", flush=True)
    print(f"Features actives (>=1 doc) : {len(active_feature_ids)} / {d_sae} "
          f"({100 * len(active_feature_ids) / d_sae:.2f}%)", flush=True)
    print(f"...dont labellisées (dict {len(labels)} labels) : {len(labeled_active)} "
          f"({100 * len(labeled_active) / max(1, len(active_feature_ids)):.2f}% des actives)", flush=True)
    print(f"L0 moyen par document : {l0_per_doc.mean():.1f} (std {l0_per_doc.std():.1f})", flush=True)

    top_feature_ids = active_feature_ids[np.argsort(freq[active_feature_ids])[::-1]]
    top_feature_ids = [f for f in top_feature_ids if f in labels][:TOP_K_FEATURES_TO_INSPECT]

    inspection = []
    for f_id in top_feature_ids:
        label = labels.get(int(f_id), f"feature_{f_id}")
        # top_documents_for_feature : LEUR méthode telle quelle (App. C / API_REFERENCE.md).
        top_docs = ds.top_documents_for_feature(int(f_id), k=TOP_DOCS_PER_FEATURE)
        print(f"\n--- Feature {f_id} ({freq[f_id]*100:.1f}% des mails) : {label!r} ---", flush=True)
        for doc in top_docs:
            snippet = doc[:300].replace("\n", " ")
            print(f"  {snippet}", flush=True)
        inspection.append({
            "feature_id": int(f_id), "label": label,
            "frequency": float(freq[f_id]), "top_documents": top_docs,
        })

    results = {
        "n_emails": len(ds),
        "d_sae": int(d_sae),
        "n_active_features": int(len(active_feature_ids)),
        "n_labeled_active_features": int(len(labeled_active)),
        "pct_active": float(100 * len(active_feature_ids) / d_sae),
        "pct_active_labeled": float(100 * len(labeled_active) / max(1, len(active_feature_ids))),
        "l0_mean": float(l0_per_doc.mean()),
        "l0_std": float(l0_per_doc.std()),
        "top_features_inspection": inspection,
    }
    out_path = OUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
