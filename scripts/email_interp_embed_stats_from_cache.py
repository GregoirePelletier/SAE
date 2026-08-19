"""Recalcule les stats du job 44413 (email_interp_embed_encode_test.py) à
partir du pickle déjà encodé (local_data/email_interp_embed/encoded_emails.pkl),
avec le correctif filter_na_rows() -- pas de GPU nécessaire, l'encodage SAE ne
se relance pas (Dataset.load_from_file(resume=False) ne recalcule rien,
compute_activations=False). CPU-only.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, "/home/h21486/SAE/external/interp_embed")

import numpy as np

OUT_DIR = Path("/home/h21486/SAE/local_data/email_interp_embed")
TOP_K_FEATURES_TO_INSPECT = 15
TOP_DOCS_PER_FEATURE = 5


def main():
    from interp_embed import Dataset

    print("Loading cached dataset (no GPU, no re-encoding)...", flush=True)
    ds = Dataset.load_from_file(str(OUT_DIR / "encoded_emails.pkl"), resume=False)
    n_before = len(ds)
    ds = ds.filter_na_rows()
    n_dropped = n_before - len(ds)
    print(f"{n_before} emails loaded, {n_dropped} failed to encode (dropped via filter_na_rows).", flush=True)

    binarized = ds.latents("binarize")
    freq = binarized.sum(axis=0) / binarized.shape[0]
    labels = ds.feature_labels()
    d_sae = binarized.shape[1]

    active_feature_ids = np.where(freq > 0)[0]
    labeled_active = [f for f in active_feature_ids if f in labels]
    l0_per_doc = binarized.sum(axis=1)

    print(f"\n=== STATS DESCRIPTIVES (corrigées) ===", flush=True)
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
        "n_emails": int(n_before), "n_dropped_encoding_failures": int(n_dropped),
        "d_sae": int(d_sae),
        "n_active_features": int(len(active_feature_ids)),
        "n_labeled_active_features": int(len(labeled_active)),
        "pct_active": float(100 * len(active_feature_ids) / d_sae),
        "pct_active_labeled": float(100 * len(labeled_active) / max(1, len(active_feature_ids))),
        "l0_mean": float(l0_per_doc.mean()),
        "l0_std": float(l0_per_doc.std()),
        "top_features_inspection": inspection,
    }
    out_path = OUT_DIR / "results_fixed.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
