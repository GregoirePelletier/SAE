"""
scripts/post_stage/e01_compare_representations.py -- Comparaison CORE/FULL/
DENSE/TFIDF sur DEV, protocole FIT-train/DEV-eval strict (Plan_execution_
SAE_15_jours_Claude_Code.md §6.3-6.4). Entierement CPU sauf l'encodage
DENSE (bge-m3) -- lance via sbatch (lecture de p1_all_doc_acts_ext_d1024.pt,
plusieurs Go, jamais sur le noeud frontal).

Usage (voir slurm/post_stage/02_e01_compare_representations.slurm) :
    .venv/bin/python -u scripts/post_stage/e01_compare_representations.py \\
        --save-dir results_post_stage_e01_fit_1b_layer13_k5 \\
        --split-assignments configs/post_stage/split_assignments.json \\
        --d-extra 1024 --out e01_representation_comparison.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy import sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.post_stage.dataset_contract import load_fit_dev_corpus_from_manifest  # noqa: E402
from src.post_stage.representations import (  # noqa: E402
    build_tfidf_representation, embed_bge_m3_documents, TfidfConfig,
)
from src.analysis.metrics import held_out_probe_accuracy  # noqa: E402
from src.analysis.stats import bootstrap_ci_by_group, paired_mcnemar_test, fdr_bh  # noqa: E402
from src.sae.sae_shared import load_all_doc_acts  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--mails-tsv-path", default="local_data/emails/Mails.tsv")
    ap.add_argument("--augmented-jsonl-path", default="local_data/emails/augmented_mails.jsonl")
    ap.add_argument("--split-assignments", required=True)
    ap.add_argument("--d-extra", type=int, required=True)
    ap.add_argument("--max-augmented-per-mail", type=int, default=13)
    ap.add_argument("--dense-model-path", default="./models/bge-m3")
    ap.add_argument("--dense-max-length", type=int, default=2048)
    ap.add_argument("--min-class-support", type=int, default=10)
    ap.add_argument("--out", default="e01_representation_comparison.json")
    args = ap.parse_args()

    t0 = time.time()
    print("[e01_compare] Chargement corpus FIT/DEV gelé...", flush=True)
    fit_texts, fit_labels, dev_texts, dev_labels, fit_groups, dev_groups = (
        load_fit_dev_corpus_from_manifest(
            args.split_assignments, args.mails_tsv_path, args.augmented_jsonl_path,
            max_augmented_per_mail=args.max_augmented_per_mail, return_groups=True,
        )
    )
    n_fit, n_dev = len(fit_texts), len(dev_texts)
    print(f"[e01_compare] FIT={n_fit} DEV={n_dev}", flush=True)

    # Filtre aux classes avec assez de support (même règle que saev5.py, §6.4) --
    # calculé sur FIT (l'entraînement), pas sur DEV.
    label_counts = pd.Series(fit_labels).value_counts()
    usable_labels = set(label_counts[label_counts >= args.min_class_support].index.tolist())
    fit_mask = np.array([lbl in usable_labels for lbl in fit_labels])
    dev_mask = np.array([lbl in usable_labels for lbl in dev_labels])
    print(f"[e01_compare] {len(usable_labels)} classes usables (support FIT >= "
          f"{args.min_class_support}) -- {fit_mask.sum()}/{n_fit} FIT, "
          f"{dev_mask.sum()}/{n_dev} DEV retenus.", flush=True)

    fit_texts_f = [t for t, m in zip(fit_texts, fit_mask) if m]
    dev_texts_f = [t for t, m in zip(dev_texts, dev_mask) if m]
    y_fit = np.array([l for l, m in zip(fit_labels, fit_mask) if m])
    y_dev = np.array([l for l, m in zip(dev_labels, dev_mask) if m])
    groups_dev_f = np.array([g for g, m in zip(dev_groups, dev_mask) if m])

    # ─── CORE / FULL (depuis le SAE déjà entraîné) ─────────────────────────
    acts_path = os.path.join(args.save_dir, "cache", f"p1_all_doc_acts_ext_d{args.d_extra}.pt")
    print(f"[e01_compare] Chargement {acts_path} (dense, plusieurs Go)...", flush=True)
    all_doc_acts = load_all_doc_acts(acts_path)
    d_total = all_doc_acts.shape[1]
    d_core = d_total - args.d_extra
    assert all_doc_acts.shape[0] == n_fit + n_dev, (
        f"all_doc_acts a {all_doc_acts.shape[0]} lignes, attendu {n_fit + n_dev} "
        "(FIT+DEV, sans filler ni diff pour ce run E01) -- ordre/format incompatible."
    )
    fit_acts = all_doc_acts[:n_fit][fit_mask].numpy()
    dev_acts = all_doc_acts[n_fit:n_fit + n_dev][dev_mask].numpy()
    del all_doc_acts

    results = {}
    for name, cols in (("core", slice(0, d_core)), ("full", slice(0, d_total))):
        X_fit = sp.csr_matrix(fit_acts[:, cols])
        X_dev = sp.csr_matrix(dev_acts[:, cols])
        res = held_out_probe_accuracy(X_fit, y_fit, X_dev, y_dev)
        results[name] = res
        print(f"[e01_compare] acc_{name} = {res['accuracy']:.4f}", flush=True)

    # ─── TFIDF ──────────────────────────────────────────────────────────────
    print("[e01_compare] TFIDF (fit FIT uniquement)...", flush=True)
    _, X_tfidf_by_split, tfidf_config = build_tfidf_representation(
        fit_texts_f, {"dev": dev_texts_f}, config=TfidfConfig()
    )
    res_tfidf = held_out_probe_accuracy(X_tfidf_by_split["fit"], y_fit, X_tfidf_by_split["dev"], y_dev)
    results["tfidf"] = res_tfidf
    print(f"[e01_compare] acc_tfidf = {res_tfidf['accuracy']:.4f}", flush=True)

    # ─── DENSE (bge-m3) ─────────────────────────────────────────────────────
    print("[e01_compare] DENSE (bge-m3)...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_fit, dense_config = embed_bge_m3_documents(
        fit_texts_f, model_path=args.dense_model_path, max_length=args.dense_max_length, device=device,
    )
    emb_dev, _ = embed_bge_m3_documents(
        dev_texts_f, model_path=args.dense_model_path, max_length=args.dense_max_length, device=device,
    )
    res_dense = held_out_probe_accuracy(emb_fit.numpy(), y_fit, emb_dev.numpy(), y_dev)
    results["dense"] = res_dense
    print(f"[e01_compare] acc_dense = {res_dense['accuracy']:.4f} "
          f"(taux de troncature FIT={dense_config['truncation_rate']:.3f})", flush=True)

    # ─── Comparaisons appariées FULL-CORE / FULL-DENSE / FULL-TFIDF ────────
    correct_full = results["full"]["correct"]
    comparisons = {}
    pvalues = []
    for other in ("core", "dense", "tfidf"):
        correct_other = results[other]["correct"]
        b = int(np.sum(correct_full & ~correct_other))
        c = int(np.sum(~correct_full & correct_other))
        mcnemar = paired_mcnemar_test(b, c)
        diff_values = correct_full.astype(float) - correct_other.astype(float)
        ci = bootstrap_ci_by_group(diff_values, groups_dev_f, statistic=np.mean, n_boot=2000, seed=42)
        comparisons[f"full_minus_{other}"] = {
            "acc_full": results["full"]["accuracy"], f"acc_{other}": results[other]["accuracy"],
            "diff": results["full"]["accuracy"] - results[other]["accuracy"],
            "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": mcnemar.p, "mcnemar_exact": mcnemar.exact,
            "bootstrap_ci_low": ci.ci_low, "bootstrap_ci_high": ci.ci_high, "bootstrap_n_groups": ci.n_groups,
        }
        pvalues.append(mcnemar.p)

    p_adjusted = fdr_bh(pvalues)
    for (key, p_adj) in zip(comparisons.keys(), p_adjusted):
        comparisons[key]["mcnemar_p_fdr_bh"] = float(p_adj)

    out = {
        "n_fit": int(fit_mask.sum()), "n_dev": int(dev_mask.sum()),
        "n_classes": len(usable_labels), "d_core": d_core, "d_extra": args.d_extra,
        "tfidf_config": tfidf_config, "dense_config": dense_config,
        "accuracies": {k: v["accuracy"] for k, v in results.items()},
        "comparisons": comparisons,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    out_path = os.path.join(args.save_dir, args.out)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp_path, out_path)
    print(f"[e01_compare] Écrit {out_path}", flush=True)
    print(json.dumps(out["comparisons"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
