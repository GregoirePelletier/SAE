"""
scripts/post_stage/e01_pooling_and_length_analysis.py -- Complète E01 (plan
§6.3) : (a) pooling alternatif top-k-moyenne comparé au max-pooling déjà
utilisé par le run de référence, (b) baseline longueur seule + accuracy par
tranche de longueur pour CORE/FULL/DENSE/TFIDF. Réutilise le checkpoint et
le corpus gelé déjà en place (job 48585) -- aucun réentraînement, CPU
uniquement (relit les fragments token-level déjà écrits par le ré-encodage,
`p1_token_fragments_ext`), lancé via sbatch (fragments + tenseur dense,
plusieurs Go, jamais sur le nœud frontal).
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
from src.analysis.metrics import held_out_probe_accuracy  # noqa: E402
from src.analysis.stats import bootstrap_ci_by_group, paired_mcnemar_test, fdr_bh  # noqa: E402
from src.sae.sae_shared import load_all_doc_acts  # noqa: E402
from src.storage.fragment_store import load_fragment, doc_topk_mean_pool  # noqa: E402


def _usable_label_mask(labels, min_support, usable_labels=None):
    if usable_labels is None:
        counts = pd.Series(labels).value_counts()
        usable_labels = set(counts[counts >= min_support].index.tolist())
    mask = np.array([lbl in usable_labels for lbl in labels])
    return mask, usable_labels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--mails-tsv-path", default="local_data/emails/Mails.tsv")
    ap.add_argument("--augmented-jsonl-path", default="local_data/emails/augmented_mails.jsonl")
    ap.add_argument("--split-assignments", required=True)
    ap.add_argument("--d-extra", type=int, required=True)
    ap.add_argument("--max-augmented-per-mail", type=int, default=13)
    ap.add_argument("--min-class-support", type=int, default=10)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--n-length-strata", type=int, default=4)
    ap.add_argument("--out", default="e01_pooling_and_length_analysis.json")
    args = ap.parse_args()

    t0 = time.time()
    print("[e01_pool_len] Chargement corpus FIT/DEV gelé...", flush=True)
    fit_texts, fit_labels, dev_texts, dev_labels, fit_groups, dev_groups = (
        load_fit_dev_corpus_from_manifest(
            args.split_assignments, args.mails_tsv_path, args.augmented_jsonl_path,
            max_augmented_per_mail=args.max_augmented_per_mail, return_groups=True,
        )
    )
    n_fit, n_dev = len(fit_texts), len(dev_texts)
    print(f"[e01_pool_len] FIT={n_fit} DEV={n_dev}", flush=True)

    fit_mask, usable_labels = _usable_label_mask(fit_labels, args.min_class_support)
    dev_mask, _ = _usable_label_mask(dev_labels, args.min_class_support, usable_labels=usable_labels)
    print(f"[e01_pool_len] {len(usable_labels)} classes usables -- "
          f"{fit_mask.sum()}/{n_fit} FIT, {dev_mask.sum()}/{n_dev} DEV retenus.", flush=True)

    fit_texts_f = [t for t, m in zip(fit_texts, fit_mask) if m]
    dev_texts_f = [t for t, m in zip(dev_texts, dev_mask) if m]
    y_fit = np.array([l for l, m in zip(fit_labels, fit_mask) if m])
    y_dev = np.array([l for l, m in zip(dev_labels, dev_mask) if m])
    groups_dev_f = np.array([g for g, m in zip(dev_groups, dev_mask) if m])

    d_extra = args.d_extra
    acts_path = os.path.join(args.save_dir, "cache", f"p1_all_doc_acts_ext_d{d_extra}.pt")

    # ─── (a) Pooling alternatif : moyenne des top-k activations par feature ─
    print(f"[e01_pool_len] Re-pooling top-{args.topk}-moyenne depuis les fragments "
          "token-level (p1_token_fragments_ext)...", flush=True)
    frag_dir = os.path.join(args.save_dir, "cache", "p1_token_fragments_ext")
    d_total = None
    topk_rows = []
    for doc_id in range(n_fit + n_dev):
        frag = load_fragment(frag_dir, doc_id)
        vec = doc_topk_mean_pool(frag, k=args.topk)
        if d_total is None:
            d_total = vec.shape[0]
        topk_rows.append(vec)
        if doc_id % 5000 == 0:
            print(f"[e01_pool_len]   {doc_id}/{n_fit + n_dev} fragments re-poolés...", flush=True)
    topk_acts = torch.stack(topk_rows).numpy()
    del topk_rows
    d_core = d_total - d_extra

    print("[e01_pool_len] Chargement du max-pooling déjà en cache (référence)...", flush=True)
    maxpool_acts = load_all_doc_acts(acts_path).numpy()
    assert maxpool_acts.shape == topk_acts.shape, (
        f"max-pool {maxpool_acts.shape} vs top-{args.topk} {topk_acts.shape} -- "
        "désaccord de forme, ordre des fragments incompatible avec all_texts."
    )

    def _split(acts, cols):
        fit_a = acts[:n_fit][fit_mask][:, cols]
        dev_a = acts[n_fit:n_fit + n_dev][dev_mask][:, cols]
        return sp.csr_matrix(fit_a), sp.csr_matrix(dev_a)

    pooling_results = {}
    for pool_name, acts in (("maxpool", maxpool_acts), (f"top{args.topk}mean", topk_acts)):
        for repr_name, cols in (("core", slice(0, d_core)), ("full", slice(0, d_total))):
            X_fit, X_dev = _split(acts, cols)
            res = held_out_probe_accuracy(X_fit, y_fit, X_dev, y_dev)
            key = f"{pool_name}_{repr_name}"
            pooling_results[key] = res
            print(f"[e01_pool_len] acc_{key} = {res['accuracy']:.4f}", flush=True)
    del maxpool_acts, topk_acts

    pooling_comparisons = {}
    pooling_pvalues = []
    for repr_name in ("core", "full"):
        a = pooling_results[f"maxpool_{repr_name}"]["correct"]
        b = pooling_results[f"top{args.topk}mean_{repr_name}"]["correct"]
        mb = int(np.sum(a & ~b))
        mc = int(np.sum(~a & b))
        mcnemar = paired_mcnemar_test(mb, mc)
        diff_values = a.astype(float) - b.astype(float)
        ci = bootstrap_ci_by_group(diff_values, groups_dev_f, statistic=np.mean, n_boot=2000, seed=42)
        pooling_comparisons[f"maxpool_minus_top{args.topk}mean_{repr_name}"] = {
            "acc_maxpool": pooling_results[f"maxpool_{repr_name}"]["accuracy"],
            f"acc_top{args.topk}mean": pooling_results[f"top{args.topk}mean_{repr_name}"]["accuracy"],
            "diff": (pooling_results[f"maxpool_{repr_name}"]["accuracy"]
                     - pooling_results[f"top{args.topk}mean_{repr_name}"]["accuracy"]),
            "mcnemar_b": mb, "mcnemar_c": mc, "mcnemar_p": mcnemar.p,
            "bootstrap_ci_low": ci.ci_low, "bootstrap_ci_high": ci.ci_high,
            "bootstrap_n_groups": ci.n_groups,
        }
        pooling_pvalues.append(mcnemar.p)
    p_adj = fdr_bh(pooling_pvalues)
    for key, pa in zip(pooling_comparisons.keys(), p_adj):
        pooling_comparisons[key]["mcnemar_p_fdr_bh"] = float(pa)

    # ─── (b) Baseline longueur seule + stratification par longueur ─────────
    print("[e01_pool_len] Baseline longueur seule...", flush=True)
    len_fit = np.array([len(t) for t in fit_texts_f], dtype=float).reshape(-1, 1)
    len_dev = np.array([len(t) for t in dev_texts_f], dtype=float).reshape(-1, 1)
    res_length_only = held_out_probe_accuracy(len_fit, y_fit, len_dev, y_dev)
    print(f"[e01_pool_len] acc_length_only = {res_length_only['accuracy']:.4f}", flush=True)

    print("[e01_pool_len] Stratification par longueur (DEV)...", flush=True)
    quartile_edges = np.quantile(len_dev.ravel(), np.linspace(0, 1, args.n_length_strata + 1))
    strata_idx = np.clip(np.digitize(len_dev.ravel(), quartile_edges[1:-1]), 0, args.n_length_strata - 1)

    # Reconstruit CORE/FULL (max-pool, référence) et charge TFIDF/DENSE
    # depuis le premier passage (e01_representation_comparison.json) pour la
    # stratification -- pas de recalcul TFIDF/DENSE ici, seulement un nouveau
    # découpage des memes predictions n'est pas possible sans les refaire :
    # on ne stratifie donc que CORE/FULL (max-pool), disponibles ici.
    maxpool_acts = load_all_doc_acts(acts_path).numpy()
    X_fit_core, X_dev_core = _split(maxpool_acts, slice(0, d_core))
    X_fit_full, X_dev_full = _split(maxpool_acts, slice(0, d_total))
    del maxpool_acts
    res_core_full = held_out_probe_accuracy(X_fit_core, y_fit, X_dev_core, y_dev)
    res_full_full = held_out_probe_accuracy(X_fit_full, y_fit, X_dev_full, y_dev)

    length_strata = {}
    for s in range(args.n_length_strata):
        sel = strata_idx == s
        n_s = int(sel.sum())
        if n_s == 0:
            continue
        length_strata[f"stratum_{s}"] = {
            "n": n_s,
            "length_range": [float(len_dev[sel].min()), float(len_dev[sel].max())],
            "acc_core": float(res_core_full["correct"][sel].mean()),
            "acc_full": float(res_full_full["correct"][sel].mean()),
            "acc_length_only": float(res_length_only["correct"][sel].mean()),
        }

    out = {
        "n_fit": int(fit_mask.sum()), "n_dev": int(dev_mask.sum()),
        "n_classes": len(usable_labels), "topk": args.topk,
        "pooling_accuracies": {k: v["accuracy"] for k, v in pooling_results.items()},
        "pooling_comparisons": pooling_comparisons,
        "length_only_baseline_accuracy": res_length_only["accuracy"],
        "length_strata": length_strata,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    out_path = os.path.join(args.save_dir, args.out)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp_path, out_path)
    print(f"[e01_pool_len] Écrit {out_path}", flush=True)
    print(json.dumps({k: out[k] for k in ("pooling_comparisons", "length_strata",
                                            "length_only_baseline_accuracy")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
