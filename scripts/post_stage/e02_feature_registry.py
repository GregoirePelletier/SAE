"""
scripts/post_stage/e02_feature_registry.py -- Registre de features CORE/EXTRA
labellisées (Plan_execution_SAE_15_jours_Claude_Code.md §7). Réutilise
l'infrastructure de labellisation déjà auditée (`src/sae/judge.py::
odd_one_out_judge`, `feature_selection_stratified_by_frequency`) plutôt que
d'en réécrire une -- CLAUDE.md interdit de dupliquer une fonctionnalité déjà
présente sans comparaison documentée.

Simplification assumée par rapport à l'ambition complète du plan : le juge
`odd_one_out_judge` produit un score binaire (interp_score 0/1) + un statut
"dead_feature"/support insuffisant en amont, pas la taxonomie fine à quatre
valeurs (unclear/syntactic/mixed/insufficient_support) -- mappé ici sur trois
statuts (`interpretable`, `unclear`, `insufficient_support`) plutôt que
d'écrire un second prompt de classification, faute de temps. Documenté
explicitement, pas silencieux.

Exemples justificatifs conservés dans le manifeste LOCAL uniquement
(SAVE_DIR, jamais git -- résultats_*/ gitignored) : extraits de mails
synthétiques, autorisés dans ce cadre mais non versionnés par principe.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.post_stage.dataset_contract import load_fit_dev_corpus_from_manifest  # noqa: E402
from src.sae.sae_shared import load_all_doc_acts  # noqa: E402
from src.sae.judge import (  # noqa: E402
    load_judge_model, odd_one_out_judge, feature_selection_stratified_by_frequency,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--mails-tsv-path", default="local_data/emails/Mails.tsv")
    ap.add_argument("--augmented-jsonl-path", default="local_data/emails/augmented_mails.jsonl")
    ap.add_argument("--split-assignments", required=True)
    ap.add_argument("--d-extra", type=int, required=True)
    ap.add_argument("--max-augmented-per-mail", type=int, default=13)
    ap.add_argument("--n-core", type=int, default=150)
    ap.add_argument("--n-extra", type=int, default=150)
    ap.add_argument("--sample-docs", type=int, default=500)
    ap.add_argument("--sae-revision", default="e01_fit_1b_layer13_k5")
    ap.add_argument("--judge-device", default="cuda",
                     help='"cuda" (1 GPU, h100/h100-bis) ou "auto" (sharding multi-GPU, '
                          'necessaire sur a100 -- cf. campaign_policy.yaml).')
    ap.add_argument("--out", default="e02_feature_registry.json")
    args = ap.parse_args()

    t0 = time.time()
    print("[e02_registry] Chargement corpus FIT/DEV gelé (FIT seul utilisé ici)...", flush=True)
    fit_texts, fit_labels, dev_texts, dev_labels, fit_groups, dev_groups = (
        load_fit_dev_corpus_from_manifest(
            args.split_assignments, args.mails_tsv_path, args.augmented_jsonl_path,
            max_augmented_per_mail=args.max_augmented_per_mail, return_groups=True,
        )
    )
    n_fit = len(fit_texts)
    print(f"[e02_registry] FIT={n_fit} (exemples de labellisation exclusivement issus de FIT, §7)", flush=True)

    frag_dir = os.path.join(args.save_dir, "cache", "p1_token_fragments_ext")
    acts_path = os.path.join(args.save_dir, "cache", f"p1_all_doc_acts_ext_d{args.d_extra}.pt")
    all_doc_acts = load_all_doc_acts(acts_path)
    d_total = all_doc_acts.shape[1]
    d_core = d_total - args.d_extra
    acts_fit = all_doc_acts[:n_fit]
    del all_doc_acts

    doc_indices = list(range(n_fit))
    print(f"[e02_registry] Sélection stratifiée par fréquence : "
          f"{args.n_core} CORE (lo=0,hi={d_core}), {args.n_extra} EXTRA (lo={d_core},hi={d_total})...",
          flush=True)
    core_ids, core_bin_info = feature_selection_stratified_by_frequency(
        frag_dir, doc_indices, d_total, args.n_core, sample_docs=args.sample_docs,
        lo=0, hi=d_core, seed=0, return_bin_info=True,
    )
    extra_ids, extra_bin_info = feature_selection_stratified_by_frequency(
        frag_dir, doc_indices, d_total, args.n_extra, sample_docs=args.sample_docs,
        lo=d_core, hi=d_total, seed=0, return_bin_info=True,
    )
    print(f"[e02_registry] {len(core_ids)} CORE + {len(extra_ids)} EXTRA candidats retenus.", flush=True)

    print("[e02_registry] Chargement du juge Qwen...", flush=True)
    model, tokenizer = load_judge_model(device=args.judge_device)

    all_ids = core_ids + extra_ids
    print(f"[e02_registry] Labellisation odd-one-out de {len(all_ids)} features (juge Qwen, exemples FIT)...",
          flush=True)
    labels = odd_one_out_judge(
        model, tokenizer, all_ids, frag_dir, acts_fit, offset=0, doc_groups=fit_groups,
    )
    del model
    torch.cuda.empty_cache()

    def _status(entry: dict) -> str:
        if entry.get("label") == "dead_feature":
            return "insufficient_support"
        if entry.get("interp_score") == 1:
            return "interpretable"
        return "unclear"

    registry = {}
    bin_info_by_id = {**core_bin_info, **extra_bin_info}
    for f_idx in all_ids:
        branch = "core" if f_idx < d_core else "extra"
        local_index = f_idx if branch == "core" else f_idx - d_core
        feature_uid = f"{args.sae_revision}:{branch}:{local_index}"
        entry = labels.get(f_idx, {})
        registry[feature_uid] = {
            "feature_uid": feature_uid, "global_index": f_idx, "branch": branch,
            "local_index": local_index, "sae_revision": args.sae_revision,
            "label": entry.get("label"), "brief_description": entry.get("brief_description"),
            "interp_score": entry.get("interp_score"),
            "rho_interp": entry.get("rho_interp"),
            "status": _status(entry),
            "selection_bin_info": bin_info_by_id.get(f_idx),
            "source": "FIT", "human_verified": False,
        }

    n_by_status = {}
    for v in registry.values():
        n_by_status[v["status"]] = n_by_status.get(v["status"], 0) + 1
    print(f"[e02_registry] Répartition des statuts : {n_by_status}", flush=True)

    out = {
        "schema_version": "post-stage-registry-v1",
        "sae_revision": args.sae_revision,
        "n_core": len(core_ids), "n_extra": len(extra_ids),
        "n_by_status": n_by_status,
        "human_verification_pending": True,
        "human_verification_target_n": "60-100 (plan §7, non fait dans cette passe)",
        "features": registry,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    out_path = os.path.join(args.save_dir, args.out)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, out_path)
    print(f"[e02_registry] Écrit {out_path} ({len(registry)} features)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
