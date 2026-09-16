"""
scripts/post_stage/e04_diffing.py -- Diffing dans le domaine cible
(Plan_execution_SAE_15_jours_Claude_Code.md §9). Reutilise integralement
l'infrastructure App D.2/K.1 deja auditee et validee (AUDIT_SAE_2026-08.md :
"Diffing structure... confirme sain, verification_rate=80%") plutot que
d'en ecrire une nouvelle -- corpus_diff_stats, select_top_diff_features_by_
frequency, generate_structured_diff_hypotheses, verify_hypotheses,
compute_verification_metrics.

Contraste choisi (§9.1, "controle apparie utile") : variantes de TON d'un
MEME mail d'origine -- axe "urgence", niveaux "panique" (A, cible) vs
"calme" (B) -- deja generees, deja connues par construction (champ de
generation), appariees par parent_id. Decouverte sur FIT, gel des
hypotheses AVANT toute lecture de CONFIRM, verification sur CONFIRM sans
jamais reveler le sens A/B au juge (verify_hypotheses ne voit que le texte
du document et l'enonce de l'hypothese).

Simplification assumee : verification par juge Qwen, pas d'audit humain
(§9.2 point 5 du plan demande un audit humain des resultats) -- non fait
ici, a faire par Gregoire.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.post_stage.dataset_contract import (  # noqa: E402
    load_fit_dev_corpus_from_manifest, load_confirm_corpus_from_manifest,
)
from src.sae.sae_shared import load_all_doc_acts  # noqa: E402
from src.sae.judge import load_judge_model  # noqa: E402
from src.analysis.cooccurrence import corpus_diff_stats, select_top_diff_features_by_frequency  # noqa: E402
from src.analysis.diff_hypothesis_generator import generate_structured_diff_hypotheses  # noqa: E402
from src.analysis.hypothesis_verifier import verify_hypotheses, compute_verification_metrics  # noqa: E402
from src.analysis.stats import proportion_with_ci, two_proportion_test, fdr_bh  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--mails-tsv-path", default="local_data/emails/Mails.tsv")
    ap.add_argument("--augmented-jsonl-path", default="local_data/emails/augmented_mails.jsonl")
    ap.add_argument("--split-assignments", required=True)
    ap.add_argument("--registry-path", required=True)
    ap.add_argument("--d-extra", type=int, required=True)
    ap.add_argument("--max-augmented-per-mail", type=int, default=13)
    ap.add_argument("--label-a", default="urgence__panique")
    ap.add_argument("--label-b", default="urgence__calme")
    ap.add_argument("--freq-diff-threshold", type=float, default=0.03)
    ap.add_argument("--top-n-features", type=int, default=200)
    ap.add_argument("--num-hypotheses", type=int, default=8)
    ap.add_argument("--verification-threshold", type=float, default=0.01)
    ap.add_argument("--out", default="e04_diffing.json")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[e04_diffing] Contraste : A={args.label_a!r} (cible) vs B={args.label_b!r}", flush=True)

    print("[e04_diffing] Chargement FIT/DEV (decouverte)...", flush=True)
    fit_texts, fit_labels, dev_texts, dev_labels, fit_groups, dev_groups = (
        load_fit_dev_corpus_from_manifest(
            args.split_assignments, args.mails_tsv_path, args.augmented_jsonl_path,
            max_augmented_per_mail=args.max_augmented_per_mail, return_groups=True,
        )
    )
    n_fit = len(fit_texts)

    with open(args.registry_path) as f:
        registry = json.load(f)["features"]
    full_labels = {v["global_index"]: v["label"] for v in registry.values() if v["status"] == "interpretable"}

    acts_path = os.path.join(args.save_dir, "cache", f"p1_all_doc_acts_ext_d{args.d_extra}.pt")
    all_doc_acts = load_all_doc_acts(acts_path)
    fit_acts = all_doc_acts[:n_fit]

    a_mask_fit = np.array([l == args.label_a for l in fit_labels])
    b_mask_fit = np.array([l == args.label_b for l in fit_labels])
    n_a_fit, n_b_fit = int(a_mask_fit.sum()), int(b_mask_fit.sum())
    print(f"[e04_diffing] FIT (decouverte) : {n_a_fit} A, {n_b_fit} B", flush=True)
    if n_a_fit < 10 or n_b_fit < 10:
        raise RuntimeError(
            f"Effectif de decouverte insuffisant (A={n_a_fit}, B={n_b_fit}) -- "
            "choisir un autre couple label_a/label_b avant de continuer."
        )
    pair_mask_fit = a_mask_fit | b_mask_fit
    diff_df = corpus_diff_stats(
        fit_acts[torch.from_numpy(pair_mask_fit)].float(),
        group_mask=a_mask_fit[pair_mask_fit], feature_labels=full_labels,
    )
    del fit_acts
    print(f"[e04_diffing] corpus_diff_stats : {len(diff_df)} features actives dans A∪B.", flush=True)

    selected = select_top_diff_features_by_frequency(
        diff_df, threshold=args.freq_diff_threshold, top_n=args.top_n_features)
    print(f"[e04_diffing] {len(selected)} features au-dessus du seuil {args.freq_diff_threshold} "
          f"(top {args.top_n_features}).", flush=True)
    features = [
        {"feature_id": int(row["feature_id"]), "label": row["label"],
         "percentage_difference": float(row["log_odds_ratio"])}
        for _, row in selected.iterrows()
    ]

    print("[e04_diffing] Chargement du juge Qwen...", flush=True)
    model, tokenizer = load_judge_model()

    print(f"[e04_diffing] Generation d'au plus {args.num_hypotheses} hypotheses structurees...", flush=True)
    hypotheses = generate_structured_diff_hypotheses(
        model, tokenizer, features,
        query=f"What distinguishes panicked/urgent-toned emails (target) from calm-toned emails (other), "
              f"among paraphrased variants of the same original emails?",
        num_hypotheses=args.num_hypotheses,
    )
    print(f"[e04_diffing] {len(hypotheses)} hypotheses generees (gelees avant lecture de CONFIRM) :", flush=True)
    for h in hypotheses:
        print(f"    - [{h.get('dataset')}, diff={h.get('percentage_difference', 0):+.2f}, "
              f"conf={h.get('confidence', 0):.2f}] {h.get('description')}", flush=True)

    if not hypotheses:
        print("[e04_diffing] Aucune hypothese valide -- arret avant verification.", flush=True)
        out = {"features_considered": features, "hypotheses": [], "verification": None}
        out_path = os.path.join(args.save_dir, args.out)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        return 0

    print("[e04_diffing] Chargement CONFIRM (verification -- jamais vu pendant la decouverte)...", flush=True)
    confirm_texts, confirm_labels, confirm_groups = load_confirm_corpus_from_manifest(
        args.split_assignments, args.mails_tsv_path, args.augmented_jsonl_path,
        max_augmented_per_mail=args.max_augmented_per_mail, return_groups=True,
    )
    a_mask_confirm = np.array([l == args.label_a for l in confirm_labels])
    b_mask_confirm = np.array([l == args.label_b for l in confirm_labels])
    n_a_confirm, n_b_confirm = int(a_mask_confirm.sum()), int(b_mask_confirm.sum())
    print(f"[e04_diffing] CONFIRM (verification) : {n_a_confirm} A, {n_b_confirm} B", flush=True)
    pair_mask_confirm = a_mask_confirm | b_mask_confirm
    confirm_pair_texts = [t for t, m in zip(confirm_texts, pair_mask_confirm) if m]
    confirm_pair_group_mask = a_mask_confirm[pair_mask_confirm]  # True = A (cible), meme convention que compute_verification_metrics

    hypothesis_texts = [h["description"] for h in hypotheses]
    print(f"[e04_diffing] Verification {len(hypothesis_texts)}x{len(confirm_pair_texts)} sur CONFIRM "
          "(juge ne voit ni le groupe ni la methode d'origine)...", flush=True)
    matrix = verify_hypotheses(model, tokenizer, hypothesis_texts, confirm_pair_texts)
    del model
    torch.cuda.empty_cache()

    per_hyp, summary = compute_verification_metrics(
        matrix, confirm_pair_group_mask, hypothesis_labels=hypothesis_texts,
        threshold=args.verification_threshold,
    )
    print("\n" + "=" * 60, flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v}", flush=True)
    print(per_hyp.to_string(), flush=True)

    # Statistiques additionnelles (plan §9.4, au-dela de verification_rate/
    # coverage du papier) : IC de Wilson par bras + test a deux proportions +
    # correction FDR-BH sur la famille d'hypotheses gelees.
    n_a_pair = int(confirm_pair_group_mask.sum())
    n_b_pair = int((~confirm_pair_group_mask).sum())
    stats_rows = []
    pvalues = []
    for i, hyp in enumerate(hypothesis_texts):
        n_success_a = int(matrix[i, confirm_pair_group_mask].sum())
        n_success_b = int(matrix[i, ~confirm_pair_group_mask].sum())
        ci_a = proportion_with_ci(n_success_a, n_a_pair) if n_a_pair else None
        ci_b = proportion_with_ci(n_success_b, n_b_pair) if n_b_pair else None
        two_prop = (two_proportion_test(n_success_a, n_a_pair, n_success_b, n_b_pair)
                    if n_a_pair and n_b_pair else None)
        pvalues.append(two_prop.p if two_prop else 1.0)
        stats_rows.append({
            "hypothesis": hyp,
            "n_a": n_a_pair, "n_success_a": n_success_a,
            "ci_a_low": ci_a.ci_low if ci_a else None, "ci_a_high": ci_a.ci_high if ci_a else None,
            "n_b": n_b_pair, "n_success_b": n_success_b,
            "ci_b_low": ci_b.ci_low if ci_b else None, "ci_b_high": ci_b.ci_high if ci_b else None,
            "diff": (two_prop.diff if two_prop else None),
            "z": (two_prop.z if two_prop else None), "p": (two_prop.p if two_prop else None),
        })
    p_adj = fdr_bh(pvalues) if pvalues else []
    for row, pa in zip(stats_rows, p_adj):
        row["p_fdr_bh"] = float(pa)

    out = {
        "contrast": {"label_a": args.label_a, "label_b": args.label_b},
        "discovery": {"n_a_fit": n_a_fit, "n_b_fit": n_b_fit, "n_features_diff_df": len(diff_df),
                      "n_features_selected": len(selected)},
        "verification": {"n_a_confirm": n_a_confirm, "n_b_confirm": n_b_confirm,
                          "summary": summary, "per_hypothesis": per_hyp.to_dict(orient="records"),
                          "stats_with_ci_and_fdr": stats_rows},
        "hypotheses": hypotheses,
        "judge": "Qwen (model-verified, human audit pending -- plan §9.2 point 5)",
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    out_path = os.path.join(args.save_dir, args.out)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, out_path)
    print(f"[e04_diffing] Écrit {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
