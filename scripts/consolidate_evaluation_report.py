"""
scripts/consolidate_evaluation_report.py — Assemble tous les artefacts d'évaluation
d'un run (cf. docs/evaluation_protocol.md) en un seul rapport markdown + un résumé
JSON, pour éviter d'ouvrir séparément les ~15 fichiers produits par les différents
scripts de diagnostic de ce projet.

Zéro calcul : lecture pure de fichiers déjà sur disque.

Usage :
    PYTHONPATH=. .venv/bin/python scripts/consolidate_evaluation_report.py results_v10_emails_main
"""
from __future__ import annotations

import glob
import json
import os
import sys

import pandas as pd


def load_json(path: str):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def section(title: str) -> str:
    return f"\n## {title}\n\n"


def consolidate(run_dir: str) -> tuple[str, dict]:
    cache = os.path.join(run_dir, "cache")
    md = [f"# Rapport d'évaluation consolidé — `{run_dir}`\n"]
    summary = {"run_dir": run_dir}

    # 1. Reconstruction (results.json)
    results = load_json(os.path.join(run_dir, "results.json"))
    md.append(section("1. Reconstruction SAE (results.json)"))
    if results:
        for pk, title in [("P1_Gemma3_SAE", "Pipeline 1"), ("P2_F2LLM_PhSAE", "Pipeline 2")]:
            metrics = results.get(pk)
            if metrics:
                md.append(f"**{title}**\n\n")
                md.append("| Métrique | Valeur |\n|---|---|\n")
                for k, v in metrics.items():
                    if not isinstance(v, (dict, list)):
                        md.append(f"| {k} | {v} |\n")
                md.append("\n")
        summary["results"] = {k: v for k, v in (results or {}).items()}
    else:
        md.append("_Absent._\n")

    # 2. Comparaison SAELens
    saelens = load_json(os.path.join(cache, "saelens_numeric_comparison.json"))
    md.append(section("2. Comparaison chiffrée avec SAELens"))
    if saelens:
        md.append("| Formule | FVE |\n|---|---|\n")
        md.append(f"| Notre `compute_metrics` | {saelens['fve_ours']:.4f} |\n")
        md.append(f"| sae_lens legacy (par token) | {saelens['explained_variance_legacy_saelens']:.4f} |\n")
        md.append(f"| sae_lens corrigée (globale) | {saelens['explained_variance_corrected_saelens']:.4f} |\n")
        summary["saelens_comparison"] = saelens
    else:
        md.append("_Absent._\n")

    # 3-4. Labellisation core / extension (odd-one-out)
    judge_ext = load_json(os.path.join(cache, "p1_judge_labels_extended.json"))
    md.append(section("3-4. Labellisation extension — gate odd-one-out"))
    if judge_ext:
        n = len(judge_ext)
        n_interp = sum(1 for v in judge_ext.values() if v.get("interp_score") == 1)
        n_dead = sum(1 for v in judge_ext.values() if v.get("label") == "dead_feature")
        md.append(f"- Features jugées : {n}\n- Interprétables (odd-one-out) : {n_interp}/{n} "
                  f"({100*n_interp/n:.1f}%)\n- Mortes : {n_dead}/{n}\n")
        summary["judge_odd_one_out"] = {"n": n, "n_interp": n_interp, "rate": n_interp / n, "n_dead": n_dead}
    else:
        md.append("_Absent._\n")

    # 5. Labellisation contrastive directe
    contrastive = load_json(os.path.join(cache, "p1_contrastive_labels.json"))
    md.append(section("5. Labellisation extension — contrastive directe (alternative)"))
    if contrastive:
        s = contrastive.get("summary", {})
        md.append(f"- Taux confident (protocole nouveau) : {s.get('confident_rate_new_protocol', 'n/a')}\n"
                  f"- Récupération parmi les non-interprétables (gate) : "
                  f"{s.get('n_original_noninterp_now_confident', 'n/a')}/{s.get('n_original_noninterp', 'n/a')}\n")
        summary["contrastive_labeling"] = s
    else:
        md.append("_Absent._\n")

    # 6. Robustesse du jugement
    robustness = load_json(os.path.join(cache, "p1_judge_robustness.json"))
    md.append(section("6. Robustesse du protocole de jugement (ordre des exemples)"))
    if robustness:
        s = robustness.get("summary", {})
        md.append(f"- Taux single-shot : {s.get('single_shot_interp_rate', 'n/a')}\n"
                  f"- Taux vote majoritaire (5 répétitions) : {s.get('majority_vote_interp_rate', 'n/a')}\n"
                  f"- Accord moyen entre répétitions : {s.get('mean_agreement_rate', 'n/a')}\n")
        summary["judge_robustness"] = s
    else:
        md.append("_Absent._\n")

    # 8. Intent/urgence
    intent = load_json(os.path.join(cache, "intent_urgency_probe_results.json"))
    md.append(section("7-8. Détection d'urgence/intention (mails réels, labels faibles regex)"))
    if intent:
        md.append("| Intention | n_pos/n_total | acc_SAE | baseline | Δ |\n|---|---|---|---|---|\n")
        for k, v in intent.items():
            delta = v["acc_sae"] - v["majority_baseline"]
            md.append(f"| {k} | {v['n_pos']}/{v['n_total']} | {v['acc_sae']:.4f} | "
                      f"{v['majority_baseline']:.4f} | {delta:+.4f} |\n")
        summary["intent_urgency_probe"] = intent
    else:
        md.append("_Absent._\n")

    # 9. Fidélité de l'explication
    fidelity = load_json(os.path.join(cache, "explanation_fidelity_results.json"))
    md.append(section("9. Fidélité de l'explication document-level (ablation)"))
    if fidelity:
        md.append("| Intention | n docs | chute top-K | chute random-K | chute bottom-K | ratio top/random |\n"
                   "|---|---|---|---|---|---|\n")
        for k, v in fidelity.items():
            md.append(f"| {k} | {v['n_docs_tested']} | {v['mean_drop_top_k']:.4f} | "
                      f"{v['mean_drop_random_k']:.4f} | {v['mean_drop_bottom_k']:.4f} | "
                      f"{v['fidelity_ratio_top_vs_random']:.1f}x |\n")
        summary["explanation_fidelity"] = {k: {kk: vv for kk, vv in v.items() if kk != "examples"}
                                            for k, v in fidelity.items()}
    else:
        md.append("_Absent._\n")

    # 10. Plausibilité de l'explication
    plausibility = load_json(os.path.join(cache, "explanation_plausibility_results.json"))
    md.append(section("10. Plausibilité de l'explication document-level (choix forcé)"))
    if plausibility:
        s = plausibility.get("summary", {})
        md.append(f"- Taux de succès : {s.get('n_correct')}/{s.get('n_tested')} "
                  f"= {100*s.get('success_rate', 0):.1f}% (hasard = 50%)\n")
        summary["explanation_plausibility"] = s
    else:
        md.append("_Absent._\n")

    # 12. Corrélations intéressantes
    correlations = load_json(os.path.join(run_dir, "p1_interesting_correlations.json"))
    md.append(section("12. Corrélations \"intéressantes\" (NPMI élevé + labels dissimilaires)"))
    if correlations:
        md.append(f"- {len(correlations)} paires retenues.\n")
        summary["interesting_correlations_n"] = len(correlations)
    else:
        md.append("_Absent._\n")

    # 13/16. Diffing
    md.append(section("13/16. Diffing cross-domaine (mails originaux vs augmentés, SAE natif)"))
    diff_dirs = sorted(glob.glob(os.path.join(run_dir, "**", "diff_*.csv"), recursive=True))
    if diff_dirs:
        rows = []
        for f in diff_dirs:
            df = pd.read_csv(f)
            n_sig = int(df["significant"].sum()) if "significant" in df.columns else None
            top = df.iloc[0]["label"] if len(df) else None
            rows.append((os.path.basename(f), n_sig, top))
        md.append("| Fichier | Features sig. | Top feature |\n|---|---|---|\n")
        for fname, n_sig, top in rows:
            md.append(f"| {fname} | {n_sig} | {top} |\n")
        summary["diffing_files"] = len(rows)
    else:
        md.append("_Absent dans ce run (chercher sous results_v9_test/ ou results_v11_baseline_objetfix/)._\n")

    return "".join(md), summary


def main():
    if len(sys.argv) < 2:
        print("Usage: consolidate_evaluation_report.py <run_dir> [<run_dir2> ...]")
        sys.exit(1)
    for run_dir in sys.argv[1:]:
        md, summary = consolidate(run_dir)
        out_md = os.path.join(run_dir, "EVALUATION_REPORT.md")
        out_json = os.path.join(run_dir, "evaluation_summary.json")
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(md)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[+] {run_dir} -> {out_md}, {out_json}")


if __name__ == "__main__":
    main()
