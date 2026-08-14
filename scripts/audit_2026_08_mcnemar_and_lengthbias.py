"""
scripts/audit_2026_08_mcnemar_and_lengthbias.py — Deux vérifications CPU-only
sur cache déjà existant, `docs/AUDIT_2026-08.md` :

  (A.2/F.2) McNemar apparié entre le gate odd-one-out (p1_judge_labels_extended.json,
  interp_score) et la labellisation contrastive directe (p1_contrastive_labels.json,
  confident) sur les features en commun -- jamais fait, le prompt le demandait
  explicitement pour clore A.2/F.2.

  (B.9) Corrélation Spearman entre longueur de document (tokens) et (a) nombre de
  features actives, (b) norme du vecteur doc-level max-poolé -- sur les activations
  déjà en cache. Répond à la question du biais de longueur du max-pooling.

Usage : sbatch slurm/validation/run_audit_mcnemar_lengthbias.slurm
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr

from src.config import SAVE_DIR, LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, CORPUS_SPLIT_SEED
from src.data.preparation import build_email_train_test_corpus
from src.analysis.stats import paired_mcnemar_test

CACHE_DIR = os.path.join(SAVE_DIR, "cache")
OUT_PATH = os.path.join(CACHE_DIR, "audit_2026_08_mcnemar_lengthbias_results.json")


def mcnemar_odd_one_out_vs_contrastive():
    print("=" * 70)
    print(" A.2/F.2 — McNemar : odd-one-out vs labellisation contrastive")
    print("=" * 70)
    with open(os.path.join(CACHE_DIR, "p1_judge_labels_extended.json"), encoding="utf-8") as f:
        oddoneout = json.load(f)
    with open(os.path.join(CACHE_DIR, "p1_contrastive_labels.json"), encoding="utf-8") as f:
        contrastive = json.load(f)["per_feature"]

    common = sorted(set(oddoneout.keys()) & set(contrastive.keys()), key=int)
    print(f"  {len(common)} features en commun entre les deux caches.")

    a_correct, b_correct = [], []  # a = odd-one-out interp==1, b = contrastive confident
    for k in common:
        a_correct.append(int(oddoneout[k].get("interp_score", 0) == 1))
        b_correct.append(int(bool(contrastive[k].get("confident", False))))
    a_arr, b_arr = np.array(a_correct), np.array(b_correct)

    n_both = int(((a_arr == 1) & (b_arr == 1)).sum())
    n_neither = int(((a_arr == 0) & (b_arr == 0)).sum())
    b_count = int(((a_arr == 1) & (b_arr == 0)).sum())  # odd-one-out oui, contrastif non
    c_count = int(((a_arr == 0) & (b_arr == 1)).sum())  # odd-one-out non, contrastif oui
    mc = paired_mcnemar_test(b_count, c_count)

    result = {
        "n_common_features": len(common),
        "rate_odd_one_out": float(a_arr.mean()),
        "rate_contrastive_confident": float(b_arr.mean()),
        "n_both_positive": n_both, "n_neither": n_neither,
        "b_oddoneout_only": b_count, "c_contrastive_only": c_count,
        "mcnemar_statistic": mc.statistic, "mcnemar_p": mc.p, "mcnemar_exact": mc.exact,
    }
    print(f"  taux odd-one-out={a_arr.mean():.3f} | taux contrastif confident={b_arr.mean():.3f}")
    print(f"  accord : both={n_both} neither={n_neither} | discordant b={b_count} c={c_count}")
    print(f"  McNemar : stat={mc.statistic} p={mc.p:.4f} exact={mc.exact}")
    return result


def length_bias_analysis():
    print("\n" + "=" * 70)
    print(" B.9 — Biais de longueur du max-pooling documentaire")
    print("=" * 70)
    train_texts, _, _, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED,
    )
    n_train = len(train_texts)
    lengths_chars = np.array([len(t) for t in train_texts])

    acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts_ext_d1024.pt")
    all_doc_acts = torch.load(acts_path, map_location="cpu", weights_only=True)
    train_acts = all_doc_acts[:n_train].float()

    n_active = (train_acts > 1e-6).sum(dim=1).numpy()
    doc_norm = train_acts.norm(dim=1).numpy()

    rho_active, p_active = spearmanr(lengths_chars, n_active)
    rho_norm, p_norm = spearmanr(lengths_chars, doc_norm)

    result = {
        "n_docs": int(n_train),
        "spearman_length_vs_n_active_features": {"rho": float(rho_active), "p": float(p_active)},
        "spearman_length_vs_doc_vector_norm": {"rho": float(rho_norm), "p": float(p_norm)},
        "mean_length_chars": float(lengths_chars.mean()),
        "mean_n_active_features": float(n_active.mean()),
    }
    print(f"  longueur (car.) vs n_features_actives : rho={rho_active:.4f} p={p_active:.2e}")
    print(f"  longueur (car.) vs norme du vecteur doc : rho={rho_norm:.4f} p={p_norm:.2e}")
    return result


def main():
    r1 = mcnemar_odd_one_out_vs_contrastive()
    r2 = length_bias_analysis()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"mcnemar_oddoneout_vs_contrastive": r1, "length_bias": r2}, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Écrit : {OUT_PATH}")


if __name__ == "__main__":
    main()
