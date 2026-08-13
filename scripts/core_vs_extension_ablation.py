"""
scripts/core_vs_extension_ablation.py — Phase 1 : le SAE core seul (GemmaScope-2,
sans extension) fait-il mieux, pareil, ou moins bien que core+extension
(`FrozenCoreResidualSAE`) ? Question posée explicitement par l'utilisateur
("pollution de notre côté") : l'extension entraînée sur le résidu pourrait ne
rien apporter, voire dégrader le signal du core, sur les métriques qui comptent
réellement (classification en aval, structure de cluster).

ZÉRO rerun / GPU / LLM : réutilise les activations denses déjà en cache du run
principal (`results_v10_emails_main/cache/p1_all_doc_acts_ext_d1024.pt`,
[n_docs, d_core+D_EXTRA]) — reconstruit uniquement les LABELS de split
(déterministe, CORPUS_SPLIT_SEED=42, aucun aléa) pour ré-aligner les lignes du
tenseur sur train/test/diff, puis calcule les mêmes métriques que le run
principal sur `acts[:, :d_core]` (core seul) vs `acts` (core+extension) --
mêmes folds de validation croisée pour les deux, afin d'autoriser un test de
McNemar apparié (pas seulement deux taux comparés à l'aveugle).

Usage : .venv/bin/python scripts/core_vs_extension_ablation.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "sae"))

from sae_shared import build_email_train_test_corpus, prepare_domain_dataset
from src.data.keywords import (
    ENERGY_KEYWORDS, SPORTS_KEYWORDS, SUPPORT_KEYWORDS,
    ENERGY_URL_PATTERNS, SPORTS_URL_PATTERNS, SUPPORT_URL_PATTERNS,
)
from src.analysis.stats import paired_mcnemar_test
from src.analysis.plotting import plot_metric_vs_hyperparam

RUN_DIR = os.path.join(REPO_ROOT, "results_v10_emails_main")
CACHE_DIR = os.path.join(RUN_DIR, "cache")
ACTS_PATH = os.path.join(CACHE_DIR, "p1_all_doc_acts_ext_d1024.pt")
D_EXTRA = 1024  # nom de fichier / config du run principal
CORPUS_SPLIT_SEED = 42
LOCAL_MAILS_PATH = os.path.join(REPO_ROOT, "local_data", "emails", "Mails.tsv")
LOCAL_AUGMENTED_MAILS_PATH = os.path.join(REPO_ROOT, "local_data", "emails", "augmented_mails.jsonl")
LOCAL_DATASET_PATH = os.path.join(
    REPO_ROOT, "datasets", "fineweb2_fra", "data", "fra_Latn", "train", "000_00000.parquet")
N_TOTAL_ENERGY = N_TOTAL_SPORTS = N_TOTAL_SUPPORT = 300


def reconstruct_splits():
    """Reproduit exactement la construction de corpus de saev5.py::__main__
    (mêmes seeds/env par défaut que le run principal) -- déterministe, aucun
    GPU/LLM requis, uniquement pour retrouver les frontières de lignes du
    tenseur d'activations déjà en cache."""
    train_texts, train_labels, test_texts, test_labels = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH,
        test_split=0.05, max_augmented_per_mail=13, seed=CORPUS_SPLIT_SEED,
    )
    energy_texts = prepare_domain_dataset(
        ENERGY_KEYWORDS, "energy", N_TOTAL_ENERGY, chunk_length=1024, max_chunks=20,
        url_patterns=ENERGY_URL_PATTERNS, local_dataset_path=LOCAL_DATASET_PATH, use_fineweb2=True,
    )
    sports_texts = prepare_domain_dataset(
        SPORTS_KEYWORDS, "sports", N_TOTAL_SPORTS, chunk_length=1024, max_chunks=20,
        url_patterns=SPORTS_URL_PATTERNS, local_dataset_path=LOCAL_DATASET_PATH, use_fineweb2=True,
    )
    support_texts = prepare_domain_dataset(
        SUPPORT_KEYWORDS, "support", N_TOTAL_SUPPORT, chunk_length=1024, max_chunks=20,
        url_patterns=SUPPORT_URL_PATTERNS, local_dataset_path=LOCAL_DATASET_PATH, use_fineweb2=True,
    )
    diff_labels = ["energy"] * len(energy_texts) + ["sports"] * len(sports_texts) + ["support"] * len(support_texts)
    return train_labels, test_labels, diff_labels


def silhouette(acts: torch.Tensor, labels: list, n_max: int = 2000) -> float:
    from sklearn.metrics import silhouette_score
    X, lbl = acts.float().numpy(), np.array(labels)
    if len(set(lbl)) < 2:
        return float("nan")
    if X.shape[0] > n_max:
        idx = np.random.RandomState(0).choice(X.shape[0], n_max, replace=False)
        X, lbl = X[idx], lbl[idx]
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return float(silhouette_score(X_norm, lbl, metric="cosine"))


def cv_predictions(acts_by_label: dict) -> tuple[float, np.ndarray, np.ndarray]:
    """Même protocole que src/analysis/metrics.py::downstream_classification
    (StratifiedKFold(5, shuffle=True, random_state=42), LogisticRegression
    C=1.0) mais retourne aussi les prédictions par document (alignées sur
    l'ordre X/y d'entrée) pour permettre un McNemar apparié entre conditions."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score

    X_list, y_list = [], []
    for label_id, (label_name, acts) in enumerate(acts_by_label.items()):
        X_list.append(acts.float().numpy())
        y_list.append(np.full(acts.shape[0], label_id))
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    solver = "liblinear" if len(acts_by_label) <= 2 else "lbfgs"

    correct = np.full(len(y), -1, dtype=int)  # -1 = jamais en test (ne devrait pas arriver, StratifiedKFold couvre tout)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000, C=1.0, solver=solver)
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        correct[test_idx] = (preds == y[test_idx]).astype(int)
        accs.append(accuracy_score(y[test_idx], preds))
    return float(np.mean(accs)), correct, y


def run_condition(acts_train, acts_test, acts_diff, test_labels, train_labels, diff_labels):
    metrics = {}
    metrics["dead_pct"] = float((acts_test.sum(dim=0) == 0).float().mean().item() * 100)
    metrics["L0"] = float((acts_test > 1e-6).float().sum(dim=-1).mean().item())
    metrics["silhouette"] = silhouette(acts_test, test_labels)

    train_labels_arr = np.array(train_labels)
    label_counts = pd.Series(train_labels_arr).value_counts()
    usable_labels = label_counts[label_counts >= 10].index.tolist()
    acts_by_label_email = {lbl: acts_train[train_labels_arr == lbl] for lbl in usable_labels}
    acc_email, correct_email, y_email = cv_predictions(acts_by_label_email)
    metrics["clf_acc_email_axes"] = acc_email

    diff_labels_arr = np.array(diff_labels)
    en_mask, sp_mask = diff_labels_arr == "energy", diff_labels_arr == "sports"
    acts_by_label_diff = {"energy": acts_diff[en_mask], "sports": acts_diff[sp_mask]}
    acc_diff, correct_diff, y_diff = cv_predictions(acts_by_label_diff)
    metrics["clf_acc_sae"] = acc_diff

    return metrics, correct_email, correct_diff


def main():
    print("Reconstruction des splits (déterministe, aucun GPU/LLM)...")
    train_labels, test_labels, diff_labels = reconstruct_splits()
    n_train, n_test, n_diff = len(train_labels), len(test_labels), len(diff_labels)
    print(f"  train={n_train} test={n_test} diff={n_diff} (total={n_train + n_test + n_diff})")

    print(f"Chargement {ACTS_PATH} ...")
    acts = torch.load(ACTS_PATH, map_location="cpu", weights_only=True)
    assert acts.shape[0] == n_train + n_test + n_diff, (
        f"Désalignement lignes tenseur ({acts.shape[0]}) vs splits reconstruits "
        f"({n_train + n_test + n_diff}) -- vérifier CORPUS_SPLIT_SEED/N_TOTAL_* "
        f"avant de faire confiance à la suite.")
    d_total = acts.shape[1]
    d_core = d_total - D_EXTRA
    print(f"  acts: {tuple(acts.shape)} (d_core={d_core}, D_EXTRA={D_EXTRA})")

    acts_train_full = acts[:n_train]
    acts_test_full = acts[n_train:n_train + n_test]
    acts_diff_full = acts[n_train + n_test:]

    results = {}
    paired = {}
    for cond, sl in [("core", slice(0, d_core)), ("core_plus_extension", slice(0, d_total))]:
        print(f"\n=== Condition: {cond} (dims {sl.start}:{sl.stop}) ===")
        m, correct_email, correct_diff = run_condition(
            acts_train_full[:, sl], acts_test_full[:, sl], acts_diff_full[:, sl],
            test_labels, train_labels, diff_labels,
        )
        print(f"  {json.dumps(m, indent=2)}")
        results[cond] = m
        paired[cond] = {"correct_email": correct_email, "correct_diff": correct_diff}

    # McNemar apparié (mêmes folds/mêmes documents, seule la présence de
    # l'extension diffère) -- b/c = discordances core-réussit/ext-échoue et
    # inverse, PAS les totaux (cf. src/analysis/stats.py).
    def mcnemar_from_pairs(a, b):
        b_count = int(((a == 1) & (b == 0)).sum())
        c_count = int(((a == 0) & (b == 1)).sum())
        return paired_mcnemar_test(b_count, c_count), b_count, c_count

    mc_email, b_e, c_e = mcnemar_from_pairs(
        paired["core"]["correct_email"], paired["core_plus_extension"]["correct_email"])
    mc_diff, b_d, c_d = mcnemar_from_pairs(
        paired["core"]["correct_diff"], paired["core_plus_extension"]["correct_diff"])

    summary = {
        "metrics": results,
        "mcnemar_email_axes": {
            "b_core_only_correct": b_e, "c_extension_only_correct": c_e,
            "statistic": mc_email.statistic, "p": mc_email.p, "exact": mc_email.exact,
        },
        "mcnemar_diff_energy_sports": {
            "b_core_only_correct": b_d, "c_extension_only_correct": c_d,
            "statistic": mc_diff.statistic, "p": mc_diff.p, "exact": mc_diff.exact,
        },
    }
    out_json = os.path.join(CACHE_DIR, "core_vs_extension_pollution.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nRésumé JSON : {out_json}")

    df = pd.DataFrame([
        {"condition": cond, **m} for cond, m in results.items()
    ])
    plots_dir = os.path.join(RUN_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out_plot = os.path.join(plots_dir, "core_vs_extension_pollution.html")
    plot_metric_vs_hyperparam(
        df, "condition", ["clf_acc_email_axes", "clf_acc_sae", "silhouette", "dead_pct"],
        title="Core seul vs core+extension — métriques en aval (Phase 1, test de pollution)",
        x_is_categorical=True, path=out_plot,
    )
    print(f"Figure : {out_plot}")

    print(f"\nMcNemar (axes email) : b={b_e} c={c_e} p={mc_email.p:.4f}")
    print(f"McNemar (energy/sports) : b={b_d} c={c_d} p={mc_diff.p:.4f}")


if __name__ == "__main__":
    main()
