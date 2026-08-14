"""
scripts/audit_2026_08_palier1_batch.py — Vérifications Palier 1 de l'audit
`AUDIT_SCIENTIFIQUE_CODE.md` (docs/AUDIT_2026-08.md), toutes CPU-only sur
activations/checkpoints déjà en cache. Ne modifie aucun fichier de résultats
existant, ne modifie aucun code de production — script d'audit autonome,
même logique que `scripts/intent_urgency_probe.py`/`core_vs_extension_ablation.py`.

Regroupe trois vérifications indépendantes dans un seul job SLURM (cf. règle
CLAUDE.md : rien ne s'exécute sur le nœud frontal, même un test CPU trivial) :

1. B.26 — INTENT_KEYWORDS_FR (\\b(radical)\\b) : mesure l'écart n_pos avant/après
   correctif (\\w* ajouté) sur le corpus réel, et rejoue intent_urgency_probe.py
   avec les labels corrigés pour comparer aux deltas publiés (+27,0/+42,6 pts).
2. B.6 — fuite de groupe dans clf_acc_email_axes : reconstruit le parent_idx
   (répliqué depuis build_email_train_test_corpus, déterministe/même seed),
   compare StratifiedKFold (actuel) vs GroupKFold (corrigé) sur les mêmes
   activations en cache.
3. B.1 (point 1) — cohérence mutuelle du dictionnaire résiduel déjà entraîné
   (p1_frozen_core_d1024_k32.pt) : cosinus max hors diagonale de W_dec_extra,
   et dérive par rapport à une éventuelle init PCA (non reconstructible ici,
   on mesure seulement l'orthogonalité résiduelle du dictionnaire final).

Usage : sbatch slurm/validation/run_audit_palier1_batch.slurm
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_PATH = os.path.join(REPO_ROOT, "docs", "audit_palier1_batch_results.json")

results = {}

# ─────────────────────────────────────────────────────────────────────────
# 1. B.26 — INTENT_KEYWORDS_FR
# ─────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("B.26 — INTENT_KEYWORDS_FR : impact du correctif \\b(radical)\\b -> \\b(radical\\w*)\\b")
print("=" * 70)

import src.data.dataset as dataset_mod
from src.config import LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, SAVE_DIR, CORPUS_SPLIT_SEED

ORIG_PATTERNS = dict(dataset_mod.INTENT_KEYWORDS_FR)
FIXED_PATTERNS = {}
for k, v in ORIG_PATTERNS.items():
    assert v.endswith(r")\b"), v
    FIXED_PATTERNS[k] = v[:-3] + r"\w*)\b"

df_orig = dataset_mod.load_mails_tsv(LOCAL_MAILS_PATH)
b26_corpus_impact = {}
for intent in ORIG_PATTERNS:
    orig_mask = df_orig["text"].str.contains(ORIG_PATTERNS[intent], flags=re.I, regex=True)
    fixed_mask = df_orig["text"].str.contains(FIXED_PATTERNS[intent], flags=re.I, regex=True)
    flip_on = int((fixed_mask & ~orig_mask).sum())
    flip_off = int((orig_mask & ~fixed_mask).sum())
    b26_corpus_impact[intent] = {
        "n_pos_original": int(orig_mask.sum()),
        "n_pos_corrige": int(fixed_mask.sum()),
        "flip_on_document_level": flip_on,
        "flip_off_document_level": flip_off,
    }
    print(f"  {intent:15s} n_orig={orig_mask.sum():5d} n_fixed={fixed_mask.sum():5d} "
          f"flip_on={flip_on} flip_off={flip_off}")
results["B26_corpus_level_impact"] = b26_corpus_impact

# Rejoue intent_urgency_probe.py (labels corrigés) sur le cache du run publié
# (résultats_v10_emails_main, celui cité dans RESULTS_TESTS.md §13.2)
dataset_mod.INTENT_KEYWORDS_FR = FIXED_PATTERNS
from src.data.preparation import build_email_train_test_corpus
from src.analysis.metrics import downstream_classification

PROBE_SAVE_DIR = os.path.join(REPO_ROOT, "results_v10_emails_main")
PROBE_CACHE_DIR = os.path.join(PROBE_SAVE_DIR, "cache")
SEED = 42


def replicate_load_and_clean_emails_with_index(tsv_path):
    df = dataset_mod.load_mails_tsv(tsv_path).rename(columns={"text": "document"})
    kept = []
    for row_idx, row in df.iterrows():
        if "document" not in row or pd.isna(row["document"]):
            continue
        raw_text = str(row["document"])
        clean_text = re.sub(r'^\s*(?:Objet|Subject)\s*:\s*[^\n]+\n*', '', raw_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'\[\s*\{\s*"start".*?\}\s*\]', '', clean_text, flags=re.DOTALL).strip()
        if clean_text:
            kept.append(row_idx)
    return kept, df


kept_row_indices, df_full = replicate_load_and_clean_emails_with_index(LOCAL_MAILS_PATH)
n_real = len(kept_row_indices)
rng = np.random.default_rng(SEED)
test_mask = rng.random(n_real) < float(os.environ.get("EMAIL_TEST_SPLIT", "0.05"))
train_positions = [i for i in range(n_real) if not test_mask[i]]
k_train_original = len(train_positions)

train_texts, train_labels, _, _ = build_email_train_test_corpus(
    LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED,
)
n_original_train = sum(1 for l in train_labels if l == "original")
assert n_original_train == k_train_original, (
    f"Incohérence {n_original_train} != {k_train_original} -- correspondance rompue.")

acts_path = os.path.join(PROBE_CACHE_DIR, "p1_all_doc_acts_ext_d1024.pt")
all_doc_acts = torch.load(acts_path, map_location="cpu", weights_only=True)
original_train_acts = all_doc_acts[:k_train_original]

row_indices_for_train_original = [kept_row_indices[i] for i in train_positions]
intent_cols = [c for c in df_full.columns if c.startswith("intent_")]
labels_df = df_full.loc[row_indices_for_train_original, intent_cols].reset_index(drop=True)

b26_probe_corrected = {}
for col in intent_cols:
    y = labels_df[col].to_numpy()
    n_pos = int(y.sum())
    if n_pos < 10 or n_pos > len(y) - 10:
        print(f"  {col}: {n_pos}/{len(y)} -- ignoré.")
        continue
    pos_mask = torch.from_numpy(y.astype(bool))
    clf = downstream_classification(acts_by_label={
        "positif": original_train_acts[pos_mask],
        "negatif": original_train_acts[~pos_mask],
    })
    majority_baseline = max(n_pos, len(y) - n_pos) / len(y)
    b26_probe_corrected[col] = {
        "n_pos": n_pos, "n_total": len(y),
        "acc_sae": clf["acc_sae"], "majority_baseline": majority_baseline,
        "delta_pts": 100 * (clf["acc_sae"] - majority_baseline),
    }
    print(f"  {col}: n_pos={n_pos}/{len(y)} acc_SAE={clf['acc_sae']:.4f} "
          f"baseline={majority_baseline:.4f} delta={100*(clf['acc_sae']-majority_baseline):+.1f}pts")
results["B26_probe_corrected_labels"] = b26_probe_corrected
dataset_mod.INTENT_KEYWORDS_FR = ORIG_PATTERNS  # restore

# ─────────────────────────────────────────────────────────────────────────
# 2. B.6 — GroupKFold vs StratifiedKFold sur clf_acc_email_axes
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("B.6 — Fuite de groupe : StratifiedKFold (actuel) vs GroupKFold (parent-aware)")
print("=" * 70)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score


def build_email_corpus_with_parent_idx(mails_tsv_path, augmented_jsonl_path,
                                        test_split=0.05, max_augmented_per_mail=13, seed=42):
    """Duplique build_email_train_test_corpus (src/data/preparation.py) à
    l'identique (même appels rng, même ordre) en conservant en plus le
    parent_idx de chaque ligne train -- non modifié en production, seule la
    sortie est étendue, cf. principe déjà appliqué par intent_urgency_probe.py."""
    from src.data.preparation import load_and_clean_emails
    real_texts, _ = load_and_clean_emails(mails_tsv_path)
    n_real = len(real_texts)
    rng = np.random.default_rng(seed)
    test_mask = rng.random(n_real) < test_split
    parent_split = {i: ("test" if test_mask[i] else "train") for i in range(n_real)}

    train_texts, train_labels, train_parent = [], [], []
    for i, text in enumerate(real_texts):
        if parent_split[i] == "train":
            train_texts.append(text)
            train_labels.append("original")
            train_parent.append(i)

    if augmented_jsonl_path and os.path.exists(augmented_jsonl_path):
        try:
            from src.data.augmentation import load_augmented
        except ImportError:
            from augmentation import load_augmented
        df_aug = load_augmented(augmented_jsonl_path)
        df_aug = df_aug[df_aug["text"].notna()].copy()
        df_aug["parent_idx"] = df_aug["parent_id"].astype(int)
        df_aug = df_aug[df_aug["parent_idx"] < n_real]
        if max_augmented_per_mail:
            sampled_idx = np.concatenate([
                rng.choice(group.index.to_numpy(), size=min(len(group), max_augmented_per_mail), replace=False)
                for _, group in df_aug.groupby("parent_idx")
            ])
            df_aug = df_aug.loc[sampled_idx]
        for row in df_aug.itertuples(index=False):
            split = parent_split.get(int(row.parent_idx), "train")
            if split == "train":
                train_texts.append(row.text)
                train_labels.append(f"{row.aug_axis}__{row.aug_level}")
                train_parent.append(int(row.parent_idx))
    return train_texts, train_labels, train_parent


_, train_labels_full, train_parent_full = build_email_corpus_with_parent_idx(
    LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED,
)
n_train_check = len(train_labels_full)

ABLATION_ACTS_PATH = os.path.join(PROBE_CACHE_DIR, "p1_all_doc_acts_ext_d1024.pt")
acts_all = torch.load(ABLATION_ACTS_PATH, map_location="cpu", weights_only=True)
# acts_all = [train | test | diff] dans cet ordre (cf. core_vs_extension_ablation.py) ;
# on ne connaît ici que n_train_check via la reconstruction -- vérification de
# cohérence avant de faire confiance à l'alignement.
print(f"  n_train reconstruit (avec parent_idx) = {n_train_check}, "
      f"acts_all.shape[0] = {acts_all.shape[0]}")

train_labels_arr = np.array(train_labels_full)
train_parent_arr = np.array(train_parent_full)
label_counts = pd.Series(train_labels_arr).value_counts()
usable_labels = label_counts[label_counts >= 10].index.tolist()
mask_usable = np.isin(train_labels_arr, usable_labels)

acts_train = acts_all[:n_train_check]
X = acts_train[mask_usable].float().numpy()
labels_used = train_labels_arr[mask_usable]
groups_used = train_parent_arr[mask_usable]
le_map = {lbl: i for i, lbl in enumerate(sorted(set(labels_used)))}
y = np.array([le_map[l] for l in labels_used])

print(f"  n usable rows = {len(y)}, n classes = {len(le_map)}, "
      f"n groupes (mails source) uniques = {len(set(groups_used))}")


def cv_acc(splitter_iter):
    accs = []
    for train_idx, test_idx in splitter_iter:
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], preds))
    return accs


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accs_stratified = cv_acc(skf.split(X, y))

gkf = GroupKFold(n_splits=5)
accs_group = cv_acc(gkf.split(X, y, groups=groups_used))

results["B6_group_leak"] = {
    "n_rows": int(len(y)), "n_classes": int(len(le_map)),
    "n_groups": int(len(set(groups_used))),
    "acc_stratified_current": accs_stratified,
    "acc_stratified_mean": float(np.mean(accs_stratified)),
    "acc_groupkfold_corrected": accs_group,
    "acc_groupkfold_mean": float(np.mean(accs_group)),
    "delta_pts": float(100 * (np.mean(accs_stratified) - np.mean(accs_group))),
}
print(f"  StratifiedKFold (actuel)  : acc={np.mean(accs_stratified):.4f} (folds={[f'{a:.3f}' for a in accs_stratified]})")
print(f"  GroupKFold (parent-aware) : acc={np.mean(accs_group):.4f} (folds={[f'{a:.3f}' for a in accs_group]})")
print(f"  Écart : {100*(np.mean(accs_stratified)-np.mean(accs_group)):+.1f} pts")

# ─────────────────────────────────────────────────────────────────────────
# 3. B.1 (point 1) — cohérence mutuelle du dictionnaire résiduel
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("B.1 (1) — Cohérence mutuelle de W_dec_extra (checkpoint entraîné, D_EXTRA=1024)")
print("=" * 70)

ckpt_path = os.path.join(PROBE_SAVE_DIR, "p1_frozen_core_d1024_k32.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
W_dec = state_dict["W_dec_extra"].float()
W_dec_n = torch.nn.functional.normalize(W_dec, dim=1)
cos = W_dec_n @ W_dec_n.T
d = cos.shape[0]
off_diag = cos[~torch.eye(d, dtype=torch.bool)]
results["B1_dictionary_coherence"] = {
    "d_extra": int(d),
    "max_abs_cosine_off_diag": float(off_diag.abs().max().item()),
    "mean_abs_cosine_off_diag": float(off_diag.abs().mean().item()),
    "frac_pairs_above_0.5": float((off_diag.abs() > 0.5).float().mean().item()),
    "frac_pairs_above_0.9": float((off_diag.abs() > 0.9).float().mean().item()),
}
print(json.dumps(results["B1_dictionary_coherence"], indent=2))

# ─────────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n[+] Écrit : {OUT_PATH}")
