"""
scripts/post_stage/e06_correlations.py -- Correlations entre proprietes,
confirmees hors decouverte (Plan_execution_SAE_15_jours_Claude_Code.md §11).

Decouverte (candidats) sur FIT+DEV : NPMI par-parent sur le catalogue de
features interpretables d'E02 (CORE+EXTRA). Filtre support conjoint >= 10
parents, retrait des quasi-synonymes (recouvrement lexical label+description),
gel de 8 paires. Verification sur un sous-echantillon de CONFIRM (parents
jamais vus en decouverte) : chaque propriete atomique jugee individuellement
par Qwen (pas la paire directement), cooccurrence calculee SUR CES JUGEMENTS,
pas sur les activations SAE brutes -- coherent avec le protocole du plan
(§11, etape 5).

Simplifications assumees, documentees explicitement :
- Filtre quasi-synonymes par recouvrement lexical (Jaccard de tokens sur
  label+description), pas par embedding dense -- suffisant a ce budget,
  pas de modele supplementaire charge pour ce seul filtre.
- Baseline de cooccurrence lexicale FIT/DEV (plan : "souhaitable") non
  calculee ici, faute de budget de session pour la prioriser sur le reste
  du plan.
- Audit humain stratifie 40-80 emails (plan §11) non fait -- necessite
  Gregoire, reste en attente.
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import fisher_exact

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src", "sae"))

from src.post_stage.dataset_contract import (  # noqa: E402
    load_fit_dev_corpus_from_manifest, load_confirm_corpus_from_manifest,
)
from src.sae.sae_shared import load_all_doc_acts  # noqa: E402
from src.sae.judge import load_judge_model, _batched_generate  # noqa: E402
from src.analysis.stats import bootstrap_ci_by_group, fdr_bh  # noqa: E402

MIN_JOINT_SUPPORT_PARENTS = 10
N_FROZEN_PAIRS = 8
SYNONYM_JACCARD_THRESHOLD = 0.5


def _parent_level_presence(acts: np.ndarray, groups: list) -> tuple:
    """Agrege docs -> parents : un parent 'a' la propriete si AU MOINS une de
    ses variantes (mail original + augmentations) l'active. Retourne
    (presence [n_parents, n_features] bool, parent_ids ordonnes)."""
    groups = np.asarray(groups)
    parent_ids = sorted(set(groups.tolist()))
    idx = {p: i for i, p in enumerate(parent_ids)}
    n_parents, n_feat = len(parent_ids), acts.shape[1]
    presence = np.zeros((n_parents, n_feat), dtype=bool)
    binarized = acts > 1e-6
    for row, g in enumerate(groups):
        presence[idx[g]] |= binarized[row]
    return presence, parent_ids


def _npmi_pairwise(presence: np.ndarray) -> np.ndarray:
    n = presence.shape[0]
    b = presence.astype(np.float64)
    cooc = b.T @ b
    p_ij = cooc / n
    p_i = b.mean(0)
    eps = 1e-12
    pmi = np.log((p_ij + eps) / (np.outer(p_i, p_i) + eps))
    npmi = pmi / (-np.log(p_ij + eps))
    npmi = np.where(cooc > 0, npmi, 0.0)
    np.fill_diagonal(npmi, 1.0)
    return npmi, cooc


def _label_tokens(feat: dict) -> set:
    text = f"{feat.get('label', '')} {feat.get('brief_description', '')}".lower()
    return set(t for t in text.replace(",", " ").replace(".", " ").split() if len(t) > 3)


def _judge_presence(model, tokenizer, definition: str, doc_text: str, batch_size: int) -> int:
    prompt = (
        "Voici la definition d'une propriete que peut avoir un email client.\n\n"
        f"Propriete : {definition}\n\n"
        f"Email :\n{doc_text[:1500]}\n\n"
        "L'email presente-t-il cette propriete ? Reponds uniquement par 'oui' ou 'non'."
    )
    resp = _batched_generate(model, tokenizer, [[{"role": "user", "content": prompt}]],
                              max_new_tokens=4, batch_size=batch_size)[0]
    return 1 if "oui" in resp.strip().lower()[:10] else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--mails-tsv-path", default="local_data/emails/Mails.tsv")
    ap.add_argument("--augmented-jsonl-path", default="local_data/emails/augmented_mails.jsonl")
    ap.add_argument("--split-assignments", required=True)
    ap.add_argument("--registry-path", required=True)
    ap.add_argument("--d-extra", type=int, required=True)
    ap.add_argument("--max-augmented-per-mail", type=int, default=13)
    ap.add_argument("--n-confirm-parents", type=int, default=350)
    ap.add_argument("--judge-device", default="cuda")
    ap.add_argument("--out", default="e06_correlations.json")
    args = ap.parse_args()
    t0 = time.time()

    print("[e06_correlations] Chargement FIT/DEV (decouverte)...", flush=True)
    fit_texts, fit_labels, dev_texts, dev_labels, fit_groups, dev_groups = load_fit_dev_corpus_from_manifest(
        args.split_assignments, args.mails_tsv_path, args.augmented_jsonl_path,
        max_augmented_per_mail=args.max_augmented_per_mail, return_groups=True,
    )
    n_fit, n_dev = len(fit_texts), len(dev_texts)
    print(f"[e06_correlations] FIT={n_fit} DEV={n_dev}", flush=True)

    with open(args.registry_path) as f:
        registry = json.load(f)["features"]
    catalog = {v["global_index"]: v for v in registry.values() if v["status"] == "interpretable"}
    catalog_idx = sorted(catalog.keys())
    print(f"[e06_correlations] Catalogue : {len(catalog_idx)} features interpretables (CORE+EXTRA).", flush=True)

    acts_path = os.path.join(args.save_dir, "cache", f"p1_all_doc_acts_ext_d{args.d_extra}.pt")
    all_doc_acts = load_all_doc_acts(acts_path)
    assert all_doc_acts.shape[0] >= n_fit + n_dev, (
        f"all_doc_acts a {all_doc_acts.shape[0]} lignes, attendu au moins {n_fit + n_dev}")
    fitdev_acts = all_doc_acts[:n_fit + n_dev][:, catalog_idx].numpy()
    del all_doc_acts
    fitdev_groups = list(fit_groups) + list(dev_groups)

    presence, parent_ids = _parent_level_presence(fitdev_acts, fitdev_groups)
    print(f"[e06_correlations] {len(parent_ids)} parents FIT+DEV, presence binarisee.", flush=True)

    npmi, cooc = _npmi_pairwise(presence)
    freq = presence.mean(0)

    # Candidats : support conjoint (parents) >= seuil, hors quasi-synonymes.
    candidates = []
    for i, j in itertools.combinations(range(len(catalog_idx)), 2):
        n_ab = int(cooc[i, j])
        if n_ab < MIN_JOINT_SUPPORT_PARENTS:
            continue
        feat_a, feat_b = catalog[catalog_idx[i]], catalog[catalog_idx[j]]
        toks_a, toks_b = _label_tokens(feat_a), _label_tokens(feat_b)
        union = toks_a | toks_b
        jaccard = len(toks_a & toks_b) / len(union) if union else 0.0
        if jaccard >= SYNONYM_JACCARD_THRESHOLD:
            continue
        candidates.append({
            "i": i, "j": j, "feature_uid_a": feat_a["feature_uid"], "feature_uid_b": feat_b["feature_uid"],
            "label_a": feat_a["label"], "label_b": feat_b["label"],
            "definition_a": feat_a["brief_description"], "definition_b": feat_b["brief_description"],
            "branch_a": feat_a["branch"], "branch_b": feat_b["branch"],
            "n_joint_parents": n_ab, "npmi_discovery": float(npmi[i, j]),
            "freq_a": float(freq[i]), "freq_b": float(freq[j]),
        })
    print(f"[e06_correlations] {len(candidates)} paires candidates (support>=10, hors quasi-synonymes) "
          f"sur {len(catalog_idx)*(len(catalog_idx)-1)//2} possibles.", flush=True)

    candidates.sort(key=lambda c: abs(c["npmi_discovery"]), reverse=True)
    frozen = candidates[:N_FROZEN_PAIRS]
    n_core_involved = sum(1 for c in frozen if "core" in (c["branch_a"], c["branch_b"]))
    n_extra_only = sum(1 for c in frozen if c["branch_a"] == "extra" and c["branch_b"] == "extra")
    print(f"[e06_correlations] {len(frozen)} paires gelees (impliquant CORE : {n_core_involved}, "
          f"EXTRA seul : {n_extra_only}). Gelees AVANT lecture de CONFIRM.", flush=True)
    for c in frozen:
        print(f"    - NPMI={c['npmi_discovery']:.2f} n_joint={c['n_joint_parents']} : "
              f"'{c['label_a']}' x '{c['label_b']}'", flush=True)

    if not frozen:
        out = {
            "n_fit": n_fit, "n_dev": n_dev, "n_candidates": len(candidates), "frozen_pairs": [],
            "status": "no_pair_survived_discovery_filters",
            "elapsed_seconds": round(time.time() - t0, 1),
        }
        out_path = os.path.join(args.save_dir, args.out)
        with open(out_path + ".tmp", "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        os.replace(out_path + ".tmp", out_path)
        print(f"[e06_correlations] Aucune paire n'a survecu aux filtres de decouverte. Écrit {out_path}", flush=True)
        return 0

    print("[e06_correlations] Chargement CONFIRM (verification -- jamais vu en decouverte)...", flush=True)
    confirm_texts, confirm_labels, confirm_groups = load_confirm_corpus_from_manifest(
        args.split_assignments, args.mails_tsv_path, args.augmented_jsonl_path,
        max_augmented_per_mail=args.max_augmented_per_mail, return_groups=True,
    )
    confirm_groups = np.asarray(confirm_groups)
    rng = np.random.default_rng(42)
    unique_parents = np.array(sorted(set(confirm_groups.tolist())))
    n_sample = min(args.n_confirm_parents, len(unique_parents))
    sampled_parents = set(rng.choice(unique_parents, size=n_sample, replace=False).tolist())
    rep_doc_idx = {}
    for i, g in enumerate(confirm_groups):
        if g in sampled_parents and g not in rep_doc_idx:
            rep_doc_idx[g] = i
    sampled_parent_ids = sorted(rep_doc_idx.keys())
    print(f"[e06_correlations] {len(sampled_parent_ids)} parents CONFIRM echantillonnes "
          f"(1 email representatif chacun, graine 42).", flush=True)

    unique_props = {}
    for c in frozen:
        unique_props[c["feature_uid_a"]] = c["definition_a"]
        unique_props[c["feature_uid_b"]] = c["definition_b"]
    prop_uids = sorted(unique_props.keys())
    print(f"[e06_correlations] {len(prop_uids)} proprietes atomiques uniques a juger x "
          f"{len(sampled_parent_ids)} parents = {len(prop_uids)*len(sampled_parent_ids)} appels.", flush=True)

    print("[e06_correlations] Chargement du juge Qwen...", flush=True)
    judge_model, judge_tokenizer = load_judge_model(device=args.judge_device)

    judged = {uid: {} for uid in prop_uids}
    for uid in prop_uids:
        definition = unique_props[uid]
        for pid in sampled_parent_ids:
            doc_text = confirm_texts[rep_doc_idx[pid]]
            judged[uid][pid] = _judge_presence(judge_model, judge_tokenizer, definition, doc_text, batch_size=8)
        rate = np.mean(list(judged[uid].values()))
        print(f"[e06_correlations]   '{uid}' : taux de presence verifie = {rate:.2f}", flush=True)

    del judge_model
    torch.cuda.empty_cache()

    print("[e06_correlations] Cooccurrence calculee SUR LES JUGEMENTS CONFIRM...", flush=True)
    results = []
    for c in frozen:
        a = np.array([judged[c["feature_uid_a"]][pid] for pid in sampled_parent_ids])
        b = np.array([judged[c["feature_uid_b"]][pid] for pid in sampled_parent_ids])
        n = len(sampled_parent_ids)
        n_a, n_b = int(a.sum()), int(b.sum())
        n_ab = int(((a == 1) & (b == 1)).sum())
        p_a, p_b, p_ab = n_a / n, n_b / n, n_ab / n
        table = np.array([[n_ab, n_a - n_ab], [n_b - n_ab, n - n_a - n_b + n_ab]])
        insufficient = (n_a < 5) or (n_b < 5) or (n_a > n - 5) or (n_b > n - 5)
        odds_ratio, fisher_p = fisher_exact(table)
        if 0 < p_ab < 1:
            eps = 1e-12
            pmi = np.log((p_ab + eps) / (p_a * p_b + eps))
            npmi_confirm = float(pmi / (-np.log(p_ab + eps)))
        else:
            npmi_confirm = None
        ci = None
        if not insufficient:
            # IC bootstrap du NPMI lui-meme (pas seulement de p_ab) : chaque
            # reechantillon recalcule p_a/p_b/p_ab sur les paires (a,b)
            # rééchantillonnées, puis la formule NPMI complete -- un
            # bootstrap de la seule moyenne de l'indicatrice AB donnerait
            # l'IC de p_ab, pas celui du NPMI (rapport non lineaire des trois
            # proportions).
            def _npmi_stat(vals: np.ndarray) -> float:
                aa, bb = vals[:, 0], vals[:, 1]
                pa_, pb_, pab_ = aa.mean(), bb.mean(), (aa * bb).mean()
                eps = 1e-12
                if not (0 < pab_ < 1):
                    return 0.0
                pmi_ = np.log((pab_ + eps) / (pa_ * pb_ + eps))
                return float(pmi_ / (-np.log(pab_ + eps)))

            paired_vals = np.stack([a.astype(np.float64), b.astype(np.float64)], axis=1)
            ci_result = bootstrap_ci_by_group(paired_vals, np.array(sampled_parent_ids),
                                               statistic=_npmi_stat, n_boot=2000, seed=42)
            ci = {"ci_low": ci_result.ci_low, "ci_high": ci_result.ci_high}
        results.append({
            "label_a": c["label_a"], "label_b": c["label_b"],
            "branch_a": c["branch_a"], "branch_b": c["branch_b"],
            "npmi_discovery_fitdev": c["npmi_discovery"], "n_joint_parents_discovery": c["n_joint_parents"],
            "n_confirm": n, "n_a_confirm": n_a, "n_b_confirm": n_b, "n_ab_confirm": n_ab,
            "p_ab_minus_pa_pb": float(p_ab - p_a * p_b),
            "odds_ratio": float(odds_ratio) if np.isfinite(odds_ratio) else None,
            "fisher_p": float(fisher_p),
            "npmi_confirm": npmi_confirm,
            "npmi_confirm_bootstrap_ci_by_parent": ci,
            "status": "insufficient_support" if insufficient else "tested",
        })

    tested = [r for r in results if r["status"] == "tested"]
    if tested:
        pvals = [r["fisher_p"] for r in tested]
        qvals = fdr_bh(pvals)
        for r, q in zip(tested, qvals):
            r["fisher_p_fdr_bh"] = float(q)

    out = {
        "n_fit": n_fit, "n_dev": n_dev, "n_candidates_discovery": len(candidates),
        "frozen_pairs": frozen, "n_confirm_parents_sampled": len(sampled_parent_ids),
        "confirmation": results,
        "human_audit_pending": True,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    out_path = os.path.join(args.save_dir, args.out)
    with open(out_path + ".tmp", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    os.replace(out_path + ".tmp", out_path)
    print(f"[e06_correlations] Écrit {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
