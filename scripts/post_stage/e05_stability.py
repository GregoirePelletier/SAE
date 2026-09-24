"""
scripts/post_stage/e05_stability.py -- Stabilite inter-graines des features
EXTRA, groupes et sous-espaces (docs/post_stage/PLAN_E00-E09.md
§10). CPU uniquement (aucun modele charge), a lancer via sbatch.

Trois extensions entrainees sur les MEMES donnees FIT, meme architecture et
hyperparametres, graines 42/43/44 (le coeur GemmaScope-2 gele est identique
par construction -- jamais compare, plan §10.1). Portee : branche EXTRA
seulement, meme espace 1B/couche13.

Composantes :
  1. Appariement individuel (chaque ordre de paire) : plus proche voisin par
     cosinus SIGNE des directions du decodeur (plusieurs-a-un autorise),
     accompagne d'un controle de profils (Pearson des activations DEV
     max-poolees par document), du recouvrement top-20 documents, de la
     reciprocite et du ratio de frequence. Temoins : dictionnaire nul
     gaussien a covariance anisotrope reelle (cosinus) et appariements
     melanges (profils).
  2. Groupes par dictionnaire sur DEV : voisinages mutuels k=10 (cosinus
     decodeur), aretes conservees si le profil DEV correle au-dela du q99 d'un
     temoin de permutation colonne par colonne, Louvain a UNE resolution,
     isolats conserves, aucune fusion forcee.
  3. Comparaison des groupes entre graines par correspondances individuelles
     (fusions/splits/absence autorises via la purete), recouvrement de
     sous-espaces (rang fixe a l'avance, jamais cache), et Jaccard@20 des
     documents les plus activants du score de groupe (max des membres
     normalises par leur p90 FIT). Chaque metrique est comparee a 100 groupes
     aleatoires de meme taille, strates de frequence appariees, passes par la
     MEME procedure de partenaire (meilleur appariement contre meilleur
     appariement) ; FDR-BH sur les groupes de chaque paire.

Simplifications assumees, documentees explicitement :
- Profils = activations DOCUMENT-level (max-pool par document, DEV 6518
  docs), pas les 50-100k tokens partages du plan (aurait exige un
  reencodage token-level des trois SAE).
- Permutation temoin colonne par colonne sur les documents, non stratifiee
  par longueur/blocs parents.
- LIMITE STRUCTURELLE : SAEBoostResidualSAE initialise le decodeur EXTRA par
  les 1024 premieres directions PCA du residu (frozen_core.py::
  _init_from_residual_pca, "1024 directions PCA injectees" dans les logs) sur
  le reservoir PARTAGE : l'initialisation est identique pour les 3 graines,
  qui ne different que par l'ordre des mini-lots. Ce script mesure donc la
  robustesse a l'ordre d'entrainement depuis une init PCA commune, PAS
  l'independance a une initialisation aleatoire (bras non disponible sans
  modifier le pipeline).
- Score de groupe : max (principal, sature pour les grands groupes) ET
  moyenne des membres actifs (diagnostic secondaire du plan §10.7), Jaccard
  @20 et @100 et Spearman sur les 6518 documents DEV.
- 3 graines seulement : "retrouvee dans les deux repetitions disponibles",
  jamais "reproductible a 100 %" ; les features ne sont PAS des
  entrainements independants (aucun bootstrap de features).
- Pas de lien aux requetes E03 (Jaccard@20/P@10 de themes nommes) : les
  groupes sont mesures tels quels, non nommes ; pas d'audit humain.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src", "sae"))

from src.analysis.stats import fdr_bh  # noqa: E402
from src.post_stage.dataset_contract import load_fit_dev_corpus_from_manifest  # noqa: E402
from src.post_stage.stability import (  # noqa: E402
    anisotropic_null_dictionary, group_partner, louvain_labels, matched_column_corr,
    mutual_knn_edges, nn_match, resample_group_same_strata, subspace_basis,
    subspace_overlap, topk_jaccard,
)
from src.sae.sae_shared import load_all_doc_acts  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

COS_THRESHOLD = 0.7          # repere de litterature, plan §10.3
PROFILE_CORR_THRESHOLD = 0.5  # controle de profils declare avec le seuil cosinus
MIN_ACTIVE_FIT = 50
MAX_FREQ_FIT = 0.5
KNN_K = 10
MIN_GROUP_SIZE = 3
OVERSIZED_GROUP = 150
N_NULL_GROUPS = 100
N_NULL_DICTS = 20
JACCARD_K = 20
N_STRATA = 5


def _find_decoder(state_dict: dict) -> np.ndarray:
    keys = [k for k in state_dict if k.endswith("W_dec_extra")]
    assert len(keys) == 1, f"attendu exactement une cle *W_dec_extra, trouve {keys}"
    return state_dict[keys[0]].float().cpu().numpy()


QUALITY_KEYS = ("fve_pretrained", "fve_extended", "dead_pct_extension", "rho_sae", "clf_acc_email_axes")


class Run:
    def __init__(self, name, save_dir, d_extra, n_fit, n_dev, arm="pca"):
        self.name, self.save_dir, self.arm = name, save_dir, arm
        with open(os.path.join(save_dir, "results.json")) as f:
            res = json.load(f)["P1_Gemma3_SAE"]
        self.quality = {k: res.get(k) for k in QUALITY_KEYS}
        ckpt = torch.load(os.path.join(save_dir, "p1_extended_sae.pt"), map_location="cpu")
        self.W = _find_decoder(ckpt["state_dict"])
        assert self.W.shape[0] == d_extra, (name, self.W.shape)
        acts = load_all_doc_acts(os.path.join(save_dir, "cache", f"p1_all_doc_acts_ext_d{d_extra}.pt"))
        assert acts.shape[0] >= n_fit + n_dev, (name, acts.shape, n_fit, n_dev)
        d_core = acts.shape[1] - d_extra
        self.fit = acts[:n_fit, d_core:].float().numpy()
        self.dev = acts[n_fit:n_fit + n_dev, d_core:].float().numpy()
        del acts
        act_fit = self.fit > 1e-6
        self.n_active_fit = act_fit.sum(axis=0)
        self.freq_fit = act_fit.mean(axis=0)
        self.p90_fit = np.full(d_extra, np.nan)
        for f in range(d_extra):
            col = self.fit[:, f]
            nz = col[col > 1e-6]
            if len(nz) >= 20:
                self.p90_fit[f] = np.percentile(nz, 90)
        self.supported = (self.n_active_fit >= MIN_ACTIVE_FIT) & (self.freq_fit <= MAX_FREQ_FIT)
        self.sup = np.where(self.supported)[0]
        fq = self.freq_fit[self.sup]
        edges = np.quantile(fq, np.linspace(0, 1, N_STRATA + 1)[1:-1])
        self.strata = np.digitize(fq, edges)          # strate par POSITION dans self.sup
        self.dev_sup = self.dev[:, self.sup]
        self.W_sup = self.W[self.sup]
        self.p90_sup = self.p90_fit[self.sup]

    def uid(self, orig_idx):
        return f"{self.name}:extra:{int(orig_idx)}"

    def status_counts(self):
        n = len(self.freq_fit)
        return {
            "arm_decoder_init": self.arm, "training_quality": self.quality,
            "n_extra": n, "n_supported": int(self.supported.sum()),
            "n_insufficient_support(<%d actifs FIT)" % MIN_ACTIVE_FIT: int((self.n_active_fit < MIN_ACTIVE_FIT).sum()),
            "n_near_universal(freq FIT>%.1f)" % MAX_FREQ_FIT: int((self.freq_fit > MAX_FREQ_FIT).sum()),
        }


def _q(x, qs=(5, 25, 50, 75, 95)):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {f"q{q}": None for q in qs}
    return {f"q{q}": float(np.percentile(x, q)) for q in qs}


def match_pair(A: Run, B: Run, rng):
    """Appariement A->B dans l'espace des positions `sup` de chacun."""
    nn, cos = nn_match(A.W_sup, B.W_sup)
    ia, jb = A.sup, B.sup[nn]
    corr = matched_column_corr(A.dev, B.dev, ia, jb)
    jac = np.array([topk_jaccard(A.dev[:, i], B.dev[:, j], JACCARD_K) for i, j in zip(ia, jb)])
    nn_back, _ = nn_match(B.W_sup, A.W_sup)
    mutual = nn_back[nn] == np.arange(len(nn))
    null_cos = np.concatenate([nn_match(A.W_sup, anisotropic_null_dictionary(B.W_sup, len(B.sup), rng))[1]
                               for _ in range(N_NULL_DICTS)])
    corr_shuf = matched_column_corr(A.dev, B.dev, ia, B.sup[rng.permutation(nn)])
    joint = (cos >= COS_THRESHOLD) & (corr >= PROFILE_CORR_THRESHOLD)
    summary = {
        "n_source_supported": int(len(ia)), "n_candidates_supported": int(len(B.sup)),
        "best_cos_quantiles": _q(cos), "frac_cos_ge_0.7": float((cos >= COS_THRESHOLD).mean()),
        "frac_cos_ge_0.5": float((cos >= 0.5).mean()),
        "null_anisotropic_best_cos_quantiles": _q(null_cos),
        "null_frac_cos_ge_0.7": float((null_cos >= COS_THRESHOLD).mean()),
        "null_frac_cos_ge_0.5": float((null_cos >= 0.5).mean()),
        "profile_corr_quantiles": _q(corr), "n_profile_corr_undefined": int(np.isnan(corr).sum()),
        "frac_profile_corr_ge_0.5": float(np.nanmean(corr >= PROFILE_CORR_THRESHOLD)),
        "shuffled_match_profile_corr_quantiles": _q(corr_shuf),
        "frac_joint_cos0.7_and_corr0.5": float(joint.mean()),
        "mean_top20_jaccard": float(jac.mean()),
        "frac_mutual_nn": float(mutual.mean()),
        "median_freq_ratio_B_over_A": float(np.median(B.freq_fit[jb] / np.maximum(A.freq_fit[ia], 1e-9))),
    }
    return {"nn": nn, "cos": cos, "corr": corr, "jac": jac, "joint": joint, "summary": summary}


def build_groups(R: Run, rng):
    n = len(R.sup)
    edges = mutual_knn_edges(R.W_sup, KNN_K)
    ei = np.array([e[0] for e in edges], dtype=int)
    ej = np.array([e[1] for e in edges], dtype=int)
    ec = np.array([e[2] for e in edges])
    corr_e = matched_column_corr(R.dev_sup, R.dev_sup, ei, ej)
    perm_dev = np.empty_like(R.dev_sup)
    for c in range(n):
        perm_dev[:, c] = rng.permutation(R.dev_sup[:, c])
    null_corr = matched_column_corr(perm_dev, perm_dev, ei, ej)
    tau = float(np.nanquantile(null_corr, 0.99))
    keep = (corr_e > tau) & ~np.isnan(corr_e)
    kept = [(int(i), int(j), float(max(c, 0.0) + 1e-6)) for i, j, c, k in zip(ei, ej, ec, keep) if k]
    labels = louvain_labels(n, kept, seed=0)
    sizes = np.bincount(labels)
    groups = []
    for gid in np.where(sizes >= MIN_GROUP_SIZE)[0]:
        members = np.where(labels == gid)[0]
        mset = set(members.tolist())
        inner = [(c, cr) for (i, j, c, k, cr) in zip(ei, ej, ec, keep, corr_e)
                 if k and i in mset and j in mset]
        groups.append({
            "group_id": int(gid), "size": int(len(members)),
            "oversized(>%d, non scinde)" % OVERSIZED_GROUP: bool(len(members) > OVERSIZED_GROUP),
            "members": [R.uid(R.sup[m]) for m in members],
            "mean_edge_decoder_cos": float(np.mean([c for c, _ in inner])) if inner else None,
            "mean_edge_profile_corr": float(np.mean([cr for _, cr in inner])) if inner else None,
        })
    return labels, {
        "n_supported_nodes": int(n), "n_mutual_knn_edges": int(len(edges)),
        "profile_corr_edge_threshold_tau_q99_permutation_witness": tau,
        "n_edges_kept": int(len(kept)),
        "n_groups(size>=%d)" % MIN_GROUP_SIZE: int(len(groups)),
        "n_isolates(size=1)": int((sizes == 1).sum()),
        "n_features_in_groups": int(sum(g["size"] for g in groups)),
        "max_group_size": int(sizes.max()) if len(sizes) else 0,
        "groups": groups,
    }


SCORE_TYPES = ("max", "mean_active")
METRICS = ["purity", "overlap"] + [f"{st}_{m}" for st in SCORE_TYPES for m in ("top20_jaccard", "top100_jaccard", "spearman")]


def _group_scores(R: Run, members) -> dict:
    """Deux scores documentaires de groupe (plan §10.7), fixes a l'avance :
    max des membres normalises par leur p90 FIT (score principal, sature pour
    les grands groupes) et moyenne des membres ACTIFS (diagnostic secondaire)."""
    raw = R.dev_sup[:, members]
    X = raw / R.p90_sup[members]
    n_act = (raw > 1e-6).sum(axis=1)
    return {"max": X.max(axis=1), "mean_active": np.where(n_act > 0, X.sum(axis=1) / np.maximum(n_act, 1), 0.0)}


def _evaluate_group(A: Run, B: Run, members, nn, labels_b):
    partner, purity, n_matched = group_partner(members, nn, labels_b)
    pm = np.where(labels_b == partner)[0]
    QA, _ = subspace_basis(A.W_sup[members])
    QB, _ = subspace_basis(B.W_sup[pm])
    overlap, r = subspace_overlap(QA, QB)
    out = {"partner_group": partner, "partner_size": int(len(pm)), "purity": purity,
           "overlap": overlap, "overlap_rank": int(r)}
    sa, sb = _group_scores(A, members), _group_scores(B, pm)
    for st in SCORE_TYPES:
        out[f"{st}_top20_jaccard"] = topk_jaccard(sa[st], sb[st], 20)
        out[f"{st}_top100_jaccard"] = topk_jaccard(sa[st], sb[st], 100)
        rho = spearmanr(sa[st], sb[st]).correlation
        out[f"{st}_spearman"] = float(rho) if rho == rho else float("nan")
    return out


def compare_groups(A: Run, B: Run, labels_a, labels_b, nn, rng):
    sizes = np.bincount(labels_a)
    pool = np.arange(len(A.sup))
    rows = []
    for gid in np.where(sizes >= MIN_GROUP_SIZE)[0]:
        members = np.where(labels_a == gid)[0]
        real = _evaluate_group(A, B, members, nn, labels_b)
        null = [_evaluate_group(A, B, resample_group_same_strata(members, A.strata, pool, rng),
                                nn, labels_b) for _ in range(N_NULL_GROUPS)]
        row = {"group_id_A": int(gid), "size_A": int(len(members)), **real}
        for m in METRICS:
            nv = np.array([x[m] for x in null], dtype=float)
            row[f"null_{m}_mean"] = float(np.nanmean(nv))
            row[f"p_null_{m}"] = float((1 + np.nansum(nv >= real[m])) / (1 + np.sum(~np.isnan(nv))))
        rows.append(row)
    for m in METRICS:
        if rows:
            q = fdr_bh([r[f"p_null_{m}"] for r in rows])
            for r, qq in zip(rows, q):
                r[f"p_null_{m}_fdr_bh"] = float(qq)
    agg = {"n_groups_compared": len(rows)}
    for m in METRICS:
        if rows:
            agg[f"mean_real_{m}"] = float(np.nanmean([r[m] for r in rows]))
            agg[f"mean_null_{m}"] = float(np.mean([r[f"null_{m}_mean"] for r in rows]))
            agg[f"n_groups_fdr_lt_0.05_{m}"] = int(sum(r[f"p_null_{m}_fdr_bh"] < 0.05 for r in rows))
    return {"aggregate": agg, "groups": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", required=True,
                    help="nom=dossier[@arm] (arm = pca|random, defaut pca), 3 fois "
                         "(ex. seed42=results_...  rand45=results_...@random)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mails-tsv-path", default="local_data/emails/Mails.tsv")
    ap.add_argument("--augmented-jsonl-path", default="local_data/emails/augmented_mails.jsonl")
    ap.add_argument("--split-assignments", required=True)
    ap.add_argument("--max-augmented-per-mail", type=int, default=13)
    ap.add_argument("--d-extra", type=int, required=True)
    ap.add_argument("--out", default="e05_stability.json")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(0)

    fit_texts, _, dev_texts, _ = load_fit_dev_corpus_from_manifest(
        args.split_assignments, args.mails_tsv_path, args.augmented_jsonl_path,
        max_augmented_per_mail=args.max_augmented_per_mail)
    n_fit, n_dev = len(fit_texts), len(dev_texts)
    print(f"[e05] FIT={n_fit} DEV={n_dev}", flush=True)

    runs = {}
    for spec in args.run:
        name, rest = spec.split("=", 1)
        path, _, arm = rest.partition("@")
        arm = arm or "pca"
        assert arm in ("pca", "random"), arm
        runs[name] = Run(name, path, args.d_extra, n_fit, n_dev, arm=arm)
        print(f"[e05] {name} : {runs[name].status_counts()}", flush=True)
    names = list(runs)

    distinct = {}
    for a in names:
        for b in names:
            if a < b:
                Wa, Wb = runs[a].W, runs[b].W
                same_idx_cos = (Wa * Wb).sum(1) / (np.linalg.norm(Wa, axis=1) * np.linalg.norm(Wb, axis=1))
                distinct[f"{a}_vs_{b}"] = {
                    "mean_same_index_cosine": float(same_idx_cos.mean()),
                    "max_abs_weight_diff": float(np.abs(Wa - Wb).max()),
                }
                assert not np.allclose(Wa, Wb), f"{a} et {b} ont des decodeurs identiques -- checkpoint reutilise ?"
    print(f"[e05] checkpoints distincts : {distinct}", flush=True)

    matches, pair_summaries = {}, {}
    for a in names:
        for b in names:
            if a != b:
                matches[(a, b)] = match_pair(runs[a], runs[b], rng)
                pair_summaries[f"{a}->{b}"] = matches[(a, b)]["summary"]
                s = matches[(a, b)]["summary"]
                print(f"[e05] {a}->{b} : cos>=0.7 {s['frac_cos_ge_0.7']:.3f} (nul {s['null_frac_cos_ge_0.7']:.3f}) ; "
                      f"cos>=0.5 {s['frac_cos_ge_0.5']:.3f} (nul {s['null_frac_cos_ge_0.5']:.3f}) ; "
                      f"joint {s['frac_joint_cos0.7_and_corr0.5']:.3f}", flush=True)

    ref = names[0]
    others = [n for n in names if n != ref]
    joint_by_other = {o: matches[(ref, o)]["joint"] for o in others}
    found = {"reference": ref, "reference_arm": runs[ref].arm, "n_reference_supported": int(len(runs[ref].sup)),
             "note": "critere = cos>=0.7 ET correlation de profils>=0.5 dans TOUTES les runs du sous-ensemble ; "
                     "peu de graines, pas une preuve de reproductibilite generale"}
    for label, subset in (("all_other_runs", others),
                          ("all_other_pca_runs", [o for o in others if runs[o].arm == "pca"]),
                          ("all_random_runs", [o for o in others if runs[o].arm == "random"])):
        if subset:
            mask = np.logical_and.reduce([joint_by_other[o] for o in subset])
            found[label] = {"runs": subset, "n_found": int(mask.sum()), "frac_found": float(mask.mean())}
    found_in_both = found

    labels, group_info = {}, {}
    for n_ in names:
        labels[n_], group_info[n_] = build_groups(runs[n_], rng)
        gi = group_info[n_]
        print(f"[e05] groupes {n_} : {gi['n_groups(size>=%d)' % MIN_GROUP_SIZE]} groupes, "
              f"{gi['n_isolates(size=1)']} isolats, tau={gi['profile_corr_edge_threshold_tau_q99_permutation_witness']:.3f}, "
              f"max={gi['max_group_size']}", flush=True)

    group_comparisons = {}
    for a in names:
        for b in names:
            if a != b:
                group_comparisons[f"{a}->{b}"] = compare_groups(
                    runs[a], runs[b], labels[a], labels[b], matches[(a, b)]["nn"], rng)
                agg = group_comparisons[f"{a}->{b}"]["aggregate"]
                print(f"[e05] groupes {a}->{b} : {agg}", flush=True)

    def _ptype(key):
        a, b = key.split("->")
        return f"{runs[a].arm}->{runs[b].arm}"

    for key in pair_summaries:
        pair_summaries[key]["pair_type"] = _ptype(key)
    for key in group_comparisons:
        group_comparisons[key]["pair_type"] = _ptype(key)

    by_type = {}
    for key, sm in pair_summaries.items():
        t = by_type.setdefault(sm["pair_type"], {"pairs": [], "ind": [], "grp": []})
        t["pairs"].append(key)
        t["ind"].append(sm)
        t["grp"].append(group_comparisons[key]["aggregate"])
    summary_by_pair_type = {}
    for t, v in by_type.items():
        row = {"pairs": v["pairs"]}
        for m in ("frac_cos_ge_0.7", "frac_cos_ge_0.5", "null_frac_cos_ge_0.7", "frac_joint_cos0.7_and_corr0.5",
                  "mean_top20_jaccard", "frac_mutual_nn"):
            row[m] = float(np.mean([x[m] for x in v["ind"]]))
        row["median_best_cos_mean_over_pairs"] = float(np.mean([x["best_cos_quantiles"]["q50"] for x in v["ind"]]))
        row["median_profile_corr_mean_over_pairs"] = float(np.mean([x["profile_corr_quantiles"]["q50"] for x in v["ind"]]))
        for m in METRICS:
            row[f"group_{m}_real"] = float(np.mean([g[f"mean_real_{m}"] for g in v["grp"]]))
            row[f"group_{m}_null"] = float(np.mean([g[f"mean_null_{m}"] for g in v["grp"]]))
            row[f"group_{m}_n_fdr_lt_0.05_mean_over_pairs"] = float(np.mean([g[f"n_groups_fdr_lt_0.05_{m}"] for g in v["grp"]]))
        summary_by_pair_type[t] = row
        print(f"[e05] type {t}: joint={row['frac_joint_cos0.7_and_corr0.5']:.3f} purity={row['group_purity_real']:.2f}/"
              f"{row['group_purity_null']:.2f} overlap={row['group_overlap_real']:.2f}/{row['group_overlap_null']:.2f}", flush=True)

    Wc = runs[ref].W_sup - runs[ref].W_sup.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Wc, full_matrices=False)
    xy = Wc @ Vt[:2].T
    map_positions = [{"feature_uid": runs[ref].uid(runs[ref].sup[i]), "x": float(xy[i, 0]), "y": float(xy[i, 1]),
                      "group_id": int(labels[ref][i]), "freq_fit": float(runs[ref].freq_fit[runs[ref].sup[i]])}
                     for i in range(len(runs[ref].sup))]

    feature_matches = {}
    for (a, b), m in matches.items():
        if a == ref:
            feature_matches[f"{a}->{b}"] = [
                {"uid_A": runs[a].uid(runs[a].sup[i]), "uid_B": runs[b].uid(runs[b].sup[m["nn"][i]]),
                 "cos": float(m["cos"][i]), "profile_corr": None if np.isnan(m["corr"][i]) else float(m["corr"][i]),
                 "top20_jaccard": float(m["jac"][i]), "joint_pass": bool(m["joint"][i])}
                for i in range(len(m["nn"]))]

    out = {
        "n_fit": n_fit, "n_dev": n_dev, "runs": {n_: runs[n_].status_counts() for n_ in names},
        "thresholds": {"cos": COS_THRESHOLD, "profile_corr": PROFILE_CORR_THRESHOLD,
                       "min_active_fit": MIN_ACTIVE_FIT, "max_freq_fit": MAX_FREQ_FIT, "knn_k": KNN_K,
                       "min_group_size": MIN_GROUP_SIZE, "n_null_groups": N_NULL_GROUPS,
                       "n_null_dictionaries": N_NULL_DICTS},
        "checkpoints_distinct_diagnostic": distinct,
        "summary_by_pair_type": summary_by_pair_type,
        "individual_matching": pair_summaries,
        "found_across_runs_reference_side": found_in_both,
        "groups": group_info, "group_comparisons": group_comparisons,
        "feature_matches_reference_side": feature_matches, "map_positions_reference": map_positions,
        "init_caveat": "les runs d'arm decoder_init=pca partagent l'init PCA du residu sur le reservoir "
                       "partage (seul l'ordre des mini-lots varie) ; seules les paires random->random "
                       "melangent init ET ordre (independance a l'init)",
        "human_validation_pending": True,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    out_path = os.path.join(args.out_dir, args.out)
    with open(out_path + ".tmp", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    os.replace(out_path + ".tmp", out_path)
    print(f"[e05] Écrit {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
