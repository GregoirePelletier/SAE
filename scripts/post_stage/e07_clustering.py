"""
scripts/post_stage/e07_clustering.py -- Clustering cible reellement compare
aux alternatives (Plan_execution_SAE_15_jours_Claude_Code.md §12).

3 axes fixes sur CONFIRM (type de probleme, action attendue, registre/
urgence), matching de features par similarite label<->requete
(`select_latents_by_similarity`, reutilise TEL QUEL -- son mapping f_idx/
label est deja correct par construction du zip, verifie par
`tests/post_stage/test_e07_clustering.py::test_select_latents_mapping_survives_unsorted_dict`
avec un dict volontairement non trie, cf. le doute souleve par le plan
§12.1 -- pas reproduit ici). CORE et FULL comparent au MEME budget de
labels (TOP_K_FEATURES chacun). Ecarts corriges par rapport a
`targeted_clustering_by_axis` (src/sae/saev5.py), qui reste inchangee car
utilisee ailleurs dans le pipeline principal :
  - pas de repli silencieux "<5 features -> tout le dictionnaire" : statut
    explicite `axis_not_supported` pour cette branche/cet axe.
  - documents au vecteur restreint nul geres comme "hors axe / sans signal"
    (statut -1, exclu du clustering, reporte separement), pas comme un
    groupe thematique via Jaccard(0,0)=1.
  - embeddings DENSE sur des EMAILS ENTIERS via `embed_bge_m3_documents`
    (max_length=2048), jamais `_embed_bge_m3` (troncature a 64 tokens,
    concue pour des labels courts).

k=4 clusters fixe pour toutes les methodes/tous les axes (budget de lecture
initial du plan), jamais choisi apres lecture de la meilleure separation.

Evaluation sans boucle auto-validante (plan §12.3) : quelques exemples par
cluster reserves a la description LLM (`generate_cluster_labels`), la
reassignation LLM (`compute_cluster_accuracy`, secondaire/diagnostique) porte
sur les AUTRES documents du meme cluster. Conductance dans l'espace DENSE
bge-m3 (documents entiers), meme espace pour toutes les methodes -- calculee
via `conductance_zscore` deja existante (reutilisee telle quelle).

Simplifications assumees, documentees explicitement :
- TFIDF ajuste directement sur l'echantillon CONFIRM (piste exploratoire
  d'indexation, meme raisonnement que E03), pas de baseline FIT-only.
- Audit humain aveugle de paires intra/inter-cluster (20-30 par axe, plan
  §12.3) NON FAIT -- necessite Gregoire, reste en attente.
- Un seul email representatif par parent CONFIRM echantillonne (pas toutes
  les variantes augmentees), meme convention que E06.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import normalize

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src", "sae"))

from src.post_stage.dataset_contract import load_confirm_corpus_from_manifest  # noqa: E402
from src.post_stage.representations import build_tfidf_representation, embed_bge_m3_documents  # noqa: E402
from src.sae.sae_shared import load_all_doc_acts  # noqa: E402
from src.sae.saev5 import select_latents_by_similarity  # noqa: E402
from src.sae.judge import load_judge_model  # noqa: E402
from src.analysis.clustering_llm import generate_cluster_labels, compute_cluster_accuracy, conductance_zscore  # noqa: E402

N_CLUSTERS = 4
MIN_MATCHED_FEATURES = 5
N_RESERVED_FOR_NAMING = 5
# Doit rester < taille du plus petit catalogue interpretable (CORE=77 sur ce
# run, cf. e02_feature_registry.json) -- top_k=150 (valeur initiale, calquee
# sur un catalogue de production bien plus grand) ne restreignait RIEN pour
# CORE : select_latents_by_similarity retournait le catalogue ENTIER quel
# que soit l'axe (77<150), rendant les 3 clusterings CORE numeriquement
# identiques d'un axe a l'autre (memes cluster_sizes exacts, verifie sur un
# premier run, job 49084) -- l'axe ne discriminait jamais CORE. 40 force une
# vraie restriction pour CORE (77) ET FULL (197 features interpretables).
TOP_K_FEATURES = 40

AXES = [
    {"axis_id": "type_probleme",
     "query": "type de probleme rencontre par le client : facturation, coupure, resiliation, probleme technique"},
    {"axis_id": "action_attendue",
     "query": "action ou resolution demandee par le client : remboursement, explication, intervention, reponse"},
    {"axis_id": "registre_urgence",
     "query": "registre emotionnel et urgence du message : calme, urgent, en colere, neutre"},
]


def _cluster_from_binarized(binarized: np.ndarray) -> np.ndarray:
    """SpectralClustering (affinite Jaccard precalculee) sur les lignes NON
    nulles de `binarized` ; retourne un vecteur de labels de meme longueur
    que `binarized`, -1 pour les lignes nulles ("hors axe / sans signal")."""
    n = binarized.shape[0]
    labels = np.full(n, -1, dtype=int)
    nonzero_rows = np.where(binarized.sum(axis=1) > 0)[0]
    if len(nonzero_rows) < N_CLUSTERS:
        return labels
    sub = binarized[nonzero_rows]
    jaccard_sim = 1.0 - squareform(pdist(sub, metric="jaccard"))
    jaccard_sim = np.nan_to_num(jaccard_sim, nan=1.0)  # deux lignes nulles -> pdist jaccard=0 -> sim=1 (cas exclu ici)
    spectral = SpectralClustering(n_clusters=N_CLUSTERS, affinity="precomputed",
                                   assign_labels="kmeans", random_state=42)
    sub_labels = spectral.fit_predict(jaccard_sim)
    labels[nonzero_rows] = sub_labels
    return labels


def _cluster_dense(vectors: np.ndarray) -> np.ndarray:
    normalized = normalize(vectors, norm="l2", axis=1)
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    return km.fit_predict(normalized)


def _evaluate_method(judge_model, judge_tokenizer, texts: list, cluster_labels: np.ndarray,
                      feature_or_dim_labels: dict, matched_indices, dense_embeddings: np.ndarray) -> dict:
    unique_clusters = sorted(set(cluster_labels.tolist()) - {-1})
    n_excluded = int((cluster_labels == -1).sum())
    if not unique_clusters:
        return {"status": "no_cluster_formed", "n_excluded_no_signal": n_excluded}

    naming_examples, eval_examples, eval_labels = {}, [], []
    for cid in unique_clusters:
        idx = np.where(cluster_labels == cid)[0]
        rng = np.random.default_rng(42 + cid)
        shuffled = rng.permutation(idx)
        reserved = shuffled[:N_RESERVED_FOR_NAMING]
        rest = shuffled[N_RESERVED_FOR_NAMING:]
        naming_examples[cid] = {
            "features": [feature_or_dim_labels.get(i, f"dim{i}") for i in
                         (matched_indices[:5] if matched_indices else [])],
            "examples": [texts[i][:300] for i in reserved],
        }
        for i in rest:
            eval_examples.append(texts[i])
            eval_labels.append(cid)

    clusters_ordered = [naming_examples[cid] for cid in unique_clusters]
    llm_labels = generate_cluster_labels(judge_model, judge_tokenizer, clusters_ordered, n_relabel=5)
    cluster_descriptions = {cid: llm_labels[k] for k, cid in enumerate(unique_clusters)}

    reassignment_acc = {}
    if eval_examples:
        reassignment_acc = compute_cluster_accuracy(
            judge_model, judge_tokenizer, eval_examples, eval_labels, cluster_descriptions, batch_size=16,
        )

    cond_z = conductance_zscore(dense_embeddings, cluster_labels, k_neighbors=10, n_random=100, seed=42)

    return {
        "status": "ok",
        "n_clusters_formed": len(unique_clusters),
        "n_excluded_no_signal": n_excluded,
        "coverage": 1.0 - n_excluded / len(cluster_labels),
        "cluster_sizes": {int(cid): int((cluster_labels == cid).sum()) for cid in unique_clusters},
        "cluster_labels_llm": {int(cid): cluster_descriptions[cid] for cid in unique_clusters},
        "reassignment_accuracy_secondary": {int(k): v for k, v in reassignment_acc.items()},
        "conductance_zscore": {int(k): v for k, v in cond_z.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--mails-tsv-path", default="local_data/emails/Mails.tsv")
    ap.add_argument("--augmented-jsonl-path", default="local_data/emails/augmented_mails.jsonl")
    ap.add_argument("--split-assignments", required=True)
    ap.add_argument("--registry-path", required=True)
    ap.add_argument("--d-extra", type=int, required=True)
    ap.add_argument("--max-augmented-per-mail", type=int, default=13)
    ap.add_argument("--n-confirm-parents", type=int, default=400)
    ap.add_argument("--dense-model-path", default="./models/bge-m3")
    ap.add_argument("--judge-device", default="cuda")
    ap.add_argument("--out", default="e07_clustering.json")
    args = ap.parse_args()
    t0 = time.time()

    print("[e07_clustering] Chargement CONFIRM...", flush=True)
    confirm_texts, confirm_labels, confirm_groups = load_confirm_corpus_from_manifest(
        args.split_assignments, args.mails_tsv_path, args.augmented_jsonl_path,
        max_augmented_per_mail=args.max_augmented_per_mail, return_groups=True,
    )
    n_confirm_total = len(confirm_texts)
    confirm_groups_arr = np.asarray(confirm_groups)

    rng = np.random.default_rng(42)
    unique_parents = np.array(sorted(set(confirm_groups_arr.tolist())))
    n_sample = min(args.n_confirm_parents, len(unique_parents))
    sampled_parents = set(rng.choice(unique_parents, size=n_sample, replace=False).tolist())
    rep_doc_idx = {}
    for i, g in enumerate(confirm_groups_arr):
        if g in sampled_parents and g not in rep_doc_idx:
            rep_doc_idx[g] = i
    sampled_local_idx = np.array(sorted(rep_doc_idx.values()))
    sampled_texts = [confirm_texts[i] for i in sampled_local_idx]
    print(f"[e07_clustering] {len(sampled_texts)} parents CONFIRM echantillonnes "
          f"(1 email representatif chacun, graine 42).", flush=True)

    with open(args.registry_path) as f:
        registry = json.load(f)["features"]
    core_labels = {v["global_index"]: v["label"] for v in registry.values()
                   if v["branch"] == "core" and v["status"] == "interpretable"}
    full_labels = {v["global_index"]: v["label"] for v in registry.values() if v["status"] == "interpretable"}

    acts_path = os.path.join(args.save_dir, "cache", f"p1_all_doc_acts_ext_d{args.d_extra}.pt")
    all_doc_acts = load_all_doc_acts(acts_path)
    confirm_acts = all_doc_acts[-n_confirm_total:]
    sampled_acts = confirm_acts[sampled_local_idx].numpy()
    del all_doc_acts, confirm_acts

    print("[e07_clustering] Encodage DENSE (bge-m3, documents entiers, max_length=2048)...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dense_embs, dense_config = embed_bge_m3_documents(
        sampled_texts, model_path=args.dense_model_path, max_length=2048, device=device,
    )
    dense_embs_np = dense_embs.numpy()
    print(f"[e07_clustering] DENSE : {dense_config}", flush=True)

    print("[e07_clustering] TFIDF (ajuste sur l'echantillon CONFIRM, piste exploratoire)...", flush=True)
    tfidf_vec, tfidf_by_split, _ = build_tfidf_representation(sampled_texts, {})
    X_tfidf = tfidf_by_split["fit"].toarray()

    print("[e07_clustering] Chargement du juge Qwen...", flush=True)
    judge_model, judge_tokenizer = load_judge_model(device=args.judge_device)

    results = {}
    for axis in AXES:
        axis_id, query = axis["axis_id"], axis["query"]
        print(f"\n[e07_clustering] === Axe '{axis_id}' : {query!r} ===", flush=True)
        axis_result = {"query": query}

        for branch, labels_dict in (("core", core_labels), ("full", full_labels)):
            matched = select_latents_by_similarity(query, labels_dict, top_k=TOP_K_FEATURES)
            if len(matched) < MIN_MATCHED_FEATURES:
                print(f"[e07_clustering]   {branch} : {len(matched)} features matchees (<{MIN_MATCHED_FEATURES}) "
                      "-- axis_not_supported, PAS de repli sur tout le dictionnaire.", flush=True)
                axis_result[branch] = {"status": "axis_not_supported", "n_matched_features": len(matched)}
                continue
            print(f"[e07_clustering]   {branch} : {len(matched)} features matchees.", flush=True)
            binarized = (sampled_acts[:, matched] > 1e-6).astype(np.float64)
            cluster_labels = _cluster_from_binarized(binarized)
            eval_out = _evaluate_method(judge_model, judge_tokenizer, sampled_texts, cluster_labels,
                                         labels_dict, matched, dense_embs_np)
            eval_out["n_matched_features"] = len(matched)
            axis_result[branch] = eval_out

        print("[e07_clustering]   dense (KMeans, cosine via L2-normalisation)...", flush=True)
        dense_cluster_labels = _cluster_dense(dense_embs_np)
        axis_result["dense"] = _evaluate_method(judge_model, judge_tokenizer, sampled_texts,
                                                  dense_cluster_labels, {}, None, dense_embs_np)

        print("[e07_clustering]   tfidf (KMeans, cosine via L2-normalisation)...", flush=True)
        tfidf_cluster_labels = _cluster_dense(X_tfidf)
        axis_result["tfidf"] = _evaluate_method(judge_model, judge_tokenizer, sampled_texts,
                                                  tfidf_cluster_labels, {}, None, dense_embs_np)

        results[axis_id] = axis_result

    del judge_model
    torch.cuda.empty_cache()

    out = {
        "n_confirm_parents_sampled": len(sampled_texts),
        "n_clusters": N_CLUSTERS,
        "axes": results,
        "human_blind_pairs_audit_pending": True,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    out_path = os.path.join(args.save_dir, args.out)
    with open(out_path + ".tmp", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    os.replace(out_path + ".tmp", out_path)
    print(f"\n[e07_clustering] Écrit {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
