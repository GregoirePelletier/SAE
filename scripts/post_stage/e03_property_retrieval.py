"""
scripts/post_stage/e03_property_retrieval.py -- Retrieval par propriete sur
CONFIRM (Plan_execution_SAE_15_jours_Claude_Code.md §8). Compare CORE, FULL,
DENSE (bge-m3), TFIDF, BM25 (LatentTermsIndex reutilise en mode lexical
generique) sur 6 proprietes x 2 formulations (lexicale + paraphrase) = 12
requetes -- correspond au budget "12 CONFIRM" du plan.

Simplification assumee, documentee explicitement (faute de relecture
humaine disponible dans cette passe) : les jugements de pertinence sont
produits par le juge Qwen (0/1/2), pas par un humain -- le plan demande une
annotation humaine en double, non faite ici. A recalibrer avec un
echantillon humain des que possible (§8.3 du plan).

p90 de normalisation (`property_based_retrieval`) calcule directement sur
CONFIRM (piste exploratoire d'indexation, §4.5 du plan), pas sur FIT --
CONFIRM sert ici de "collection a indexer" pour la recherche, un protocole
transductif standard en IR, distinct de la piste stricte utilisee pour les
sondes E01.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy import sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _REPO_ROOT)
# src/sae/saev5.py contient des imports bruts ("from sae_shared import ...",
# style module plat) qui supposent src/sae/ lui-meme sur sys.path -- vrai
# quand saev5.py est lance directement (python src/sae/saev5.py, sys.path[0]
# = son propre dossier), pas quand il est importe comme module depuis un
# script externe comme celui-ci. Ajoute explicitement plutot que de
# dupliquer property_based_retrieval ici.
sys.path.insert(0, os.path.join(_REPO_ROOT, "src", "sae"))

from src.post_stage.dataset_contract import load_confirm_corpus_from_manifest  # noqa: E402
from src.post_stage.representations import (  # noqa: E402
    build_tfidf_representation, embed_bge_m3_documents, TfidfConfig,
)
from src.sae.sae_shared import load_all_doc_acts  # noqa: E402
from src.sae.saev5 import property_based_retrieval  # noqa: E402
from src.sae.retrieval.latent_terms import LatentTermsIndex  # noqa: E402
from src.sae.judge import load_judge_model, _batched_generate  # noqa: E402
from src.analysis.stats import bootstrap_ci_by_group  # noqa: E402


QUERY_FAMILIES = [
    {
        "family": "relances_repetees",
        "definition": ("Le client indique avoir deja contacte le service (telephone, mail, agence) "
                       "au moins une fois auparavant pour le MEME probleme, sans qu'il soit resolu."),
        "lexical": "Le client a déjà contacté le service plusieurs fois sans solution.",
        "paraphrase": "Ce n'est pas la première fois que ce client signale ce problème, sans succès jusqu'ici.",
    },
    {
        "family": "menace_resiliation",
        "definition": "Le client indique explicitement qu'il mettra fin a son contrat si le probleme n'est pas resolu.",
        "lexical": "Le client menace de résilier si le problème persiste.",
        "paraphrase": "Le client prévient qu'il changera de fournisseur si rien ne change.",
    },
    {
        "family": "incident_collectif",
        "definition": "Le mail decrit un probleme touchant plusieurs personnes/logements a la fois, pas un cas isole.",
        "lexical": "Plusieurs personnes ou logements semblent concernés par un même problème.",
        "paraphrase": "Le problème décrit touche apparemment tout un quartier ou plusieurs foyers, pas un cas isolé.",
    },
    {
        "family": "explication_montant",
        "definition": "Le client demande une explication/un detail du calcul d'un montant facture, sans demander de remboursement.",
        "lexical": "Le client demande l'explication d'un montant plutôt qu'un remboursement.",
        "paraphrase": "Le client veut comprendre comment un montant a été calculé, sans réclamer d'argent en retour.",
    },
    {
        "family": "coupure_repetee",
        "definition": "Le client decrit des coupures d'electricite RECURRENTES, pas un incident unique/isole.",
        "lexical": "Le client décrit une coupure répétée plutôt qu'une panne isolée.",
        "paraphrase": "Les coupures de courant reviennent régulièrement chez ce client, ce n'est pas un incident ponctuel.",
    },
    {
        "family": "urgence_implicite",
        "definition": "Le ton du message exprime une urgence/attente d'action rapide, meme sans utiliser le mot urgent.",
        "lexical": "Le client exprime une urgence immédiate, même sans employer le mot urgent.",
        "paraphrase": "Le ton du message suggère qu'une action rapide est attendue, sans que le mot 'urgent' n'apparaisse.",
    },
]


def _judge_relevance(model, tokenizer, definition: str, doc_text: str, batch_size: int = 8) -> int:
    prompt = (
        "Voici la definition d'une propriete que peut avoir un email client, et un email a evaluer.\n\n"
        f"Propriete : {definition}\n\n"
        f"Email :\n{doc_text[:1500]}\n\n"
        "L'email presente-t-il cette propriete ? Reponds uniquement par un chiffre : "
        "0 (non), 1 (partiellement/ambigu), 2 (clairement oui)."
    )
    resp = _batched_generate(model, tokenizer, [[{"role": "user", "content": prompt}]],
                              max_new_tokens=4, batch_size=batch_size)[0]
    import re
    m = re.search(r"[012]", resp)
    return int(m.group()) if m else 0


def _rank_by_similarity(query_vec, doc_matrix, top_k=10):
    sims = cosine_similarity(query_vec, doc_matrix).ravel()
    order = np.argsort(sims)[::-1][:top_k]
    return [(int(i), float(sims[i])) for i in order if sims[i] > 0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--mails-tsv-path", default="local_data/emails/Mails.tsv")
    ap.add_argument("--augmented-jsonl-path", default="local_data/emails/augmented_mails.jsonl")
    ap.add_argument("--split-assignments", required=True)
    ap.add_argument("--registry-path", required=True)
    ap.add_argument("--d-extra", type=int, required=True)
    ap.add_argument("--max-augmented-per-mail", type=int, default=13)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--dense-model-path", default="./models/bge-m3")
    ap.add_argument("--out", default="e03_property_retrieval.json")
    args = ap.parse_args()

    t0 = time.time()
    print("[e03_retrieval] Chargement CONFIRM (diff_texts du run d'encodage dedie)...", flush=True)
    confirm_texts, confirm_labels, confirm_groups = load_confirm_corpus_from_manifest(
        args.split_assignments, args.mails_tsv_path, args.augmented_jsonl_path,
        max_augmented_per_mail=args.max_augmented_per_mail, return_groups=True,
    )
    n_confirm = len(confirm_texts)
    print(f"[e03_retrieval] CONFIRM={n_confirm}", flush=True)
    text_to_idx = {t: i for i, t in enumerate(confirm_texts)}

    with open(args.registry_path) as f:
        registry = json.load(f)["features"]
    core_labels = {v["global_index"]: v["label"] for v in registry.values()
                   if v["branch"] == "core" and v["status"] == "interpretable"}
    full_labels = {v["global_index"]: v["label"] for v in registry.values()
                   if v["status"] == "interpretable"}
    print(f"[e03_retrieval] {len(core_labels)} labels CORE, {len(full_labels)} labels FULL (dont EXTRA) "
          "utilisables pour le matching de requete.", flush=True)

    # CORE/FULL : activations CONFIRM = les DERNIERES lignes de all_doc_acts
    # (all_texts = train++test++diff, diff=CONFIRM ici, aucun filler dans ce run).
    acts_path = os.path.join(args.save_dir, "cache", f"p1_all_doc_acts_ext_d{args.d_extra}.pt")
    all_doc_acts = load_all_doc_acts(acts_path)
    confirm_acts = all_doc_acts[-n_confirm:]
    assert confirm_acts.shape[0] == n_confirm
    del all_doc_acts

    # DENSE
    print("[e03_retrieval] Encodage DENSE (bge-m3) de CONFIRM...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    confirm_dense, dense_config = embed_bge_m3_documents(
        confirm_texts, model_path=args.dense_model_path, max_length=2048, device=device,
    )
    confirm_dense_np = confirm_dense.numpy()

    # TFIDF -- ajuste sur CONFIRM lui-meme (piste exploratoire d'indexation,
    # meme raisonnement que le p90 ci-dessus) : coherent avec le fait que
    # cette experience n'a pas de FIT-texts charges ici (script dedie a
    # l'evaluation retrieval, pas de reentrainement).
    print("[e03_retrieval] TFIDF (ajuste sur CONFIRM, piste exploratoire d'indexation)...", flush=True)
    tfidf_vec, tfidf_by_split, tfidf_config = build_tfidf_representation(confirm_texts, {})
    X_tfidf_confirm = tfidf_by_split["fit"]  # "fit" = le corpus passe en 1er arg, ici CONFIRM lui-meme

    # BM25 lexical (LatentTermsIndex reutilise en mode generique -- W_docs =
    # comptages de mots, pas des activations SAE, cf. docstring module).
    print("[e03_retrieval] BM25 lexical (LatentTermsIndex generique)...", flush=True)
    count_vec = CountVectorizer(max_features=20000)
    W_confirm = count_vec.fit_transform(confirm_texts)
    bm25_index = LatentTermsIndex(W_confirm)

    print("[e03_retrieval] Chargement du juge Qwen (relevance judging)...", flush=True)
    judge_model, judge_tokenizer = load_judge_model()

    all_query_results = []
    for fam in QUERY_FAMILIES:
        for formulation in ("lexical", "paraphrase"):
            query_text = fam[formulation]
            query_id = f"{fam['family']}__{formulation}"
            print(f"[e03_retrieval] Requete '{query_id}': {query_text!r}", flush=True)

            rankings = {}
            for method, labels in (("core", core_labels), ("full", full_labels)):
                results = property_based_retrieval(
                    query_text, confirm_acts, confirm_texts, labels, top_n_results=args.top_k,
                )
                rankings[method] = [(text_to_idx[t], s) for t, s in results if t in text_to_idx]

            query_dense, _ = embed_bge_m3_documents(
                [query_text], model_path=args.dense_model_path, max_length=64, device=device,
            )
            rankings["dense"] = _rank_by_similarity(
                query_dense.numpy(), confirm_dense_np, top_k=args.top_k)

            q_tfidf = tfidf_vec.transform([query_text])
            rankings["tfidf"] = _rank_by_similarity(q_tfidf, X_tfidf_confirm, top_k=args.top_k)

            w_q = np.asarray(count_vec.transform([query_text]).todense()).ravel()
            rankings["bm25"] = bm25_index.search(w_q, top_k=args.top_k)

            union_idx = sorted(set(i for r in rankings.values() for i, _ in r))
            print(f"[e03_retrieval]   union a juger : {len(union_idx)} documents", flush=True)
            relevance = {
                i: _judge_relevance(judge_model, judge_tokenizer, fam["definition"], confirm_texts[i])
                for i in union_idx
            }

            metrics = {}
            for method, ranked in rankings.items():
                top = [i for i, _ in ranked[:args.top_k]]
                n_judged = len(top)
                strict_hits = sum(1 for i in top if relevance.get(i, 0) == 2)
                tolerant_hits = sum(1 for i in top if relevance.get(i, 0) >= 1)
                metrics[method] = {
                    "p_at_10_strict": strict_hits / args.top_k,
                    "p_at_10_tolerant": tolerant_hits / args.top_k,
                    "n_results": n_judged,
                }
                print(f"[e03_retrieval]   {method}: P@10 strict={metrics[method]['p_at_10_strict']:.2f} "
                      f"tolerant={metrics[method]['p_at_10_tolerant']:.2f}", flush=True)

            all_query_results.append({
                "family": fam["family"], "formulation": formulation, "query_id": query_id,
                "query_text": query_text, "n_union_judged": len(union_idx),
                "metrics": metrics,
            })

    del judge_model
    torch.cuda.empty_cache()

    # Agregation par famille (moyenne lexical+paraphrase) et comparaison FULL vs autres.
    families = sorted(set(r["family"] for r in all_query_results))
    methods = ("core", "full", "dense", "tfidf", "bm25")
    per_family_p10 = {m: [] for m in methods}
    for fam_name in families:
        fam_rows = [r for r in all_query_results if r["family"] == fam_name]
        for m in methods:
            vals = [r["metrics"][m]["p_at_10_strict"] for r in fam_rows]
            per_family_p10[m].append(float(np.mean(vals)))

    comparisons = {}
    for other in ("core", "dense", "tfidf", "bm25"):
        diffs = np.array(per_family_p10["full"]) - np.array(per_family_p10[other])
        ci = bootstrap_ci_by_group(diffs, np.arange(len(diffs)), statistic=np.mean, n_boot=2000, seed=42)
        comparisons[f"full_minus_{other}"] = {
            "mean_diff": float(diffs.mean()), "per_family_diff": diffs.tolist(),
            "bootstrap_ci_low": ci.ci_low, "bootstrap_ci_high": ci.ci_high,
            "n_families": len(families),
            "note": "n=6 familles seulement -- variabilite a lire comme telle, pas un test definitif (plan §8.5).",
        }

    out = {
        "n_confirm": n_confirm, "top_k": args.top_k,
        "methods": methods,
        "per_query_results": all_query_results,
        "per_family_mean_p_at_10_strict": {m: dict(zip(families, per_family_p10[m])) for m in methods},
        "comparisons_full_vs_others": comparisons,
        "judge": "Qwen (model-judged, human calibration pending)",
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    out_path = os.path.join(args.save_dir, args.out)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, out_path)
    print(f"[e03_retrieval] Écrit {out_path}", flush=True)
    print(json.dumps(comparisons, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
