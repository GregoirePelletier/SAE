"""
scripts/latent_retrieval_precision_eval.py — Évaluation quantitative de
src/sae/retrieval/latent_terms.py (Latent Terms, Clavié et al. 2026,
arXiv:2605.29384, réimplémentation token-level : BM25 sur le vocabulaire
latent d'un SAE entraîné par pure reconstruction sur des activations token
F2LLM d'un corpus GÉNÉRIQUE hors-domaine, §3.1).

Remplace la version précédente (SAE phrase-level entraîné directement sur
Mails.tsv) : RESULTS_TESTS.md §26/§68/§69, supersédés par §80.

Protocole :
  1. Corpus = mails originaux de Mails.tsv (3480 mails), 4 intentions déjà
     validées comme suffisamment équilibrées (>=30 positifs) dans les tests
     précédents : réclamation, remboursement, information, urgence
     (`src.data.dataset.INTENT_KEYWORDS_FR`, patterns V2 en production, N5
     docs/archive/audits/AUDIT_SAE_2026-08.md §8 -- motif "remboursement" resserré).
  2. Pour chaque intention, une requête en langage naturel PARAPHRASANT (pas
     copiant mot pour mot) le motif regex de l'intention -- teste la
     généralisation sémantique, pas juste le rappel de mots-clés exacts.
  3. Index Latent Terms construit sur les 3480 mails ENTIERS (pas de
     découpage en phrases -- écart corrigé par rapport à la version
     précédente, cf. docstring de latent_terms.py).
  4. Baseline de comparaison : TF-IDF + cosinus sur le texte brut, mêmes
     requêtes, même corpus.
  5. Combinaison de rangs (RRF, App. G "Combining results", `src.analysis.
     metrics.reciprocal_rank_fusion`) entre Latent Terms et TF-IDF.
  6. Reranking LLM du top 50 du classement RRF-combiné ("second stage
     retrieval", App. G -- `src.analysis.retrieval_rerank.llm_rerank`,
     protocole de reranking non publié dans le papier, pointwise 0-10 en
     écart documenté, cf. docstring du module).
  7. RBO (Rank-Biased Overlap, p=0.98, App. G Figure 27) entre les classements
     Latent Terms et TF-IDF -- quantifie leur désaccord, indépendamment de
     leur précision individuelle.
  8. Métriques : Precision@10/@20 par requête + MAP/MP@10 agrégées sur les 4
     requêtes (App. G), pour chacune des 4 méthodes (TF-IDF, Latent Terms,
     RRF, RRF+reranking LLM).

Usage : PYTHONPATH=. .venv/bin/python scripts/latent_retrieval_precision_eval.py
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import LOCAL_MAILS_PATH, SAVE_DIR, CACHE_DIR, D_SAE, K_SPARSE, EMB_MODEL
from src.data.dataset import load_mails_tsv
from src.sae.retrieval.latent_terms import (
    load_f2llm, build_token_training_pool, load_or_train_latent_terms_sae,
    latent_doc_weights, LatentTermsIndex, TRAIN_TOKENS,
)
from src.sae.judge import load_judge_model
from src.analysis.retrieval_rerank import llm_rerank
from src.analysis.metrics import (
    reciprocal_rank_fusion, rank_biased_overlap,
    average_precision, precision_at_k, mean_average_precision, mean_precision_at_k,
)

os.makedirs(CACHE_DIR, exist_ok=True)  # saev5.py le fait au chargement du module, ce script autonome pas -- crash
                                        # confirmé (job 44996) sur un SAVE_DIR fraîchement créé (dédoublonnage GPU)

TOP_K = (10, 20)
RANK_DEPTH = 50   # profondeur de classement pour RRF/rerank/RBO -- "top 50" (App. G)
RERANK_TOP_K = 50

# Requêtes en paraphrase (pas les mots exacts du regex INTENT_KEYWORDS_FR) --
# teste si le système retrouve le CONCEPT, pas juste les mots déclencheurs.
QUERIES = {
    "reclamation": "je ne suis pas du tout satisfait de ce qui m'est facturé, c'est inacceptable",
    "remboursement": "je souhaite être remboursé du montant que j'ai payé en trop",
    "information": "pouvez-vous m'expliquer comment faire pour changer d'offre",
    "urgence": "il faut intervenir tout de suite, je n'ai plus d'électricité",
}

METHODS = ("tfidf", "latent_terms", "rrf", "rrf_reranked")


def main():
    print("[retrieval-eval] Chargement de Mails.tsv (mails originaux)...")
    df = load_mails_tsv(LOCAL_MAILS_PATH)
    texts = df["text"].tolist()
    print(f"[retrieval-eval] {len(texts)} mails originaux.")

    intents = [c for c in QUERIES if f"intent_{c}" in df.columns]
    for intent in intents:
        n_pos = int(df[f"intent_{intent}"].sum())
        print(f"  intent_{intent}: {n_pos}/{len(df)} positifs ({100*n_pos/len(df):.1f}%)")

    print("[retrieval-eval] Chargement F2LLM + pool d'entraînement générique (hors domaine)...")
    tokenizer, model = load_f2llm()
    d_in = model.config.hidden_size
    model_tag = os.path.basename(EMB_MODEL.rstrip("/"))

    token_pool = build_token_training_pool(
        TRAIN_TOKENS, tokenizer, model,
        cache_path=os.path.join(CACHE_DIR, f"lt_generic_token_pool_n{TRAIN_TOKENS}_{model_tag}"))
    print("[retrieval-eval] Entraînement du SAE token-level (reconstruction pure, hors domaine)...")
    sae, _ = load_or_train_latent_terms_sae(
        d_in=d_in, d_sae=D_SAE, k=K_SPARSE, token_pool=token_pool,
        save_path=os.path.join(SAVE_DIR, f"lt_sae_token_d{D_SAE}_k{K_SPARSE}_tok{TRAIN_TOKENS}_{model_tag}.pt"))

    print("[retrieval-eval] Indexation des 3480 mails (token-level, sum-pooling par document)...")
    W_docs = latent_doc_weights(sae, texts, tokenizer, model)
    index = LatentTermsIndex(W_docs)

    print("[retrieval-eval] Baseline TF-IDF...")
    tfidf = TfidfVectorizer(max_features=20000)
    X_tfidf = tfidf.fit_transform(texts)

    print("[retrieval-eval] Chargement du juge (reranking LLM, second stage retrieval)...")
    judge_model, judge_tokenizer = load_judge_model()

    per_query = {}
    relevance_by_method = {m: [] for m in METHODS}
    rbo_scores = []

    for intent, query in QUERIES.items():
        if f"intent_{intent}" not in df.columns:
            continue
        relevant = df[f"intent_{intent}"].to_numpy().astype(bool)
        base_rate = float(relevant.mean())

        # Latent Terms
        W_q = latent_doc_weights(sae, [query], tokenizer, model)
        w_q = np.asarray(W_q.todense()).ravel()
        lt_ranked = [i for i, _ in index.search(w_q, top_k=RANK_DEPTH)]

        # TF-IDF
        q_tfidf = tfidf.transform([query])
        sims = cosine_similarity(q_tfidf, X_tfidf).ravel()
        tfidf_ranked = list(np.argsort(sims)[::-1][:RANK_DEPTH])

        # RRF (App. G, "Combining results") -- k=60, cf. docstring reciprocal_rank_fusion.
        rrf_ranked = [doc_id for doc_id, _ in reciprocal_rank_fusion([lt_ranked, tfidf_ranked])][:RANK_DEPTH]

        # Reranking LLM du top 50 du classement RRF ("second stage retrieval", App. G).
        doc_texts = {i: texts[i] for i in rrf_ranked[:RERANK_TOP_K]}
        rrf_reranked = llm_rerank(
            judge_model, judge_tokenizer, query, rrf_ranked, doc_texts, top_k=RERANK_TOP_K,
        )

        # RBO (App. G Figure 27, p=0.98) : désaccord entre Latent Terms et TF-IDF.
        rbo = rank_biased_overlap(lt_ranked, tfidf_ranked, p=0.98)
        rbo_scores.append(rbo)

        rankings = {
            "tfidf": tfidf_ranked, "latent_terms": lt_ranked,
            "rrf": rrf_ranked, "rrf_reranked": rrf_reranked,
        }
        entry = {"query": query, "base_rate": base_rate, "rbo_tfidf_vs_latent_terms": rbo}
        for method, ranked in rankings.items():
            rel_list = [bool(relevant[i]) for i in ranked]
            relevance_by_method[method].append(rel_list)
            entry[f"average_precision_{method}"] = average_precision(rel_list)
            for k in TOP_K:
                entry[f"precision_at_{k}_{method}"] = precision_at_k(rel_list, k)
        per_query[intent] = entry

        print(f"[retrieval-eval] {intent} (base_rate={base_rate:.3f}, RBO={rbo:.3f}) : " +
              " | ".join(f"{m} P@10={entry[f'precision_at_10_{m}']:.2f}" for m in METHODS))

    aggregate = {}
    for method in METHODS:
        rel_lists = relevance_by_method[method]
        aggregate[method] = {
            "MAP": mean_average_precision(rel_lists),
            **{f"MP@{k}": mean_precision_at_k(rel_lists, k) for k in TOP_K},
        }
    aggregate["mean_rbo_tfidf_vs_latent_terms"] = float(np.mean(rbo_scores)) if rbo_scores else float("nan")

    print("\n[retrieval-eval] Agrégat (MAP/MP@k par méthode, RBO moyen) :")
    for method in METHODS:
        print(f"  {method}: {aggregate[method]}")
    print(f"  mean_rbo_tfidf_vs_latent_terms: {aggregate['mean_rbo_tfidf_vs_latent_terms']:.3f}")

    out_path = os.path.join(CACHE_DIR, "latent_retrieval_precision_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"per_query": per_query, "aggregate": aggregate}, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Écrit : {out_path}")


if __name__ == "__main__":
    main()
