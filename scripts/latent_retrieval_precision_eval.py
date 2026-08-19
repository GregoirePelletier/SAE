"""
scripts/latent_retrieval_precision_eval.py — Évaluation quantitative de
src/sae/retrieval/latent_terms.py (Latent Terms, Clavié et al. 2026,
arXiv:2605.29384, réimplémentation token-level : BM25 sur le vocabulaire
latent d'un SAE entraîné par pure reconstruction sur des activations token
F2LLM d'un corpus GÉNÉRIQUE hors-domaine, §3.1).

Remplace la version précédente (SAE phrase-level entraîné directement sur
Mails.tsv) : RESULTS_TESTS.md §26/§68/§69, supersédés par §<N-À-COMPLÉTER>.

Protocole (inchangé par rapport aux runs précédents, seule l'implémentation
Latent Terms change) :
  1. Corpus = mails originaux de Mails.tsv (3480 mails), 4 intentions déjà
     validées comme suffisamment équilibrées (>=30 positifs) dans les tests
     précédents : réclamation, remboursement, information, urgence
     (`src.data.dataset.INTENT_KEYWORDS_FR`, patterns V2 en production).
  2. Pour chaque intention, une requête en langage naturel PARAPHRASANT (pas
     copiant mot pour mot) le motif regex de l'intention -- teste la
     généralisation sémantique, pas juste le rappel de mots-clés exacts.
  3. Index Latent Terms construit sur les 3480 mails ENTIERS (pas de
     découpage en phrases -- écart corrigé par rapport à la version
     précédente, cf. docstring de latent_terms.py).
  4. Baseline de comparaison : TF-IDF + cosinus sur le texte brut, mêmes
     requêtes, même corpus.
  5. Métrique : Precision@10 et Precision@20 (fraction de documents
     pertinents, au sens du label faible d'intention, dans le top-k),
     comparée au taux de base de l'intention dans le corpus.

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

TOP_K = (10, 20)

# Requêtes en paraphrase (pas les mots exacts du regex INTENT_KEYWORDS_FR) --
# teste si le système retrouve le CONCEPT, pas juste les mots déclencheurs.
QUERIES = {
    "reclamation": "je ne suis pas du tout satisfait de ce qui m'est facturé, c'est inacceptable",
    "remboursement": "je souhaite être remboursé du montant que j'ai payé en trop",
    "information": "pouvez-vous m'expliquer comment faire pour changer d'offre",
    "urgence": "il faut intervenir tout de suite, je n'ai plus d'électricité",
}


def precision_at_k(ranked_idx: list[int], relevant_mask: np.ndarray, k: int) -> float:
    top = ranked_idx[:k]
    if not top:
        return 0.0
    return float(relevant_mask[top].sum()) / len(top)


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

    results = {}
    for intent, query in QUERIES.items():
        if f"intent_{intent}" not in df.columns:
            continue
        relevant = df[f"intent_{intent}"].to_numpy().astype(bool)
        base_rate = float(relevant.mean())

        # Latent Terms
        W_q = latent_doc_weights(sae, [query], tokenizer, model)
        w_q = np.asarray(W_q.todense()).ravel()
        lt_ranked = [i for i, _ in index.search(w_q, top_k=max(TOP_K))]

        # TF-IDF
        q_tfidf = tfidf.transform([query])
        sims = cosine_similarity(q_tfidf, X_tfidf).ravel()
        tfidf_ranked = list(np.argsort(sims)[::-1][:max(TOP_K)])

        entry = {"query": query, "base_rate": base_rate}
        for k in TOP_K:
            entry[f"precision_at_{k}_latent_terms"] = precision_at_k(lt_ranked, relevant, k)
            entry[f"precision_at_{k}_tfidf"] = precision_at_k(tfidf_ranked, relevant, k)
        results[intent] = entry

        print(f"[retrieval-eval] {intent} (base_rate={base_rate:.3f}) : " +
              " | ".join(f"P@{k} latent={entry[f'precision_at_{k}_latent_terms']:.2f} "
                         f"tfidf={entry[f'precision_at_{k}_tfidf']:.2f}" for k in TOP_K))

    out_path = os.path.join(CACHE_DIR, "latent_retrieval_precision_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Écrit : {out_path}")


if __name__ == "__main__":
    main()
