"""
scripts/latent_retrieval_precision_eval.py — Évaluation quantitative (jamais faite)
de src/sae/retrieval/latent_terms.py (Latent Terms, Clavié et al. 2026,
arXiv:2605.29384 : BM25 sur le vocabulaire latent d'un SAE entraîné par pure
reconstruction sur les phrases, Pipeline 2).

Jusqu'ici, ce module n'était exercé QUE via scripts/retrieval_demo.py (1-2 requêtes
inspectées à l'œil, sur des données de substitution FineWeb2/Wikipedia -- écrit sur
une machine sans Mails.tsv) et le dashboard (parcours interactif, pas de métrique).
Sur le cluster, Mails.tsv (mails originaux) est disponible : ce script construit un
vrai protocole précision/rappel en utilisant les labels faibles d'intention déjà
présents dans le corpus (`src.data.dataset.INTENT_KEYWORDS_FR`, régime réutilisé par
scripts/intent_urgency_probe.py, explanation_fidelity_test.py,
steering_fidelity_test.py) comme vérité terrain de substitution.

Protocole :
  1. Corpus = mails originaux de Mails.tsv (3474 mails), 4 intentions déjà validées
     comme suffisamment équilibrées (>=30 positifs) dans les tests précédents :
     réclamation, remboursement, information, urgence.
  2. Pour chaque intention, une requête en langage naturel PARAPHRASANT (pas copiant
     mot pour mot) le motif regex de l'intention -- teste la généralisation
     sémantique, pas juste le rappel de mots-clés exacts.
  3. Index Latent Terms (BM25 sur activations SAE de phrases, F2LLM-v2-80M +
     PhraseLevelSAE entraîné ICI par pure reconstruction sur ce corpus, dim320/8192/
     k16 -- mêmes défauts que Pipeline 2) construit sur l'ensemble des 3474 mails.
  4. Baseline de comparaison : TF-IDF + cosinus sur le texte brut, mêmes requêtes,
     même corpus -- un système de retrieval "mots" classique et bien compris.
  5. Métrique : Precision@10 et Precision@20 (fraction de documents pertinents,
     au sens du label faible d'intention, dans le top-k), comparée au taux de base
     de l'intention dans le corpus (performance d'un tirage aléatoire).

Coût : F2LLM-v2-80M sur ~3474 mails (quelques milliers de phrases après découpage,
`split_into_phrases`) + entraînement d'un petit PhraseLevelSAE dédié -- GPU requis
mais très rapide (quelques minutes), pas de Gemma-3-12B.

Usage : PYTHONPATH=. .venv/bin/python scripts/latent_retrieval_precision_eval.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import LOCAL_MAILS_PATH, SAVE_DIR, CACHE_DIR, D_SAE, K_SPARSE
from src.data.dataset import load_mails_tsv
from src.data.preparation import split_into_phrases
from src.sae.phrase_sae import extract_f2llm_embeddings, load_or_train_sae
from src.sae.retrieval.latent_terms import latent_doc_weights, LatentTermsIndex

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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

    print("[retrieval-eval] Découpage en phrases + embeddings F2LLM...")
    phrases, p2d = split_into_phrases(texts, max_phrases_per_doc=20)
    p2d = np.array(p2d)
    print(f"[retrieval-eval] {len(phrases)} phrases.")

    emb, d_in = extract_f2llm_embeddings(
        phrases, max_length=128,
        cache_path=os.path.join(CACHE_DIR, f"lt_eval_phrase_emb_n{len(phrases)}"),
    )
    print("[retrieval-eval] Entraînement du PhraseLevelSAE (reconstruction pure)...")
    sae, _ = load_or_train_sae(
        d_in=d_in, d_sae=D_SAE, k=K_SPARSE, embeddings=emb,
        save_path=os.path.join(SAVE_DIR, f"lt_eval_sae_d{D_SAE}_k{K_SPARSE}.pt"),
    )
    sae = sae.to(DEVICE)

    W_docs = latent_doc_weights(sae, emb, p2d, n_docs=len(texts))
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
        q_emb, _ = extract_f2llm_embeddings([query], max_length=128, cache_path=None)
        w_q = np.asarray(
            latent_doc_weights(sae, q_emb, np.zeros(1, dtype=int), n_docs=1).todense()
        ).ravel()
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
