"""
scripts/retrieval_demo.py — Câble src.sae.retrieval.latent_terms (jamais appelé ailleurs
dans le dépôt) à un point d'entrée testable en local.

Mails.tsv (corpus EDF réel) est absent de cette machine : ce script construit un
substitut public équivalent au format attendu par load_mails_tsv() (colonnes
index/document/segments) à partir d'échantillons FineWeb-2/Wikipedia FR déjà chargés
par src.data.preparation.prepare_domain_dataset, puis lance une requête de démonstration
via l'index BM25-sur-features-SAE de latent_terms.

Usage :
    python scripts/retrieval_demo.py --n-docs 60 --query "énergie électrique renouvelable"
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from src.config import CACHE_DIR
from src.data.preparation import prepare_domain_dataset
from src.data.keywords import ENERGY_KEYWORDS, ENERGY_URL_PATTERNS


def build_demo_mails_tsv(path: str, n_docs: int) -> None:
    """Génère un TSV au format attendu par load_mails_tsv (colonnes index/document/segments)
    à partir d'un petit échantillon public FineWeb-2/Wikipedia FR (proxy pour Mails.tsv,
    absent localement)."""
    texts = prepare_domain_dataset(
        keywords=ENERGY_KEYWORDS, domain_name="energy", n_target=n_docs,
        url_patterns=ENERGY_URL_PATTERNS, use_fineweb2=True,
    )
    df = pd.DataFrame({
        "index": range(len(texts)),
        "document": texts,
        "segments": [""] * len(texts),
    })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    print(f"[retrieval_demo] {len(df)} documents (proxy FineWeb-2) -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=60)
    ap.add_argument("--query", default="énergie électrique renouvelable")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    demo_tsv = os.path.join(CACHE_DIR, "demo_mails.tsv")
    if not os.path.exists(demo_tsv):
        build_demo_mails_tsv(demo_tsv, args.n_docs)

    from src.sae.retrieval.latent_terms import (
        load_mails_tsv, split_into_phrases, extract_f2llm_embeddings,
        load_or_train_sae, latent_doc_weights, LatentTermsIndex, DEVICE,
    )
    from src.config import D_SAE, K_SPARSE, SAVE_DIR
    import numpy as np

    df = load_mails_tsv(demo_tsv)
    texts = df["text"].tolist()
    phrases, p2d = split_into_phrases(texts, max_phrases_per_doc=20)
    p2d = np.array(p2d)
    print(f"{len(texts)} documents -> {len(phrases)} phrases")

    emb, d_in = extract_f2llm_embeddings(
        phrases, max_length=128,
        cache_path=os.path.join(CACHE_DIR, f"lt_demo_phrase_emb_n{len(phrases)}"),
    )
    sae, _ = load_or_train_sae(
        d_in=d_in, d_sae=D_SAE, k=K_SPARSE, embeddings=emb,
        save_path=os.path.join(SAVE_DIR, f"lt_demo_sae_d{D_SAE}_k{K_SPARSE}.pt"),
    )
    sae = sae.to(DEVICE)

    W_docs = latent_doc_weights(sae, emb, p2d, n_docs=len(texts))
    index = LatentTermsIndex(W_docs)

    q_emb, _ = extract_f2llm_embeddings([args.query], max_length=128, cache_path=None)
    w_q = np.asarray(
        latent_doc_weights(sae, q_emb, np.zeros(1, dtype=int), n_docs=1).todense()
    ).ravel()

    print(f"\nRequête : {args.query!r}")
    for rank, (i, s) in enumerate(index.search(w_q, top_k=args.top_k), 1):
        print(f"  #{rank}  BM25={s:8.3f}  | {texts[i][:110]}...")


if __name__ == "__main__":
    main()
