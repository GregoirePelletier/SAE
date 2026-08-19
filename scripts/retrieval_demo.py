"""
scripts/retrieval_demo.py — Câble src.sae.retrieval.latent_terms (jamais appelé ailleurs
dans le dépôt en dehors de latent_retrieval_precision_eval.py) à un point d'entrée
testable en local, sans Mails.tsv.

Mails.tsv (corpus EDF réel) est absent de cette machine : ce script construit un
substitut public équivalent au format attendu par load_mails_tsv() (colonnes
index/document/segments) à partir d'échantillons FineWeb-2/Wikipedia FR déjà chargés
par src.data.preparation.prepare_domain_dataset, puis lance une requête de démonstration
via l'index BM25-sur-features-SAE token-level de latent_terms (le SAE lui-même
s'entraîne, comme en production, sur un échantillon FineWeb2-fr générique distinct de
ce corpus de démo).

Usage :
    python scripts/retrieval_demo.py --n-docs 60 --query "énergie électrique renouvelable"
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from src.config import CACHE_DIR, SAVE_DIR, D_SAE, K_SPARSE, EMB_MODEL
from src.data.preparation import prepare_domain_dataset
from src.data.keywords import ENERGY_KEYWORDS, ENERGY_URL_PATTERNS
from src.sae.retrieval.latent_terms import (
    load_f2llm, build_token_training_pool, load_or_train_latent_terms_sae,
    latent_doc_weights, LatentTermsIndex, TRAIN_TOKENS,
)


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
    ap.add_argument("--demo-train-tokens", type=int, default=min(TRAIN_TOKENS, 2_000_000),
                     help="Pool d'entraînement du SAE réduit pour une démo rapide (le run "
                          "de production utilise LT_TRAIN_TOKENS, défaut 33M).")
    args = ap.parse_args()

    demo_tsv = os.path.join(CACHE_DIR, "demo_mails.tsv")
    if not os.path.exists(demo_tsv):
        build_demo_mails_tsv(demo_tsv, args.n_docs)

    from src.data.dataset import load_mails_tsv
    df = load_mails_tsv(demo_tsv)
    texts = df["text"].tolist()
    print(f"{len(texts)} documents (proxy de démo).")

    tokenizer, model = load_f2llm()
    d_in = model.config.hidden_size
    model_tag = os.path.basename(EMB_MODEL.rstrip("/"))

    token_pool = build_token_training_pool(
        args.demo_train_tokens, tokenizer, model,
        cache_path=os.path.join(CACHE_DIR, f"lt_demo_token_pool_n{args.demo_train_tokens}_{model_tag}"))
    sae, _ = load_or_train_latent_terms_sae(
        d_in=d_in, d_sae=D_SAE, k=K_SPARSE, token_pool=token_pool,
        save_path=os.path.join(SAVE_DIR, f"lt_demo_sae_d{D_SAE}_k{K_SPARSE}_{model_tag}.pt"))

    W_docs = latent_doc_weights(sae, texts, tokenizer, model)
    index = LatentTermsIndex(W_docs)

    W_q = latent_doc_weights(sae, [args.query], tokenizer, model)
    w_q = np.asarray(W_q.todense()).ravel()

    print(f"\nRequête : {args.query!r}")
    for rank, (i, s) in enumerate(index.search(w_q, top_k=args.top_k), 1):
        print(f"  #{rank}  BM25={s:8.3f}  | {texts[i][:110]}...")


if __name__ == "__main__":
    main()
