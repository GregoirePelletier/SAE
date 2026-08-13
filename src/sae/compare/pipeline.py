"""
pipeline.py — Orchestration bout-en-bout (exécutable).
Usage:
  python -m src.sae.compare.pipeline --mails Mails.tsv --mode p1  # Pipeline 1 (token-level, Gemma)
  python -m src.sae.compare.pipeline --mails Mails.tsv --mode compare \
      --model-a F2LLM-v2-80M --model-b intfloat/multilingual-e5-small
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch

from ...data.dataset import load_mails_tsv, build_bilingual_corpus
from ...analysis.activations import extract_residual_acts, maxpool_sae_docs
from ...analysis.cooccurrence import cooccurrence_graph, corpus_diff_stats, cluster_in_feature_space
from .model_compare import compare_embedding_models
from .crosslingual import crosslingual_alignment, downstream_report, ce_loss_increase
from ...analysis import visualization as viz
from ...config import DTYPE

OUT = Path("results_v9")
_DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_TORCH_DTYPE = torch.bfloat16 if DTYPE == "bf16" else torch.float16


def embed_corpus(texts: list[str], model_name: str, device=_DEFAULT_DEVICE, batch=32) -> torch.Tensor:
    """Embeddings phrase-level mean-poolés (Pipeline 2)."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name, torch_dtype=_TORCH_DTYPE).to(device).eval()
    embs = []
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            enc = tok(texts[s:s + batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=512)
            ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            out = mdl(input_ids=ids, attention_mask=attn)
            try:
                from src.sae.phrase_sae import _mean_pool   # implémentation unique du mean-pooling masqué
            except ImportError:
                from phrase_sae import _mean_pool
            embs.append(_mean_pool(out, attn).float().cpu())
    return torch.cat(embs)


def train_phrase_sae(emb, d_sae=2048, k=16, epochs=60, tag="A", device=_DEFAULT_DEVICE):
    """Délègue au harnais d'entraînement du repo : BatchTopKEncoder (θ), AuxK, cache load_or_train."""
    try:
        from src.sae.phrase_sae import load_or_train_sae
    except ImportError:
        from phrase_sae import load_or_train_sae
    OUT.mkdir(exist_ok=True)
    sae, _ = load_or_train_sae(emb.shape[1], d_sae, k, emb,
                               str(OUT / f"phrase_sae_{tag}.pt"), epochs=epochs)
    sae.eval()
    with torch.no_grad():
        Z = torch.cat([sae.encode(emb[s:s + 2048].to(device)).float().cpu()
                       for s in range(0, len(emb), 2048)])
    return sae, Z


def run_compare(args):
    df = load_mails_tsv(args.mails)
    texts = df["text"].tolist()
    labels = df.filter(like="intent_").astype(int).values.argmax(1) \
        if df.filter(like="intent_").shape[1] else np.zeros(len(df), int)

    embA = embed_corpus(texts, args.model_a)
    embB = embed_corpus(texts, args.model_b)
    _, ZA = train_phrase_sae(embA, tag='A')
    _, ZB = train_phrase_sae(embB, tag='B')

    res = compare_embedding_models(ZA, ZB, embA, embB, labels)
    OUT.mkdir(exist_ok=True)
    res["matches"].to_parquet(OUT / "matches.parquet")
    res["report_A"].per_feature.to_parquet(OUT / "pollution_A.parquet")
    res["report_B"].per_feature.to_parquet(OUT / "pollution_B.parquet")
    viz.plot_pollution_report(res["report_A"], OUT / "pollution_A.html")
    viz.plot_pollution_report(res["report_B"], OUT / "pollution_B.html")
    print(f"verdict={res['verdict']}  mean_match_corr={res['mean_match_corr']:.3f}  "
          f"A: {res['report_A'].n_flagged} flagged / mass {res['report_A'].model_score:.3f}  "
          f"B: {res['report_B'].n_flagged} / {res['report_B'].model_score:.3f}")


def run_analysis(args):
    df = load_mails_tsv(args.mails)
    texts = df["text"].tolist()
    emb = embed_corpus(texts, args.model_a)
    _, Z = train_phrase_sae(emb)

    OUT.mkdir(exist_ok=True)
    G = cooccurrence_graph(Z)
    viz.plot_cooccurrence_graph(G, OUT / "cooc_graph.html")
    lab2d, e2d = cluster_in_feature_space(Z)
    viz.plot_feature_space_clusters(e2d, lab2d, texts, OUT / "clusters_sae.html")
    if df["intent_reclamation"].any():
        d = corpus_diff_stats(Z, df["intent_reclamation"].values)
        d.to_parquet(OUT / "diff_reclamation.parquet")
        viz.plot_corpus_diff(d, path=OUT / "corpus_diff.html")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mails", required=True)
    p.add_argument("--mode", choices=["analysis", "compare"], default="analysis")
    p.add_argument("--model-a", default="codefuse-ai/F2LLM-v2-80M")
    p.add_argument("--model-b", default="intfloat/multilingual-e5-small")
    a = p.parse_args()
    (run_compare if a.mode == "compare" else run_analysis)(a)