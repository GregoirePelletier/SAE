"""
scripts/baseline_gemmascope.py — Baseline stricte : SAE GemmaScope 2 NON modifié
(pas de FrozenCoreResidualSAE, pas d'extension d_extra) sur mails originaux + augmentés.

Zéro nouveau code d'inférence : ne fait qu'orchestrer l'existant.
  - src/sae/gemma_scope_loader.load_gemma_scope_sae   (SAE natif sae-lens)
  - src/analysis/activations.extract_residual_acts    (stream résidus L24, fp32)
  - src/analysis/activations.maxpool_sae_docs         (max_t SAE(x_t), token-level)
  - src/analysis/cooccurrence.corpus_diff_stats       (Fisher + BH)
  - src/sae/neuronpedia_labels                        (labels officiels)
  - src/data/augmentation.load_augmented              (variantes tracées)
  - src/storage/fragment_store                        (cache CSR fp32)

Question posée : pour chaque axe de perturbation, quelles features GemmaScope
de base changent significativement de fréquence d'activation vs mails originaux ?
"""
from __future__ import annotations

import os

import pandas as pd
import torch

from src.sae.gemma_scope_loader import load_gemma_scope_sae
from src.sae.neuronpedia_labels import fetch_neuronpedia_labels, merge_with_judge_labels
from src.analysis.activations import extract_residual_acts, maxpool_sae_docs
from src.analysis.cooccurrence import corpus_diff_stats
from src.analysis.visualization import plot_corpus_diff
from src.data.preparation import load_and_clean_emails
from src.data.augmentation import load_augmented

CACHE = os.environ.get("CACHE_DIR", "cache_baseline")
os.makedirs(CACHE, exist_ok=True)
DEVICE = "cuda"


def encode_corpus(texts: list[str], sae, model, tokenizer, tag: str) -> torch.Tensor:
    """doc_acts [n_docs, 262144] max-poolés token-level, cachés en .pt."""
    path = os.path.join(CACHE, f"baseline_doc_acts_{tag}.pt")
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")
    stream = extract_residual_acts(texts, model, tokenizer, layer=24, device=DEVICE)
    # SAE natif : encode() sae-lens, AUCUNE extension. fp32 obligatoire (outliers ~130k).
    doc_acts = maxpool_sae_docs(
        act_stream=stream,
        encode_fn=lambda x: sae.encode(x.to(sae.W_enc.dtype)),
        n_docs=len(texts),
        d_sae=sae.cfg.d_sae,
        device=DEVICE,
    )
    torch.save(doc_acts, path)
    return doc_acts


def main(mails_tsv: str, augmented_jsonl: str):
    # 1. SAE GemmaScope tel quel + labels officiels
    sae = load_gemma_scope_sae(os.environ["LOCAL_SAE_DIR"], device=DEVICE)
    np_labels = fetch_neuronpedia_labels(
        layer=24, width="262k",
        cache_path=os.path.join(CACHE, "neuronpedia_labels_l24_262k.json"),
    )
    labels = merge_with_judge_labels(np_labels, "p1_saebench_judge_labels.json")

    # 2. Corpus
    texts_orig, _ = load_and_clean_emails(mails_tsv)
    df_orig = pd.DataFrame({"text": texts_orig, "group": "original",
                            "aug_axis": None, "aug_level": None})
    df_aug = load_augmented(augmented_jsonl)[["text", "aug_axis", "aug_level"]]
    df_aug["group"] = "augmented"
    df = pd.concat([df_orig, df_aug], ignore_index=True)

    # 3. Modèle + activations (charger/décharger comme saev5 ; malloc_trim après)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-12b-it", torch_dtype=torch.bfloat16, device_map=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")

    doc_acts = encode_corpus(df["text"].tolist(), sae, model, tokenizer, tag="all")

    del model
    import ctypes, gc
    gc.collect(); torch.cuda.empty_cache()
    ctypes.CDLL("libc.so.6").malloc_trim(0)

    # 4. Diff par axe puis par niveau : originaux vs perturbés
    reports = {}
    for axis in df["aug_axis"].dropna().unique():
        for level in df.loc[df["aug_axis"] == axis, "aug_level"].unique():
            mask_pair = (df["group"] == "original") | (
                (df["aug_axis"] == axis) & (df["aug_level"] == level))
            sub_acts = doc_acts[mask_pair.values]
            group_mask = (df.loc[mask_pair, "group"] == "augmented").values
            diff = corpus_diff_stats(sub_acts, group_mask, feature_labels=labels)
            key = f"{axis}__{level}"
            diff.to_csv(os.path.join(CACHE, f"diff_{key}.csv"), index=False)
            plot_corpus_diff(diff, path=os.path.join(CACHE, f"diff_{key}.html"))
            reports[key] = diff
            n_sig = int(diff["significant"].sum())
            print(f"[baseline] {key}: {n_sig} features significatives "
                  f"(top: {diff.iloc[0]['label'][:60]!r}, LOR={diff.iloc[0]['log_odds_ratio']:.2f})")
    return reports


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])
