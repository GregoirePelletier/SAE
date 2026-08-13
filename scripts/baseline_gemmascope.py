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
from src.sae.neuronpedia_labels import fetch_neuronpedia_labels
from src.analysis.activations import extract_residual_acts, maxpool_sae_docs
from src.analysis.cooccurrence import corpus_diff_stats
from src.analysis.visualization import plot_corpus_diff
from src.data.preparation import load_and_clean_emails
from src.data.augmentation import load_augmented
from src.config import (
    MODEL_ID, LAYER, SAE_ID, LOCAL_SAE_ROOT, SAE_SNAPSHOT, HOOK_TYPE, DTYPE,
    NEURONPEDIA_LABELS_PATH,
)

CACHE = os.environ.get("CACHE_DIR", "cache_baseline")
os.makedirs(CACHE, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if DTYPE == "bf16" else torch.float16

# Ex: "layer_12_width_65k_l0_medium" -> largeur "65k" (Neuronpedia).
_SAE_WIDTH = SAE_ID.split("_width_")[1].split("_")[0] if "_width_" in SAE_ID else "16k"


def encode_corpus(texts: list[str], sae, model, tokenizer, tag: str) -> torch.Tensor:
    """doc_acts [n_docs, d_sae] max-poolés token-level, cachés en .pt."""
    path = os.path.join(CACHE, f"baseline_doc_acts_{tag}.pt")
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")
    stream = extract_residual_acts(texts, model, tokenizer, layer=LAYER, device=DEVICE)
    # SAE natif : encode() sae-lens, AUCUNE extension. fp32 obligatoire (outliers massifs).
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
    sae_dir = os.path.join(LOCAL_SAE_ROOT, "snapshots", SAE_SNAPSHOT, HOOK_TYPE, SAE_ID)
    sae = load_gemma_scope_sae(sae_dir, device=DEVICE)
    model_np_id = MODEL_ID.split("/")[-1] if "/" in MODEL_ID else MODEL_ID
    # Cache partagé canonique (src/config.py) : réutilisé par tous les scripts/runs,
    # jamais dupliqué par run, jamais re-téléchargé une fois présent.
    np_labels = fetch_neuronpedia_labels(
        model_id=model_np_id, layer=LAYER, width=_SAE_WIDTH,
        cache_path=NEURONPEDIA_LABELS_PATH,
    )
    # merge_with_judge_labels("p1_saebench_judge_labels.json") retiré : ce script
    # utilise le SAE GemmaScope natif (pas de FrozenCoreResidualSAE/d_extra) -> il
    # n'existe aucune feature d'extension à labelliser par le juge ici, l'appel
    # était un no-op (fichier jamais présent au chemin relatif attendu).
    labels = np_labels

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
        MODEL_ID, torch_dtype=TORCH_DTYPE, device_map=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    doc_acts = encode_corpus(df["text"].tolist(), sae, model, tokenizer, tag="all")

    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if os.name != "nt":
        import ctypes
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

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
            if len(diff) == 0:
                # Sous-corpus trop petit / features trop sparses pour cette paire
                # axe/niveau : corpus_diff_stats renvoie un DataFrame vide (cf. garde
                # ajoutée dans cooccurrence.py) plutôt que de planter ici sur iloc[0].
                print(f"[baseline] {key}: 0 feature active (corpus trop petit pour ce sous-ensemble).")
                continue
            n_sig = int(diff["significant"].sum())
            print(f"[baseline] {key}: {n_sig} features significatives "
                  f"(top: {diff.iloc[0]['label'][:60]!r}, LOR={diff.iloc[0]['log_odds_ratio']:.2f})")
    return reports


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])
