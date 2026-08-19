"""Validation ponctuelle (pas un test formel) : reproduit les cellules de calcul
de external/interp_embed/examples/diff_models.ipynb sur les .pkl précomputés
(nickjiang/feature_labels, artifacts/gpt52_gemini3_tutorial/) déjà en cache HF
local. Aucune inférence modèle -- valide seulement que interp_embed s'importe et
se comporte comme documenté sur des activations SAE déjà calculées. CPU-only.
"""
import sys
sys.path.insert(0, "/home/h21486/SAE/external/interp_embed")

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from interp_embed import Dataset

repo_id = "nickjiang/feature_labels"
paths = {
    name: hf_hub_download(repo_id=repo_id, filename=f"artifacts/gpt52_gemini3_tutorial/{name}")
    for name in ["prompts.pkl", "gpt-52.pkl", "gemini-3.pkl"]
}

print("Loading datasets...", flush=True)
gpt = Dataset.load_from_file(paths["gpt-52.pkl"])
gemini = Dataset.load_from_file(paths["gemini-3.pkl"])
prompts = Dataset.load_from_file(paths["prompts.pkl"])
print(f"GPT 5.2: {len(gpt)} documents", flush=True)
print(f"Gemini 3: {len(gemini)} documents", flush=True)
print(f"Prompts: {len(prompts)} documents", flush=True)


def diff_features(ds1, ds2):
    fa1 = ds1.latents("binarize")
    fa2 = ds2.latents("binarize")
    freq1 = np.sum(fa1, axis=0) / fa1.shape[0]
    freq2 = np.sum(fa2, axis=0) / fa2.shape[0]
    diff = freq1 - freq2
    labels = ds1.feature_labels()
    n = freq1.shape[0]
    return pd.DataFrame({
        "feature": [labels.get(i, "") for i in range(n)],
        "feature_id": np.arange(n),
        "gpt": freq1, "gemini": freq2, "diff": diff,
    })


df = diff_features(gpt, gemini)
df = df.sort_values(by="diff", ascending=False, key=abs).reset_index(drop=True)
print("\n=== TOP 20 FEATURES (|diff|) ===", flush=True)
print(df.head(20).to_string(), flush=True)

expected_top = {24183, 2805, 61439, 7398, 20863}
got_top = set(df.head(5)["feature_id"].tolist())
print(f"\nOverlap with notebook's expected top-5 feature IDs: {got_top & expected_top} / {expected_top}", flush=True)
