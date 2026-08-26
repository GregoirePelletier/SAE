"""
scripts/npmi_similarity_diagnostic.py — diagnostic ponctuel : pourquoi
npmi_verified_test.py trouve 0 paires candidates (NPMI>0.3, sim label<0.2)
parmi les 68 features interprétables de results_v10_emails_main. Affiche la
distribution complète (npmi, sim) pour toutes les paires NPMI>0.1, sans
charger de juge (juste bge-m3, plus léger) -- pas destiné à produire un
résultat, juste à calibrer les seuils.

Usage :
    SAVE_DIR=./results_v10_emails_main/ PYTHONPATH=. \
      .venv/bin/python scripts/npmi_similarity_diagnostic.py
"""
import json
import os
import sys

import torch
from src.sae.sae_shared import load_all_doc_acts
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import SAVE_DIR, LATENT_LABEL_EMB_MODEL
from src.analysis.cooccurrence import compute_npmi

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = os.path.join(SAVE_DIR, "cache")
EXT_FEATURES_PATH = os.path.join(SAVE_DIR, "p1_top_extended_features.json")


def _embed(texts, batch_size=64):
    tok = AutoTokenizer.from_pretrained(LATENT_LABEL_EMB_MODEL, local_files_only=True)
    mdl = AutoModel.from_pretrained(LATENT_LABEL_EMB_MODEL, local_files_only=True).to(DEVICE).eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=64,
                      return_tensors="pt").to(DEVICE)
            cls = mdl(**enc).last_hidden_state[:, 0]
            embs.append(F.normalize(cls, p=2, dim=-1).cpu())
    return torch.cat(embs, dim=0)


def main():
    with open(EXT_FEATURES_PATH, encoding="utf-8") as f:
        ext_data = json.load(f)
    feature_labels = {int(fid): v["label"] for fid, v in ext_data.items() if v.get("interp_score") == 1}
    fids = sorted(feature_labels.keys())
    print(f"{len(fids)} features interprétables", flush=True)

    all_doc_acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts_ext_d1024.pt")
    if not os.path.exists(all_doc_acts_path):
        all_doc_acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts.pt")
    all_doc_sae_acts = load_all_doc_acts(all_doc_acts_path)
    sub_acts = all_doc_sae_acts[:, fids]
    npmi = compute_npmi(sub_acts)

    label_embs = _embed([feature_labels[fid] for fid in fids])

    rows = []
    n = len(fids)
    for a in range(n):
        for b in range(a + 1, n):
            v = float(npmi[a, b])
            if v > 0.1:
                sim = float(label_embs[a] @ label_embs[b])
                rows.append((v, sim, feature_labels[fids[a]], feature_labels[fids[b]]))
    rows.sort(key=lambda r: r[0], reverse=True)
    print(f"\n{len(rows)} paires avec NPMI>0.1, triées par NPMI décroissant :\n", flush=True)
    for v, sim, li, lj in rows:
        print(f"  npmi={v:.3f} sim={sim:.3f}  {li!r:35s} <-> {lj!r:35s}", flush=True)

    # Distribution des sim pour les paires NPMI>0.3 (seuil actuel du script principal)
    high_npmi = [r for r in rows if r[0] > 0.3]
    if high_npmi:
        sims = [r[1] for r in high_npmi]
        print(f"\n{len(high_npmi)} paires NPMI>0.3 : sim min={min(sims):.3f} max={max(sims):.3f} "
              f"median={sorted(sims)[len(sims)//2]:.3f}", flush=True)


if __name__ == "__main__":
    main()
