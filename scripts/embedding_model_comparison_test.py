"""
scripts/embedding_model_comparison_test.py — Diagnostic ponctuel : F2LLM-v2-80M
donne de bons résultats pour select_latents_by_similarity sur la requête "urgence
réclamation client" mais de mauvais résultats sur "facturation résiliation panne"
(labels sans rapport). Teste si bge-m3 (multilingue, déjà présent dans models/,
potentiellement mieux adapté au matching cross-lingue court FR<->labels FR/EN)
fait mieux, sur GPU pour éviter la lenteur CPU observée avec F2LLM (~11-20 min
par requête sur 13,7k labels).

Usage : PYTHONPATH=. .venv/bin/python scripts/embedding_model_comparison_test.py
"""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUERIES = ["urgence réclamation client", "facturation résiliation panne"]


def load_label_map():
    with open("local_data/neuronpedia_labels/neuronpedia_labels_24-gemmascope-2-res-16k.json") as f:
        labels_core = {int(k): v for k, v in json.load(f).items()}
    with open("results_v10_emails_main/cache/p1_judge_labels_extended.json") as f:
        judge_ext = json.load(f)
    label_map = dict(labels_core)
    for k, v in judge_ext.items():
        label_map[int(k)] = "[EXT] " + v.get("label", f"F{k}")
    return label_map


def embed_bge_m3(texts: list[str], batch_size: int = 64) -> torch.Tensor:
    """[CLS] pooling normalisé -- convention BGE (cf. model card bge-m3)."""
    tokenizer = AutoTokenizer.from_pretrained("models/bge-m3", local_files_only=True)
    model = AutoModel.from_pretrained("models/bge-m3", local_files_only=True).to(DEVICE).eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEVICE)
            out = model(**enc)
            cls = out.last_hidden_state[:, 0]
            embs.append(F.normalize(cls, p=2, dim=-1).cpu())
    return torch.cat(embs, dim=0)


def main():
    label_map = load_label_map()
    items = [(f, lbl) for f, lbl in label_map.items() if lbl and not __import__("re").fullmatch(r"(\[EXT\]\s*)?F\d+", lbl.strip())]
    f_ids, labels = zip(*items)
    print(f"[compare] {len(labels)} labels à comparer, modèle bge-m3, device={DEVICE}")

    all_texts = list(QUERIES) + list(labels)
    embs = embed_bge_m3(all_texts)
    query_embs, label_embs = embs[:len(QUERIES)], embs[len(QUERIES):]

    for qi, q in enumerate(QUERIES):
        sims = (label_embs @ query_embs[qi]).numpy()
        order = sims.argsort()[::-1][:15]
        print(f"\n--- bge-m3 : requête {q!r} ---")
        for idx in order:
            print(f"    {f_ids[idx]} {labels[idx]} (sim={sims[idx]:.3f})")


if __name__ == "__main__":
    main()
