"""
scripts/audit_2026_08_e9_size_matched_embedding_compare.py — E.9
(docs/AUDIT_2026-08.md) : `embedding_model_comparison_test.py` (job 40730)
concluait que bge-m3 bat F2LLM pour `select_latents_by_similarity`, mais
comparait bge-m3 (≈568M paramètres, 4,3 Go sur disque) à F2LLM-v2-80M
(≈80M, 166 Mo) -- un écart de taille ≈26x, confond non contrôlé entre
architecture et taille de modèle. Ce script refait la même comparaison à
taille bien plus proche : bge-m3 vs F2LLM-v2-330M (653 Mo, ≈330M, la
variante que le projet a par ailleurs jugée "assez grande" pour la Pipeline 2,
RESULTS_TESTS.md §16.4) -- écart réduit à ≈6,6x au lieu de ≈26x.

Mêmes 2 requêtes, même jeu de labels, top-15 des deux modèles affiché côte à
côte pour comparaison directe. F2LLM embeddé par mean-pooling masqué (même
convention que `phrase_sae.py::_mean_pool`, PAS le CLS pooling de bge-m3 --
convention propre à chaque famille de modèle).

Usage : sbatch slurm/validation/run_audit_e9_size_matched_embedding_compare.slurm
"""
from __future__ import annotations

import json
import re

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUERIES = ["urgence réclamation client", "facturation résiliation panne"]
BGE_M3_PATH = "models/bge-m3"
F2LLM_330M_PATH = "models/F2LLM-v2-330M"
OUT_PATH = "docs/audit_2026_08_e9_size_matched_embedding_compare_results.json"


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
    tokenizer = AutoTokenizer.from_pretrained(BGE_M3_PATH, local_files_only=True)
    model = AutoModel.from_pretrained(BGE_M3_PATH, local_files_only=True).to(DEVICE).eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEVICE)
            out = model(**enc)
            cls = out.last_hidden_state[:, 0]
            embs.append(F.normalize(cls, p=2, dim=-1).cpu())
    del model
    return torch.cat(embs, dim=0)


def embed_f2llm(texts: list[str], batch_size: int = 64) -> torch.Tensor:
    """Mean-pooling masqué -- même convention que phrase_sae.py::_mean_pool."""
    tokenizer = AutoTokenizer.from_pretrained(F2LLM_330M_PATH, local_files_only=True)
    model = AutoModel.from_pretrained(F2LLM_330M_PATH, local_files_only=True).to(DEVICE).eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEVICE)
            out = model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
            pooled = torch.sum(out.last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            embs.append(F.normalize(pooled, p=2, dim=-1).cpu())
    del model
    return torch.cat(embs, dim=0)


def top15(f_ids, labels, embs, query_embs, qi):
    sims = (embs @ query_embs[qi]).numpy()
    order = sims.argsort()[::-1][:15]
    return [{"f_id": f_ids[i], "label": labels[i], "sim": float(sims[i])} for i in order]


def main():
    label_map = load_label_map()
    items = [(f, lbl) for f, lbl in label_map.items()
             if lbl and not re.fullmatch(r"(\[EXT\]\s*)?F\d+", lbl.strip())]
    f_ids, labels = zip(*items)
    print(f"[e9-sizematch] {len(labels)} labels, bge-m3 (568M) vs F2LLM-v2-330M (330M) -- "
          f"écart de taille réduit de ~26x (job 40730) à ~6,6x.")

    all_texts = list(QUERIES) + list(labels)

    print("[e9-sizematch] Embedding bge-m3...")
    embs_bge = embed_bge_m3(all_texts)
    q_bge, l_bge = embs_bge[:len(QUERIES)], embs_bge[len(QUERIES):]

    print("[e9-sizematch] Embedding F2LLM-v2-330M...")
    embs_f2llm = embed_f2llm(all_texts)
    q_f2llm, l_f2llm = embs_f2llm[:len(QUERIES)], embs_f2llm[len(QUERIES):]

    results = {}
    for qi, q in enumerate(QUERIES):
        top_bge = top15(f_ids, labels, l_bge, q_bge, qi)
        top_f2llm = top15(f_ids, labels, l_f2llm, q_f2llm, qi)
        results[q] = {"bge_m3": top_bge, "f2llm_v2_330m": top_f2llm}
        print(f"\n--- requête {q!r} ---")
        print("  bge-m3 (568M):")
        for r in top_bge[:10]:
            print(f"    {r['f_id']} {r['label']} (sim={r['sim']:.3f})")
        print("  F2LLM-v2-330M (330M):")
        for r in top_f2llm[:10]:
            print(f"    {r['f_id']} {r['label']} (sim={r['sim']:.3f})")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Écrit : {OUT_PATH}")


if __name__ == "__main__":
    main()
