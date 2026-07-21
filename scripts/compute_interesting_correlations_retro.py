"""
scripts/compute_interesting_correlations_retro.py — Calcule find_interesting_pairs
rétroactivement pour un run produit AVANT l'ajout de cette analyse au pipeline
principal (cf. RESULTS_TESTS.md §15.3 : cooccurrence_graph n'était jamais appelée dans
saev5.py avant cette session). Réutilise test_doc_acts déjà en cache -- aucune
réextraction Gemma-3, seul bge-m3 est chargé (petit, rapide).

Usage : PYTHONPATH=. .venv/bin/python scripts/compute_interesting_correlations_retro.py results_v10_emails_main
"""
from __future__ import annotations

import json
import os
import sys

import torch

from src.analysis.cooccurrence import cooccurrence_graph, find_interesting_pairs
from src.data.preparation import build_email_train_test_corpus
from src.config import LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH


def load_label_map(cache_dir: str) -> dict[int, str]:
    with open("local_data/neuronpedia_labels/neuronpedia_labels_24-gemmascope-2-res-16k.json") as f:
        labels_core = {int(k): v for k, v in json.load(f).items()}
    judge_path = os.path.join(cache_dir, "p1_judge_labels_extended.json")
    label_map = dict(labels_core)
    if os.path.exists(judge_path):
        with open(judge_path) as f:
            judge_ext = json.load(f)
        for k, v in judge_ext.items():
            label_map[int(k)] = "[EXT] " + v.get("label", f"F{k}")
    return label_map


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "results_v10_emails_main"
    cache_dir = os.path.join(run_dir, "cache")

    print("[retro-corr] Reconstruction du split train/test (déterministe, pas de GPU)...")
    train_texts, _, test_texts, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=42,
    )
    n_train, n_test = len(train_texts), len(test_texts)
    print(f"[retro-corr] n_train={n_train}, n_test={n_test}")

    acts_path = os.path.join(cache_dir, "p1_all_doc_acts_ext_d1024.pt")
    if not os.path.exists(acts_path):
        acts_path = os.path.join(cache_dir, "p1_all_doc_acts.pt")
    all_doc_acts = torch.load(acts_path, map_location="cpu", weights_only=True)
    test_doc_acts = all_doc_acts[n_train:n_train + n_test]
    print(f"[retro-corr] test_doc_acts: {test_doc_acts.shape}")

    label_map = load_label_map(cache_dir)

    print("[retro-corr] Construction du graphe de cooccurrence (NPMI)...")
    G = cooccurrence_graph(test_doc_acts, feature_labels=label_map)
    print(f"[retro-corr] Graphe : {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes.")

    if G.number_of_edges() == 0:
        print("[retro-corr] Aucune arête -- rien à filtrer.")
        interesting_pairs = []
    else:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "sae"))
        from src.sae.saev5 import _embed_bge_m3
        node_ids = list(G.nodes)
        label_embs = _embed_bge_m3([G.nodes[n]["label"] for n in node_ids])
        label_embeddings = {n: label_embs[i].numpy() for i, n in enumerate(node_ids)}
        interesting_pairs = find_interesting_pairs(G, label_embeddings)

    out_path = os.path.join(run_dir, "p1_interesting_correlations.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(interesting_pairs, f, indent=2, ensure_ascii=False)
    print(f"[retro-corr] {len(interesting_pairs)} paires intéressantes -> {out_path}")


if __name__ == "__main__":
    main()
