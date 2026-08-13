"""
scripts/interesting_pairs_synthetic_validation.py — validation par injection
synthétique de `find_interesting_pairs` (`src/analysis/cooccurrence.py`),
à la manière du papier de référence (interp_embed, Appendix E.2).

Le biais "Objet:" du corpus augmenté (`RESULTS_TESTS.md` §14.1) est désormais
filtré au chargement (`src/data/augmentation.py`), donc plus reproductible
directement comme cas réel -- ce script reproduit le PRINCIPE de la
validation par injection synthétique : construire des
activations SAE synthétiques avec une corrélation connue et contrôlée entre deux
features (co-occurrence forte, labels sémantiquement dissimilaires -- exactement
le signal que `find_interesting_pairs` est censé isoler), et vérifier que la
fonction la retrouve. Si elle échoue à la retrouver sur un signal injecté
délibérément fort, la fonction ne peut pas être fiable sur un signal réel plus
faible.

Usage (CPU uniquement, aucune dépendance aux runs existants) :
    PYTHONPATH=. .venv/bin/python scripts/interesting_pairs_synthetic_validation.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.cooccurrence import cooccurrence_graph, find_interesting_pairs

OUT_PATH = "./local_data/interesting_pairs_synthetic_validation_results.json"
SEED = 42


def build_synthetic_corpus(n_docs: int = 2000, n_features: int = 200, seed: int = SEED):
    """doc_acts synthétique : bruit de fond sparse aléatoire + UNE paire de
    features injectée avec co-occurrence forte et contrôlée (actives ensemble
    dans ~40% des docs, jamais l'une sans l'autre) -- imite un artefact de
    génération partagé (type "Objet:") entre deux concepts a priori sans rapport."""
    rng = np.random.default_rng(seed)
    acts = (rng.random((n_docs, n_features)) < 0.03).astype(np.float32)  # bruit de fond sparse
    injected_a, injected_b = n_features // 20, n_features * 3 // 4  # indices arbitraires, écartés
    co_mask = rng.random(n_docs) < 0.4
    acts[co_mask, injected_a] = 1.0
    acts[co_mask, injected_b] = 1.0
    acts[:, injected_a] = np.where(co_mask, 1.0, acts[:, injected_a] * 0.0)  # jamais actif hors injection
    acts[:, injected_b] = np.where(co_mask, 1.0, acts[:, injected_b] * 0.0)
    return torch.tensor(acts), injected_a, injected_b


def main() -> None:
    doc_acts, feat_a, feat_b = build_synthetic_corpus()

    # Labels textuels DÉLIBÉRÉMENT dissimilaires (embeddings synthétiques orthogonaux)
    # -- simule deux concepts métier sans rapport partageant un artefact commun.
    feature_labels = {feat_a: "réclamation facturation", feat_b: "compteur linky raccordement"}
    label_embeddings = {
        feat_a: np.array([1.0, 0.0, 0.0]),
        feat_b: np.array([0.0, 1.0, 0.0]),
    }
    # Toutes les autres features actives reçoivent un embedding aléatoire mais
    # PROCHE l'un de l'autre (simule des synonymes/quasi-doublons qui ne doivent
    # PAS être remontés comme "intéressants" même si NPMI élevé).
    rng = np.random.default_rng(SEED)
    freq = (doc_acts > 1e-6).float().mean(0)
    live_features = ((freq >= 0.01) & (freq <= 0.5)).nonzero(as_tuple=True)[0].tolist()
    for f in live_features:
        if f not in label_embeddings:
            base = np.array([0.0, 0.0, 1.0])
            label_embeddings[f] = base + rng.normal(scale=0.05, size=3)
            feature_labels[f] = f"F{f}"

    G = cooccurrence_graph(doc_acts, feature_labels=feature_labels)
    print(f"[validation] Graphe : {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes.")
    print(f"[validation] Paire injectée dans le graphe ? {G.has_edge(feat_a, feat_b)}")
    if G.has_edge(feat_a, feat_b):
        print(f"[validation] NPMI de la paire injectée : {G[feat_a][feat_b]['npmi']:.3f}")

    pairs = find_interesting_pairs(G, label_embeddings)
    found = any({p["feature_a"], p["feature_b"]} == {feat_a, feat_b} for p in pairs)
    rank = next((i for i, p in enumerate(pairs) if {p["feature_a"], p["feature_b"]} == {feat_a, feat_b}), None)

    print(f"\n[validation] {len(pairs)} paires 'intéressantes' détectées au total.")
    print(f"[validation] Paire injectée retrouvée : {found}"
          + (f" (rang {rank+1}/{len(pairs)})" if found else ""))

    results = {
        "n_docs": doc_acts.shape[0], "n_features": doc_acts.shape[1],
        "injected_pair": [feat_a, feat_b],
        "graph_n_nodes": G.number_of_nodes(), "graph_n_edges": G.number_of_edges(),
        "injected_pair_is_edge": G.has_edge(feat_a, feat_b),
        "injected_pair_npmi": G[feat_a][feat_b]["npmi"] if G.has_edge(feat_a, feat_b) else None,
        "n_interesting_pairs_found": len(pairs),
        "injected_pair_recovered": found,
        "injected_pair_rank": rank,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Écrit : {OUT_PATH}")


if __name__ == "__main__":
    main()
