from scripts.interesting_pairs_synthetic_validation import build_synthetic_corpus
from src.analysis.cooccurrence import cooccurrence_graph, find_interesting_pairs
import numpy as np


def test_find_interesting_pairs_recovers_injected_correlation():
    doc_acts, feat_a, feat_b = build_synthetic_corpus(n_docs=1000, n_features=200)
    label_embeddings = {feat_a: np.array([1.0, 0.0, 0.0]), feat_b: np.array([0.0, 1.0, 0.0])}
    feature_labels = {feat_a: "a", feat_b: "b"}
    G = cooccurrence_graph(doc_acts, feature_labels=feature_labels)
    assert G.has_edge(feat_a, feat_b)
    pairs = find_interesting_pairs(G, label_embeddings)
    assert any({p["feature_a"], p["feature_b"]} == {feat_a, feat_b} for p in pairs)
