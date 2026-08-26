"""Teste src/analysis/clustering_llm.py (App F.1, §4.3/K.3) : parsing des
réponses LLM (mock de _batched_generate, CPU uniquement), union de latents
par mots-clés, et conductance/z-score sur un nuage de points synthétique à
2 blobs bien séparés (résultat attendu connu : conductance quasi nulle,
z-score très négatif pour les vrais clusters -- "lower = tighter", §4.3)."""
from unittest.mock import patch

import numpy as np

from src.analysis.clustering_llm import (
    compute_cluster_accuracy,
    conductance,
    conductance_zscore,
    generate_cluster_labels,
    generate_keywords,
    select_latents_union,
    _build_knn_graph,
)


def test_generate_keywords_splits_lines():
    def _fake(model, tokenizer, list_of_messages, max_new_tokens, batch_size=16):
        return ["sports\npolitics\n- technology\n"]

    with patch("src.analysis.clustering_llm._batched_generate", _fake):
        kws = generate_keywords(model=None, tokenizer=None, query="cluster by topic")
    assert kws == ["sports", "politics", "technology"]


def test_select_latents_union_deduplicates_across_keywords():
    def fake_select_fn(query, feature_labels, top_k):
        return {"sports": [1, 2, 3], "politics": [3, 4]}[query]

    union = select_latents_union(["sports", "politics"], feature_labels={}, select_fn=fake_select_fn, top_k=3)
    assert union == [1, 2, 3, 4]


def test_generate_cluster_labels_parses_numbered_format():
    clusters = [
        {"features": ["billing"], "examples": ["invoice text"]},
        {"features": ["outage"], "examples": ["power cut text"]},
    ]

    def _fake(model, tokenizer, list_of_messages, max_new_tokens, batch_size=16):
        return ["Cluster 0: [Billing complaints]\nCluster 1: [Power outages]"]

    with patch("src.analysis.clustering_llm._batched_generate", _fake):
        labels = generate_cluster_labels(model=None, tokenizer=None, clusters=clusters)
    assert labels == ["Billing complaints", "Power outages"]


def test_generate_cluster_labels_missing_cluster_defaults_to_unclear():
    clusters = [{"features": [], "examples": []}, {"features": [], "examples": []}]

    def _fake(model, tokenizer, list_of_messages, max_new_tokens, batch_size=16):
        return ["Cluster 0: [Some label]"]  # cluster 1 absent de la réponse

    with patch("src.analysis.clustering_llm._batched_generate", _fake):
        labels = generate_cluster_labels(model=None, tokenizer=None, clusters=clusters)
    assert labels == ["Some label", "UNCLEAR"]


def test_compute_cluster_accuracy_counts_matches_per_original_cluster():
    texts = ["t0", "t1", "t2", "t3"]
    cluster_ids = [0, 0, 1, 1]
    descriptions = {0: "billing", 1: "outage"}

    # 3 des 4 réassignations correctes (t1 mal réassigné -> cluster 1)
    def _fake(model, tokenizer, list_of_messages, max_new_tokens, batch_size=16):
        return ["0", "1", "1", "1"]

    with patch("src.analysis.clustering_llm._batched_generate", _fake):
        acc = compute_cluster_accuracy(model=None, tokenizer=None, texts=texts, cluster_ids=cluster_ids,
                                        cluster_descriptions=descriptions)
    assert acc[0] == 0.5   # 1/2 texte du cluster 0 réassigné à 0
    assert acc[1] == 1.0   # 2/2 textes du cluster 1 réassignés à 1


def _two_blob_embeddings(seed=0, n_per_blob=20, gap=10.0):
    rng = np.random.default_rng(seed)
    blob1 = rng.normal(loc=0.0, scale=0.1, size=(n_per_blob, 2))
    blob2 = rng.normal(loc=gap, scale=0.1, size=(n_per_blob, 2))
    embeddings = np.vstack([blob1, blob2])
    labels = np.array([0] * n_per_blob + [1] * n_per_blob)
    return embeddings, labels


def test_conductance_near_zero_for_well_separated_blob():
    embeddings, labels = _two_blob_embeddings()
    graph = _build_knn_graph(embeddings, k_neighbors=5)
    assert conductance(graph, labels == 0) < 0.05


def test_conductance_zscore_strongly_negative_for_real_clusters():
    """"lower = tighter" (§4.3) : les vrais clusters (blobs bien séparés)
    doivent avoir une conductance largement sous la moyenne d'échantillons
    aléatoires de même taille -- z-score fortement négatif."""
    embeddings, labels = _two_blob_embeddings()
    z = conductance_zscore(embeddings, labels, k_neighbors=5, n_random=50, seed=0)
    assert set(z.keys()) == {0, 1}
    assert z[0] < -3.0
    assert z[1] < -3.0


def test_conductance_zscore_excludes_hdbscan_noise_label():
    embeddings, labels = _two_blob_embeddings()
    labels_with_noise = labels.copy()
    labels_with_noise[0] = -1  # bruit HDBSCAN
    z = conductance_zscore(embeddings, labels_with_noise, k_neighbors=5, n_random=20, seed=0)
    assert -1 not in z


def test_conductance_degenerate_full_or_empty_mask_is_zero():
    embeddings, _ = _two_blob_embeddings()
    graph = _build_knn_graph(embeddings, k_neighbors=5)
    n = embeddings.shape[0]
    assert conductance(graph, np.ones(n, dtype=bool)) == 0.0
    assert conductance(graph, np.zeros(n, dtype=bool)) == 0.0
