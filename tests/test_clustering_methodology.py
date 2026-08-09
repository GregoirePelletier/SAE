import numpy as np

from scripts.clustering_methodology_audit import (
    compute_metrics,
    config_b_raw_cosine,
    fit_pca,
    fit_umap,
    run_hdbscan,
)


def _three_blobs(seed: int = 0) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    centers = np.eye(3, 30) * 8.0  # 3 well-separated one-hot-ish centers in 30D
    X = np.concatenate([rng.normal(loc=c, scale=0.5, size=(20, 30)) for c in centers])
    labels = ["a"] * 20 + ["b"] * 20 + ["c"] * 20
    return X, labels


def test_run_hdbscan_raw_cosine_finds_the_blobs():
    X, labels = _three_blobs()
    hdb_labels, validity = config_b_raw_cosine(X, min_cluster_size=5)
    assert hdb_labels.shape == (60,)
    assert 0.0 <= validity <= 1.0
    m = compute_metrics(hdb_labels, X, labels, metric="cosine")
    assert m["n_clusters"] >= 2
    assert m["ami_vs_email_axes"] > 0.5  # bien aligné sur les 3 groupes connus


def test_fit_umap_then_hdbscan_returns_coherent_types():
    X, labels = _three_blobs()
    coords = fit_umap(X, n_components=2, seed=0)
    assert coords.shape == (60, 2)
    hdb_labels, validity = run_hdbscan(coords, min_cluster_size=5, metric="euclidean")
    assert hdb_labels.shape == (60,)
    assert isinstance(validity, float)
    m = compute_metrics(hdb_labels, coords, labels, metric="euclidean")
    assert set(m.keys()) == {"noise_frac", "n_clusters", "ami_vs_email_axes", "silhouette"}
    assert 0.0 <= m["noise_frac"] <= 1.0


def test_fit_pca_then_hdbscan_is_deterministic_and_finds_the_blobs():
    X, labels = _three_blobs()
    emb1, var1 = fit_pca(X, n_components=2)
    emb2, var2 = fit_pca(X, n_components=2)
    assert np.allclose(emb1, emb2)  # PCA doit être déterministe (pas de seed à fixer)
    assert 0.0 <= var1 <= 1.0
    hdb_labels, validity = run_hdbscan(emb1, min_cluster_size=5, metric="euclidean")
    m = compute_metrics(hdb_labels, emb1, labels, metric="euclidean")
    assert m["n_clusters"] >= 2
    assert m["ami_vs_email_axes"] > 0.5
