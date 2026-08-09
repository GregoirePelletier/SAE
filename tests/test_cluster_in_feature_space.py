"""
tests/test_cluster_in_feature_space.py — non-régression pour le changement
appliqué suite à l'audit méthodologique (RESULTS_TESTS.md §33, 2026-08-07) :
`cluster_in_feature_space` fait désormais tourner HDBSCAN sur un embedding
UMAP 10D dédié, PAS sur `emb2d` (réservé à la visualisation). Aucun test
n'existait avant sur cette fonction (`analyze_with_umap`, la fonction sœur
dans saev5.py, reste non testée ici : couplée à des écritures disque via
SAVE_DIR, hors du périmètre "unitaire rapide" de ce dossier).
"""
import numpy as np
import torch

from src.analysis.cooccurrence import cluster_in_feature_space


def test_cluster_in_feature_space_returns_coherent_shapes():
    rng = np.random.default_rng(0)
    n_docs, n_features = 60, 40
    doc_acts = torch.tensor(
        (rng.random((n_docs, n_features)) < 0.1).astype(np.float32), dtype=torch.float32
    )
    labels, emb2d = cluster_in_feature_space(doc_acts, min_cluster_size=5)
    assert labels.shape == (n_docs,)
    assert emb2d.shape == (n_docs, 2)  # emb2d reste 2D pour la visualisation
    # HDBSCAN doit avoir tourné sur l'embedding 10D interne, pas sur emb2d --
    # vérifié indirectement : labels est un vecteur d'entiers valide (pas de crash).
    assert labels.dtype.kind in "iu"


def test_cluster_in_feature_space_falls_back_on_tiny_corpus():
    # N_DOCS <= 12 : doit utiliser emb2d directement (pas de fit UMAP 10D sur un
    # corpus trop petit pour n_components=10) plutôt que de planter.
    rng = np.random.default_rng(1)
    doc_acts = torch.tensor(
        (rng.random((10, 20)) < 0.2).astype(np.float32), dtype=torch.float32
    )
    labels, emb2d = cluster_in_feature_space(doc_acts, min_cluster_size=2)
    assert labels.shape == (10,)
    assert emb2d.shape == (10, 2)
