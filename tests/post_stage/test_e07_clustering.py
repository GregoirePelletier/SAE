"""Tests CPU rapides pour scripts/post_stage/e07_clustering.py et le doute
souleve par le plan (§12.1) sur select_latents_by_similarity : "corriger le
mapping labels/identifiants... test avec un JSON volontairement non trie."
`_embed_bge_m3` est mocke (aucun poids HuggingFace charge)."""
from unittest.mock import patch

import numpy as np
import torch

from scripts.post_stage.e07_clustering import _cluster_from_binarized
from src.sae.saev5 import select_latents_by_similarity


def test_select_latents_mapping_survives_unsorted_dict():
    """dict volontairement non trie ET dont les cles ne sont pas dans l'ordre
    d'insertion des labels attendus -- si f_idx/label se desynchronisaient
    (le doute souleve par le plan), le feature id renvoye en tete ne
    correspondrait pas au label le plus proche de la requete."""
    feature_labels = {103: "ton calme et neutre", 7: "urgence immediate", 55: "colere du client"}

    # embeddings deterministes : la requete matche EXACTEMENT le label de la
    # feature 7 ("urgence immediate"), partiellement celui de 55, pas du tout
    # celui de 103 -- l'ordre d'iteration du dict place 103 en PREMIER, pas 7.
    fixed = {
        "urgence": np.array([1.0, 0.0]),
        "ton calme et neutre": np.array([0.0, 1.0]),
        "urgence immediate": np.array([1.0, 0.0]),
        "colere du client": np.array([0.7071, 0.7071]),
    }

    def _fake_embed(texts, batch_size=64):
        return torch.tensor(np.stack([fixed[t] for t in texts]), dtype=torch.float32)

    with patch("src.sae.saev5._embed_bge_m3", side_effect=_fake_embed):
        ranked = select_latents_by_similarity("urgence", feature_labels, top_k=3)

    assert ranked[0] == 7, (
        f"attendu feature 7 ('urgence immediate', similarite parfaite) en tete, obtenu {ranked} -- "
        "le mapping f_idx/label ne survit pas a un dict non trie."
    )
    assert ranked[1] == 55  # similarite partielle
    assert 103 not in ranked or ranked.index(103) == len(ranked) - 1  # similarite nulle, en dernier si presente


def test_cluster_from_binarized_marks_zero_rows_as_no_signal():
    binarized = np.array([
        [1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1],
    ], dtype=np.float64)
    labels = _cluster_from_binarized(binarized)
    assert labels[2] == -1  # ligne nulle -> hors axe / sans signal
    assert (labels[np.arange(len(labels)) != 2] != -1).all()


def test_cluster_from_binarized_returns_all_no_signal_below_min_cluster_size():
    binarized = np.zeros((3, 5), dtype=np.float64)
    binarized[0, 0] = 1  # un seul document avec signal, < N_CLUSTERS
    labels = _cluster_from_binarized(binarized)
    assert (labels == -1).all()
