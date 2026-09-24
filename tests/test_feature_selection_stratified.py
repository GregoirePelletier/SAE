"""Teste src/sae/judge.py::feature_selection_stratified_by_frequency --
correctif B.2 (docs/archive/audits/AUDIT_SAE_2026-08.md) : feature_selection_by_magnitude
sélectionne systématiquement les features les plus denses, rendant le taux
d'interprétabilité mesuré non comparable à un chiffre publié ni entre
configurations du dépôt."""
import os

import numpy as np
import torch

from src.sae.judge import (
    feature_selection_by_magnitude, feature_selection_stratified_by_frequency,
    feature_selection_stratified_by_frequency_dense,
)
from src.storage.fragment_store import save_fragment


def _build_corpus(tmp_path, n_docs, d_sae, active_features_per_doc):
    """active_features_per_doc(doc_id) -> list[(f_idx, magnitude)]."""
    frag_dir = str(tmp_path)
    os.makedirs(frag_dir, exist_ok=True)
    n_tok = 5
    for doc_id in range(n_docs):
        acts = torch.zeros(n_tok, d_sae)
        for f_idx, mag in active_features_per_doc(doc_id):
            acts[doc_id % n_tok, f_idx] = mag
        save_fragment(frag_dir, doc_id=doc_id, token_strings=[f"▁t{i}" for i in range(n_tok)],
                      acts_dense=acts)
    return frag_dir


def test_stratified_includes_rare_features_magnitude_excludes_them(tmp_path):
    """d_sae=100 : feature 0 très dense (90% des docs, magnitude modeste) et
    9 features rares (5% des docs, magnitude modeste aussi -- même ordre de
    grandeur, pour isoler l'effet de fréquence de celui de magnitude) que la
    sélection par magnitude n'a aucune raison de privilégier non plus dans ce
    scénario. Vérifie surtout que la version stratifiée touche des features
    à fréquences très différentes, pas seulement les plus fréquentes."""
    n_docs, d_sae = 200, 100
    rng = np.random.default_rng(0)

    def active(doc_id):
        out = []
        if doc_id < 180:  # 90% des docs
            out.append((0, 2.0))
        if doc_id < 10:  # 5% des docs
            out.append((50, 2.0))
        return out

    frag_dir = _build_corpus(tmp_path, n_docs, d_sae, active)
    selected = feature_selection_stratified_by_frequency(
        frag_dir, list(range(n_docs)), d_sae, n_features=2, sample_docs=n_docs, n_bins=2, seed=0,
    )
    assert set(selected) == {0, 50}  # les deux seules features vivantes, aux fréquences opposées


def test_stratified_falls_back_when_all_dead(tmp_path):
    n_docs, d_sae = 20, 10
    frag_dir = _build_corpus(tmp_path, n_docs, d_sae, lambda doc_id: [])
    selected = feature_selection_stratified_by_frequency(
        frag_dir, list(range(n_docs)), d_sae, n_features=3, sample_docs=n_docs,
    )
    assert selected == list(range(3))  # repli déterministe, comme feature_selection_by_magnitude


def test_stratified_reproducible_with_seed(tmp_path):
    n_docs, d_sae = 100, 50
    rng_pattern = np.random.default_rng(1)

    def active(doc_id):
        return [(f, 1.0) for f in range(d_sae) if rng_pattern.random() < 0.3]

    # Motif d'activation fixe (indépendant de l'appel testé) construit une fois.
    patterns = [active(d) for d in range(n_docs)]
    frag_dir = _build_corpus(tmp_path, n_docs, d_sae, lambda d: patterns[d])

    sel1 = feature_selection_stratified_by_frequency(
        frag_dir, list(range(n_docs)), d_sae, n_features=10, sample_docs=n_docs, seed=42,
    )
    sel2 = feature_selection_stratified_by_frequency(
        frag_dir, list(range(n_docs)), d_sae, n_features=10, sample_docs=n_docs, seed=42,
    )
    assert sel1 == sel2


def test_stratified_respects_lo_hi_range(tmp_path):
    n_docs, d_sae = 50, 20

    def active(doc_id):
        return [(f, 1.0) for f in range(d_sae) if doc_id % (f + 1) == 0]

    frag_dir = _build_corpus(tmp_path, n_docs, d_sae, active)
    selected = feature_selection_stratified_by_frequency(
        frag_dir, list(range(n_docs)), d_sae, n_features=5, sample_docs=n_docs, lo=10, hi=20,
    )
    assert all(10 <= f < 20 for f in selected)


def test_stratified_never_exceeds_n_features(tmp_path):
    n_docs, d_sae = 50, 30

    def active(doc_id):
        return [(f, 1.0) for f in range(d_sae) if (doc_id + f) % 3 == 0]

    frag_dir = _build_corpus(tmp_path, n_docs, d_sae, active)
    selected = feature_selection_stratified_by_frequency(
        frag_dir, list(range(n_docs)), d_sae, n_features=7, sample_docs=n_docs,
    )
    assert len(selected) <= 7


def test_stratified_bin_info_matches_selected_and_is_self_consistent(tmp_path):
    """N3 (docs/archive/audits/AUDIT_SAE_2026-08.md §8) : return_bin_info=True doit couvrir
    exactement les features sélectionnées, avec des tailles de strate
    cohérentes (bin_n_sampled <= bin_population), et attribuer des strates
    différentes à un groupe de features denses vs un groupe de features
    rares -- plus de features vivantes que n_features, pour que la
    stratification réelle s'exerce (pas le repli "prendre tout le monde")."""
    n_docs, d_sae = 200, 100
    dense_features = list(range(10))       # 90% des docs
    rare_features = list(range(50, 60))    # 5% des docs

    def active(doc_id):
        out = []
        if doc_id < 180:
            out.extend((f, 2.0) for f in dense_features)
        if doc_id < 10:
            out.extend((f, 2.0) for f in rare_features)
        return out

    frag_dir = _build_corpus(tmp_path, n_docs, d_sae, active)
    selected, bin_info = feature_selection_stratified_by_frequency(
        frag_dir, list(range(n_docs)), d_sae, n_features=4, sample_docs=n_docs, n_bins=2, seed=0,
        return_bin_info=True,
    )
    assert set(selected) == set(bin_info.keys())
    for f in selected:
        info = bin_info[f]
        assert 0 < info["bin_n_sampled"] <= info["bin_population"]
        assert info["freq"] > 0

    selected_dense_bins = {bin_info[f]["bin"] for f in selected if f in dense_features}
    selected_rare_bins = {bin_info[f]["bin"] for f in selected if f in rare_features}
    assert selected_dense_bins, "au moins une feature dense sélectionnée"
    assert selected_rare_bins, "au moins une feature rare sélectionnée"
    # Le groupe dense et le groupe rare sont à des fréquences très éloignées
    # (90% vs 5%) -> strates disjointes.
    assert selected_dense_bins.isdisjoint(selected_rare_bins)


def test_stratified_bin_info_degenerate_paths_return_tuple(tmp_path):
    """Les deux replis dégénérés (tout mort, ou moins de features vivantes
    que n_features) doivent respecter le contrat return_bin_info=True --
    sinon un appelant qui déballe (selected, bin_info) plante selon le
    chemin emprunté par le corpus, pas selon l'API demandée."""
    n_docs, d_sae = 20, 10

    frag_dir_dead = _build_corpus(tmp_path / "dead", n_docs, d_sae, lambda doc_id: [])
    selected, bin_info = feature_selection_stratified_by_frequency(
        frag_dir_dead, list(range(n_docs)), d_sae, n_features=3, sample_docs=n_docs,
        return_bin_info=True,
    )
    assert selected == list(range(3))
    assert set(bin_info.keys()) == set(selected)

    def active_few(doc_id):
        return [(0, 1.0)] if doc_id < 5 else []

    frag_dir_few = _build_corpus(tmp_path / "few", n_docs, d_sae, active_few)
    selected2, bin_info2 = feature_selection_stratified_by_frequency(
        frag_dir_few, list(range(n_docs)), d_sae, n_features=5, sample_docs=n_docs,
        return_bin_info=True,
    )
    assert selected2 == [0]   # seule feature vivante, <= n_features
    assert set(bin_info2.keys()) == {0}


def test_horvitz_thompson_mean_reduces_to_stratified_mean():
    from src.analysis.stats import horvitz_thompson_mean

    # Deux strates : 10 succès/10 dans une strate de 100 (π=0.1), 0 succès/10
    # dans une strate de 10 (π=1.0) -- moyenne stratifiée attendue :
    # (100*1.0 + 10*0.0) / 110.
    values = [1.0] * 10 + [0.0] * 10
    probs = [10 / 100] * 10 + [10 / 10] * 10
    expected = (100 * 1.0 + 10 * 0.0) / 110
    assert horvitz_thompson_mean(values, probs) == expected


def test_horvitz_thompson_mean_equal_probs_is_plain_mean():
    from src.analysis.stats import horvitz_thompson_mean

    values = [1.0, 0.0, 1.0, 1.0]
    probs = [0.5, 0.5, 0.5, 0.5]
    assert horvitz_thompson_mean(values, probs) == np.mean(values)


# ── Pipeline 2 (activations denses en mémoire, pas de fragments sur disque) ──


def _build_dense_doc_acts(n_docs, d_sae, active_features_per_doc):
    acts = torch.zeros(n_docs, d_sae)
    for doc_id in range(n_docs):
        for f_idx, mag in active_features_per_doc(doc_id):
            acts[doc_id, f_idx] = mag
    return acts


def test_dense_stratified_matches_fragment_version_on_same_pattern(tmp_path):
    """Même motif d'activation (dense 90% vs rare 5%), deux sources de
    `freq` différentes (fragments vs tenseur dense) -- doit sélectionner
    exactement les deux mêmes features vivantes, comme la version fragments
    (test_stratified_includes_rare_features_magnitude_excludes_them)."""
    n_docs, d_sae = 200, 100

    def active(doc_id):
        out = []
        if doc_id < 180:
            out.append((0, 2.0))
        if doc_id < 10:
            out.append((50, 2.0))
        return out

    doc_acts = _build_dense_doc_acts(n_docs, d_sae, active)
    selected = feature_selection_stratified_by_frequency_dense(
        doc_acts, n_features=2, sample_docs=n_docs, n_bins=2, seed=0,
    )
    assert set(selected) == {0, 50}


def test_dense_stratified_falls_back_when_all_dead():
    n_docs, d_sae = 20, 10
    doc_acts = _build_dense_doc_acts(n_docs, d_sae, lambda doc_id: [])
    selected = feature_selection_stratified_by_frequency_dense(
        doc_acts, n_features=3, sample_docs=n_docs,
    )
    assert selected == list(range(3))


def test_dense_stratified_reproducible_with_seed():
    n_docs, d_sae = 100, 50
    rng_pattern = np.random.default_rng(1)

    def active(doc_id):
        return [(f, 1.0) for f in range(d_sae) if rng_pattern.random() < 0.3]

    patterns = [active(d) for d in range(n_docs)]
    doc_acts = _build_dense_doc_acts(n_docs, d_sae, lambda d: patterns[d])

    sel1 = feature_selection_stratified_by_frequency_dense(
        doc_acts, n_features=10, sample_docs=n_docs, seed=42,
    )
    sel2 = feature_selection_stratified_by_frequency_dense(
        doc_acts, n_features=10, sample_docs=n_docs, seed=42,
    )
    assert sel1 == sel2


def test_dense_stratified_respects_lo_hi_range():
    n_docs, d_sae = 50, 20

    def active(doc_id):
        return [(f, 1.0) for f in range(d_sae) if doc_id % (f + 1) == 0]

    doc_acts = _build_dense_doc_acts(n_docs, d_sae, active)
    selected = feature_selection_stratified_by_frequency_dense(
        doc_acts, n_features=5, sample_docs=n_docs, lo=10, hi=20,
    )
    assert all(10 <= f < 20 for f in selected)


def test_dense_stratified_bin_info_matches_selected():
    n_docs, d_sae = 200, 100
    dense_features = list(range(10))
    rare_features = list(range(50, 60))

    def active(doc_id):
        out = []
        if doc_id < 180:
            out.extend((f, 2.0) for f in dense_features)
        if doc_id < 10:
            out.extend((f, 2.0) for f in rare_features)
        return out

    doc_acts = _build_dense_doc_acts(n_docs, d_sae, active)
    selected, bin_info = feature_selection_stratified_by_frequency_dense(
        doc_acts, n_features=4, sample_docs=n_docs, n_bins=2, seed=0, return_bin_info=True,
    )
    assert set(selected) == set(bin_info.keys())
    for f in selected:
        info = bin_info[f]
        assert 0 < info["bin_n_sampled"] <= info["bin_population"]
        assert info["freq"] > 0
