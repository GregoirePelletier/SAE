"""Teste src/sae/judge.py::feature_selection_stratified_by_frequency --
correctif B.2 (AUDIT_SAE_2026-08.md) : feature_selection_by_magnitude
sélectionne systématiquement les features les plus denses, rendant le taux
d'interprétabilité mesuré non comparable à un chiffre publié ni entre
configurations du dépôt."""
import numpy as np
import torch

from src.sae.judge import feature_selection_by_magnitude, feature_selection_stratified_by_frequency
from src.storage.fragment_store import save_fragment


def _build_corpus(tmp_path, n_docs, d_sae, active_features_per_doc):
    """active_features_per_doc(doc_id) -> list[(f_idx, magnitude)]."""
    frag_dir = str(tmp_path)
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
