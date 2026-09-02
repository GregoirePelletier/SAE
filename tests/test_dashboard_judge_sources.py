"""Teste src/visualization/dashboard.py::_judge_label_sources -- avant ce
correctif (AUDIT_SAE_2026-08.md §9), la page Features affichait toujours le
cache p1_judge_labels_extended.json sans dire de quel juge il vient, alors
que Gemma et Qwen coexistent maintenant dans le même SAVE_DIR (p1_judge_
model_separation_*.json, b1_stratified_mixte_qwen_rejudge_*.json,
b2_stratified_selection_rejudge.json). Ce module lit UNIQUEMENT des fichiers
déjà sur disque (aucun modèle, CPU-only)."""
import json
import os

import pytest

pytest.importorskip("streamlit")
from src.visualization.dashboard import _judge_label_sources


def _write(cache_dir, name, content):
    with open(os.path.join(cache_dir, name), "w", encoding="utf-8") as f:
        json.dump(content, f)


def test_flat_cache_without_sidecar_flags_judge_as_unrecorded(tmp_path):
    run_dir = tmp_path / "results_test_run"
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True)
    _write(cache_dir, "p1_judge_labels_extended.json", {"16384": {"interp_score": 1}})

    sources = _judge_label_sources(str(run_dir), "p1")
    assert len(sources) == 1
    (key,) = sources.keys()
    assert "non enregistré" in key
    assert sources[key] == {"16384": {"interp_score": 1}}


def test_flat_cache_with_sidecar_shows_recorded_judge(tmp_path):
    run_dir = tmp_path / "results_test_run"
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True)
    _write(cache_dir, "p1_judge_labels_extended.json", {"16384": {"interp_score": 1}})
    _write(cache_dir, "p1_judge_labels_extended.json.meta.json",
           {"judge_model_id": "/home/h21486/SAE/models/Qwen3.8-27B",
            "feature_selection_method": "stratified", "doc_groups_dedup": True})

    sources = _judge_label_sources(str(run_dir), "p1")
    (key,) = sources.keys()
    assert "Qwen3.8-27B" in key
    assert "stratified" in key


def test_discovers_judge_separation_comparison_file(tmp_path):
    run_dir = tmp_path / "results_test_run"
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True)
    _write(cache_dir, "p1_judge_labels_extended.json", {"16384": {"interp_score": 0}})
    _write(cache_dir, "p1_judge_model_separation_Qwen3.8-27B_seed42.json", {
        "summary": {"judge_alternative": "/home/h21486/SAE/models/Qwen3.8-27B",
                     "interp_rate_alternative": 0.787},
        "alt_per_feature": {"16384": {"interp_score": 1}},
    })

    sources = _judge_label_sources(str(run_dir), "p1")
    assert len(sources) == 2
    qwen_key = next(k for k in sources if "Qwen3.8-27B" in k and "comparaison" in k)
    assert sources[qwen_key] == {"16384": {"interp_score": 1}}


def test_discovers_b2_stratified_rejudge_file(tmp_path):
    run_dir = tmp_path / "results_test_run"
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True)
    _write(cache_dir, "b2_stratified_selection_rejudge.json", {
        "selected_features": [1, 2, 3],
        "results": {"1": {"interp_score": 1}},
        "bin_info": {},
    })

    sources = _judge_label_sources(str(run_dir), "p1")
    (key,) = sources.keys()
    assert "b2_stratified_selection_rejudge.json" in key
    assert sources[key] == {"1": {"interp_score": 1}}


def test_p2_prefix_ignores_p1_specific_files(tmp_path):
    """Les fichiers de comparaison juge (p1_judge_model_separation_*, b1_*,
    b2_*) sont spécifiques à Pipeline 1 -- l'onglet P2 ne doit trouver que
    p2_feature_labels.json, pas les hériter par accident."""
    run_dir = tmp_path / "results_test_run"
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True)
    _write(cache_dir, "p1_judge_model_separation_Qwen3.8-27B_seed42.json", {
        "summary": {"judge_alternative": "Qwen3.8-27B"}, "alt_per_feature": {"1": {}},
    })
    _write(cache_dir, "p2_feature_labels.json", {"1": {"interp_score": 1}})

    sources = _judge_label_sources(str(run_dir), "p2")
    assert len(sources) == 1
    (key,) = sources.keys()
    assert "p2_feature_labels.json" in key


def test_no_cache_files_returns_empty(tmp_path):
    run_dir = tmp_path / "results_test_run"
    (run_dir / "cache").mkdir(parents=True)
    assert _judge_label_sources(str(run_dir), "p1") == {}
    assert _judge_label_sources(str(run_dir), "p2") == {}


def test_contaminated_negative_source_tagged_and_sorted_last(tmp_path):
    """RESULTS_TESTS.md §115/§117 : un cache dont le négatif odd-one-out est
    majoritairement trivial (≤3 mots) prédate le correctif profond et donne un taux non
    comparable au chiffre de référence du rapport (65,7%, R0/§119) -- il ne doit ni
    apparaître en premier (sélection par défaut du dashboard) ni se présenter comme un
    taux ordinaire dans le sélecteur."""
    run_dir = tmp_path / "results_test_run"
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True)
    # Cache plat propre (négatifs riches en contexte, post-correctif).
    _write(cache_dir, "p1_judge_labels_extended.json", {
        str(i): {"interp_score": 1, "neg_example": "un négatif avec largement assez de mots de contexte gauche pour ne pas être trivial"}
        for i in range(10)
    })
    # Source alternative contaminée (négatifs à 1-2 mots, comme R0 avant §117).
    _write(cache_dir, "p1_judge_model_separation_Qwen3.8-27B_seed42.json", {
        "summary": {"judge_alternative": "Qwen3.8-27B"},
        "alt_per_feature": {str(i): {"interp_score": 1, "neg_example": "Bonjour,"} for i in range(10)},
    })

    sources = _judge_label_sources(str(run_dir), "p1")
    keys = list(sources.keys())
    assert len(keys) == 2
    assert "p1_judge_labels_extended.json" in keys[0]
    assert "⚠" not in keys[0]
    assert "p1_judge_model_separation" in keys[1]
    assert "⚠ négatif non corrigé" in keys[1]
