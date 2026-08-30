"""Smoke-teste src/visualization/dashboard.py (Streamlit) : chaque page,
sur plusieurs runs réels, ne doit jamais lever d'exception. Persiste en test
ce qui n'était vérifié jusqu'ici qu'à la main (`streamlit.testing.v1.AppTest`,
AUDIT_SAE_2026-08.md) -- CPU-only, lit uniquement des artefacts déjà sur
disque, aucun modèle chargé. Ne vérifie PAS que les chiffres affichés sont
corrects (cf. test_dashboard_judge_sources.py pour ça), seulement que rien ne
plante."""
import glob
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PAGES = [
    "Vue d'ensemble", "UMAP", "Features", "Diagnostics d'entraînement", "Diffing",
    "Recherche", "Urgence/Robustesse", "Explication (fidélité/plausibilité)",
    "Clustering & Corrélations", "Sweeps (échelle & layer)", "Rapport consolidé",
    "Comparaison mail original / augmenté", "Audit méthodologique (archive)",
]


def _available_run_dirs(limit=3):
    dirs = sorted(glob.glob(os.path.join(REPO_ROOT, "results_*")))
    names = [os.path.basename(d) for d in dirs
             if os.path.isdir(d) and os.path.basename(d) != "results_diagnostics"]
    return names[:limit]


@pytest.mark.skipif(not _available_run_dirs(), reason="aucun results_*/ sur cette machine (hors cluster)")
def test_dashboard_all_pages_no_exception_on_default_run():
    at = AppTest.from_file(
        os.path.join(REPO_ROOT, "src", "visualization", "dashboard.py"), default_timeout=60,
    )
    at.run()
    assert not at.exception, f"exception au chargement initial : {at.exception}"
    for page in PAGES:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"exception sur la page {page!r} : {at.exception}"


@pytest.mark.skipif(len(_available_run_dirs()) < 2, reason="moins de 2 results_*/ sur cette machine")
def test_dashboard_features_page_no_exception_across_runs():
    """La page Features est celle qui varie le plus selon le run (sélecteur
    de source de labels juge, cf. _judge_label_sources) -- vérifiée
    séparément sur plusieurs runs plutôt qu'une seule fois sur le défaut."""
    for run_dir in _available_run_dirs():
        at = AppTest.from_file(
            os.path.join(REPO_ROOT, "src", "visualization", "dashboard.py"), default_timeout=60,
        )
        at.run()
        at.sidebar.selectbox[0].set_value(run_dir).run()
        at.sidebar.radio[0].set_value("Features").run()
        assert not at.exception, f"run={run_dir} : exception sur Features : {at.exception}"
