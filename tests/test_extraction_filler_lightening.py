"""Teste src/data/preparation.py::is_filler_document -- allègement de
l'extraction P1 côté filler (AUDIT_SAE_2026-08.md §2.2/§2.5) : un document
filler ne doit ni être encodé par le core SAE ni écrit en fragment, seul son
résidu brut compte pour le réservoir. Complète test_reencode_skips_filler.py
(qui teste l'exclusion filler du ré-encodage) côté extraction."""
from src.data.preparation import is_filler_document


def test_train_documents_are_never_filler():
    n_train, n_filler = 10, 5
    for i in range(n_train):
        assert is_filler_document(i, n_train, n_filler) is False


def test_filler_range_is_correctly_flagged():
    n_train, n_filler = 10, 5
    for i in range(n_train, n_train + n_filler):
        assert is_filler_document(i, n_train, n_filler) is True


def test_documents_after_filler_are_never_filler():
    """Test/diff (après le filler dans all_texts) ne doivent jamais être
    signalés comme filler."""
    n_train, n_filler = 10, 5
    for i in range(n_train + n_filler, n_train + n_filler + 20):
        assert is_filler_document(i, n_train, n_filler) is False


def test_no_filler_configured_never_flags_anything():
    """N_VOLUME_FILLER_TARGET_CHUNKS=0 (défaut) : aucun document n'est filler,
    comportement d'extraction identique à avant ce correctif."""
    n_train, n_filler = 10, 0
    for i in range(30):
        assert is_filler_document(i, n_train, n_filler) is False


def test_boundaries_are_exact():
    n_train, n_filler = 10, 5
    assert is_filler_document(n_train - 1, n_train, n_filler) is False  # dernier train
    assert is_filler_document(n_train, n_train, n_filler) is True       # premier filler
    assert is_filler_document(n_train + n_filler - 1, n_train, n_filler) is True   # dernier filler
    assert is_filler_document(n_train + n_filler, n_train, n_filler) is False      # premier test
