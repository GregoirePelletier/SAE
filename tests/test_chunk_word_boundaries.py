"""Teste src/data/preparation.py::_chunk_on_word_boundaries -- correctif
B.10 (AUDIT_SAE_2026-08.md) : le chunking par `txt[i:i+chunk_length]`
coupait indifféremment mots et phrases, une distribution de tokens qui
n'existe dans aucun usage réel -- en particulier pour le filler FineWeb2
qui domine le volume d'entraînement du SAE résiduel."""
from src.data.preparation import _chunk_on_word_boundaries


def test_no_word_is_split_across_chunks():
    txt = " ".join(f"mot{i}" for i in range(200))  # mots courts, nombreux
    chunks = _chunk_on_word_boundaries(txt, chunk_length=50, max_chunks=100)
    reconstructed_words = " ".join(chunks).split(" ")
    original_words = txt.split(" ")
    # Chaque mot du texte original apparaît intact (pas tronqué) dans la
    # reconstruction, dans le même ordre.
    assert reconstructed_words == original_words[:len(reconstructed_words)]
    for c in chunks:
        for w in c.split(" "):
            assert w in original_words


def test_chunks_stay_close_to_target_length():
    txt = " ".join(["motcourt"] * 300)
    chunks = _chunk_on_word_boundaries(txt, chunk_length=100, max_chunks=50)
    for c in chunks[:-1]:  # le dernier chunk peut être plus court
        assert len(c) <= 100
        assert len(c) > 50  # pas dégénérément petit non plus


def test_respects_max_chunks():
    txt = " ".join(["mot"] * 1000)
    chunks = _chunk_on_word_boundaries(txt, chunk_length=20, max_chunks=3)
    assert len(chunks) <= 3


def test_single_word_longer_than_chunk_length_becomes_its_own_chunk():
    long_word = "a" * 2000
    txt = f"debut {long_word} fin"
    chunks = _chunk_on_word_boundaries(txt, chunk_length=100, max_chunks=10)
    assert any(long_word in c for c in chunks)


def test_empty_text_returns_empty_list():
    assert _chunk_on_word_boundaries("", chunk_length=100, max_chunks=10) == []


def test_short_text_returns_single_chunk():
    chunks = _chunk_on_word_boundaries("un texte court", chunk_length=1024, max_chunks=6)
    assert chunks == ["un texte court"]
