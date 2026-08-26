"""Teste src/analysis/retrieval_rerank.py -- reranking LLM pointwise du
top-k (App. G, "second stage retrieval"). Mocke _batched_generate (CPU
uniquement, pas de modèle réel)."""
from unittest.mock import MagicMock, patch

from src.analysis.retrieval_rerank import llm_rerank, _parse_relevance_score


def test_parse_relevance_score_valid_json():
    assert _parse_relevance_score('{"relevance": 7}') == 7.0


def test_parse_relevance_score_clamps_out_of_range():
    assert _parse_relevance_score('{"relevance": 15}') == 10.0
    assert _parse_relevance_score('{"relevance": -3}') == 0.0


def test_parse_relevance_score_malformed_falls_back_to_zero():
    assert _parse_relevance_score("not json at all") == 0.0
    assert _parse_relevance_score("") == 0.0


def test_llm_rerank_reorders_top_k_by_score_descending():
    doc_texts = {i: f"doc{i}" for i in range(5)}
    ranked = [0, 1, 2, 3, 4]
    # Scores en désordre par rapport au classement de base -- le rerank doit
    # produire l'ordre 2 (score 9), 4 (score 8), 0 (score 5), 1 (score 5), 3 (score 1),
    # avec 0/1 à égalité départagés par l'ordre de base (0 avant 1).
    fake_scores = {0: 5, 1: 5, 2: 9, 3: 1, 4: 8}
    responses = [f'{{"relevance": {fake_scores[d]}}}' for d in ranked]

    with patch("src.analysis.retrieval_rerank._batched_generate", return_value=responses):
        result = llm_rerank(MagicMock(), object(), "query", ranked, doc_texts, top_k=5)

    assert result == [2, 4, 0, 1, 3]


def test_llm_rerank_only_touches_top_k_leaves_tail_unchanged():
    doc_texts = {i: f"doc{i}" for i in range(10)}
    ranked = list(range(10))
    responses = ['{"relevance": 0}'] * 3   # top_k=3, réponses sans effet sur l'ordre (égalités)

    with patch("src.analysis.retrieval_rerank._batched_generate", return_value=responses) as mock_gen:
        result = llm_rerank(MagicMock(), object(), "query", ranked, doc_texts, top_k=3)

    assert result[3:] == ranked[3:]   # queue inchangée
    messages = mock_gen.call_args[0][2]   # seulement les 3 premiers documents formatés en prompt
    assert len(messages) == 3
    assert "doc0" in messages[0][0]["content"]
    assert "doc2" in messages[2][0]["content"]


def test_llm_rerank_empty_ranking_returns_empty():
    assert llm_rerank(MagicMock(), object(), "query", [], {}, top_k=50) == []


def test_llm_rerank_shorter_than_top_k_reranks_everything():
    doc_texts = {i: f"doc{i}" for i in range(3)}
    ranked = [0, 1, 2]
    responses = ['{"relevance": 1}', '{"relevance": 9}', '{"relevance": 5}']

    with patch("src.analysis.retrieval_rerank._batched_generate", return_value=responses):
        result = llm_rerank(MagicMock(), object(), "query", ranked, doc_texts, top_k=50)

    assert result == [1, 2, 0]
