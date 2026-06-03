import pytest
from src.evaluation.scorer import score_relevance, score_faithfulness


def test_score_relevance_empty():
    assert score_relevance("question", []) == 0.0


def test_score_relevance_basic():
    chunks = [{"score": 0.8}, {"score": 0.6}]
    assert score_relevance("question", chunks) == 0.7


def test_score_faithfulness_empty():
    assert score_faithfulness("answer", []) == 0.0


def test_score_faithfulness_basic():
    chunks = [{"content": "the quick brown fox"}]
    answer = "the fox"
    result = score_faithfulness(answer, chunks)
    assert result == 1.0
