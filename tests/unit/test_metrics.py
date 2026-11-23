import pytest
from app.evaluation.metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
    calculate_ndcg_at_k,
    calculate_faithfulness,
    calculate_answer_correctness,
    calculate_context_relevance,
)


def test_precision_at_k():
    retrieved = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
    expected = ["doc1.pdf", "doc3.pdf"]
    # Top 2 has doc1 (hit) and doc2 (miss) -> 1/2 = 0.5
    assert calculate_precision_at_k(retrieved, expected, k=2) == 0.5
    # Top 3 has doc1 (hit), doc2 (miss), doc3 (hit) -> 2/3 = 0.6667
    assert calculate_precision_at_k(retrieved, expected, k=3) == 0.6667


def test_recall_at_k():
    retrieved = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
    expected = ["doc1.pdf", "doc3.pdf"]
    # Top 1 captures doc1 out of 2 expected -> 1/2 = 0.5
    assert calculate_recall_at_k(retrieved, expected, k=1) == 0.5
    # Top 3 captures both doc1 and doc3 -> 2/2 = 1.0
    assert calculate_recall_at_k(retrieved, expected, k=3) == 1.0


def test_reciprocal_rank():
    # First hit is at index 0 (Rank 1) -> RR = 1.0
    assert calculate_reciprocal_rank(["docA.pdf", "docB.pdf"], ["docA.pdf"]) == 1.0
    # First hit is at index 1 (Rank 2) -> RR = 0.5
    assert calculate_reciprocal_rank(["docB.pdf", "docA.pdf"], ["docA.pdf"]) == 0.5
    # Hit not present -> RR = 0.0
    assert calculate_reciprocal_rank(["docB.pdf", "docC.pdf"], ["docA.pdf"]) == 0.0


def test_ndcg_at_k():
    # Perfect ranking: expected doc at rank 1
    assert calculate_ndcg_at_k(["doc1.pdf", "doc2.pdf"], ["doc1.pdf"], k=2) == 1.0


def test_faithfulness_fully_grounded():
    context = "Continuous integration is a DevOps practice where developers regularly commit code changes to a shared repository."
    answer = "Developers commit code changes to a shared repository in continuous integration."
    score, unsupported = calculate_faithfulness(answer, context)
    assert score == 1.0
    assert len(unsupported) == 0


def test_faithfulness_detects_hallucination():
    context = "Continuous integration is a software practice where code is merged frequently."
    answer = "Continuous integration was invented by Albert Einstein in 1905 on Mars."
    score, unsupported = calculate_faithfulness(answer, context)
    assert score < 1.0
    assert len(unsupported) > 0
    assert "Albert Einstein" in unsupported[0]


def test_faithfulness_handles_refusal():
    context = "DevOps life cycle details."
    answer = "I do not have sufficient information in the provided documents to answer this question."
    score, unsupported = calculate_faithfulness(answer, context)
    assert score == 1.0
    assert len(unsupported) == 0


def test_answer_correctness():
    generated = "Continuous integration automates code builds and automated testing."
    truth = "Continuous integration automates code builds and automated testing."
    assert calculate_answer_correctness(generated, truth) == 1.0
