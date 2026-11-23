import pytest
from app.rag.pipeline import RAGPipeline
from app.rag.generator import RAGGenerator


@pytest.fixture(scope="module")
def rag_pipeline():
    return RAGPipeline(retrieval_k=4, rerank_k=2, reranker_provider="local", llm_provider="mock")


def test_rag_pipeline_functional_query(rag_pipeline):
    query = "What is continuous integration and how is code merged?"
    result = rag_pipeline.run(query)

    assert result.has_sufficient_context is True
    assert len(result.initial_candidates) > 0
    assert len(result.reranked_chunks) == 2
    assert len(result.sources) > 0
    assert "continuous integration" in result.answer.lower()


def test_generator_refusal_on_empty_context():
    generator = RAGGenerator(provider="mock")
    response = generator.generate("What is the speed of light?", chunks=[])

    assert response.has_sufficient_context is False
    assert "not have sufficient information" in response.answer.lower()
    assert len(response.sources) == 0
