import os
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

from app.rag.retriever import RAGRetriever, RetrievedChunk
from app.rag.reranker import get_reranker, BaseReranker, RerankedChunk
from app.rag.generator import RAGGenerator, RAGResponse

load_dotenv()


@dataclass
class RAGPipelineResult:
    """Complete structured output from an end-to-end RAG query execution."""
    query: str
    answer: str
    has_sufficient_context: bool
    sources: List[Dict[str, Any]]
    initial_candidates: List[RetrievedChunk]
    reranked_chunks: List[RerankedChunk]
    model_used: str
    reranker_used: str
    total_latency_seconds: float


class RAGPipeline:
    """
    Unified end-to-end RAG Pipeline (System A: System Under Test).
    Orchestrates candidate retrieval, cross-encoder reranking, and grounded answer generation.
    """

    def __init__(
        self,
        retrieval_k: int = 6,
        rerank_k: int = 3,
        reranker_provider: Optional[str] = None,
        llm_provider: Optional[str] = None,
    ):
        self.retrieval_k = retrieval_k
        self.rerank_k = rerank_k
        self.reranker_provider = reranker_provider or os.getenv("RERANKER_PROVIDER", "local")
        self.llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "mock")

        # Initialize sub-modules
        self.retriever = RAGRetriever(top_k=self.retrieval_k)
        self.reranker = get_reranker(provider=self.reranker_provider)
        self.generator = RAGGenerator(provider=self.llm_provider)

    def run(self, query: str) -> RAGPipelineResult:
        """
        Executes the complete RAG flow for a user query.

        Flow:
            Query -> Vector Retrieval (Top N) -> Reranking (Top K) -> Grounded Generation
        """
        start_time = time.perf_counter()

        # Step 1: Initial broad semantic retrieval
        candidate_chunks = self.retriever.retrieve(query=query, top_k=self.retrieval_k)

        # Step 2: Cross-encoder reranking
        refined_chunks = self.reranker.rerank(
            query=query,
            chunks=candidate_chunks,
            top_k=self.rerank_k,
        )

        # Step 3: Grounded answer generation
        generation_response = self.generator.generate(
            query=query,
            chunks=refined_chunks,
        )

        total_latency = time.perf_counter() - start_time

        return RAGPipelineResult(
            query=query,
            answer=generation_response.answer,
            has_sufficient_context=generation_response.has_sufficient_context,
            sources=generation_response.sources,
            initial_candidates=candidate_chunks,
            reranked_chunks=refined_chunks,
            model_used=generation_response.model_used,
            reranker_used=self.reranker_provider,
            total_latency_seconds=round(total_latency, 3),
        )


if __name__ == "__main__":
    print("--- Testing Complete End-to-End RAG Pipeline ---")

    pipeline = RAGPipeline(retrieval_k=6, rerank_k=3)
    test_query = "What is continuous integration and how is code merged?"

    print(f">> Query: '{test_query}'\n")
    result = pipeline.run(test_query)

    print("Pipeline Execution Summary:")
    print(f"  Total Latency: {result.total_latency_seconds}s")
    print(f"  Reranker: {result.reranker_used}")
    print(f"  LLM Model: {result.model_used}")
    print(f"  Initial Candidates: {len(result.initial_candidates)}")
    print(f"  Reranked Best Chunks: {len(result.reranked_chunks)}")
    print("\nAnswer:")
    print("-" * 50)
    print(result.answer)
    print("-" * 50)
