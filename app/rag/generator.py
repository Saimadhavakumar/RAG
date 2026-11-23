import os
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage
from app.rag.reranker import RerankedChunk

load_dotenv()

DEFAULT_SYSTEM_PROMPT = """You are RAGGuard AI Assistant, a reliable and truthful technical assistant.
Your task is to answer the user's question accurately using ONLY the provided retrieved context.

Strict Grounding Rules:
1. Base your answer strictly on the facts directly stated in the context below.
2. If the context does not contain enough information to answer the question, state:
   "I do not have sufficient information in the provided documents to answer this question."
3. Do NOT extrapolate, speculate, or bring in outside knowledge not mentioned in the context.
4. Cite your sources using the format: [Source: <filename>, Page: <page>].
"""


@dataclass
class RAGResponse:
    """Structured response from the RAG generation engine."""
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    has_sufficient_context: bool
    model_used: str
    latency_seconds: float


class RAGGenerator:
    """
    RAG Answer Generation Engine.
    Constructs grounded prompts, injects retrieved context, and invokes the configured LLM.
    Supports OpenAI, Groq (OpenAI-compatible), and an offline Mock provider for testing.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.provider = (
            provider or os.getenv("LLM_PROVIDER", "mock")
        ).strip().lower()
        self.temperature = temperature
        self.model_name = model_name or self._get_default_model()
        self.client = self._init_client()

    def _get_default_model(self) -> str:
        if self.provider == "openai":
            return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        elif self.provider == "groq":
            return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        return "mock-grounded-generator"

    def _init_client(self):
        if self.provider == "openai":
            from langchain_openai import ChatOpenAI
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key or api_key.startswith("your_"):
                raise ValueError("OPENAI_API_KEY is missing or invalid in .env")
            return ChatOpenAI(
                model=self.model_name,
                api_key=api_key,
                temperature=self.temperature,
            )

        elif self.provider == "groq":
            from langchain_openai import ChatOpenAI
            api_key = os.getenv("GROQ_API_KEY", "").strip()
            if not api_key:
                raise ValueError("GROQ_API_KEY is not configured in .env")
            return ChatOpenAI(
                model=self.model_name,
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
                temperature=self.temperature,
            )

        elif self.provider == "mock":
            return None

        else:
            raise ValueError(
                f"Unknown LLM_PROVIDER '{self.provider}'. Supported: 'openai', 'groq', 'mock'."
            )

    def _format_context(self, chunks: List[RerankedChunk]) -> str:
        """Formats retrieved chunks with source and page markers for context injection."""
        context_blocks = []
        for i, chunk in enumerate(chunks, 1):
            source_file = os.path.basename(chunk.source)
            block = (
                f"--- Document Context [{i}] ---\n"
                f"Source: {source_file}\n"
                f"Page: {chunk.page}\n"
                f"Content:\n{chunk.content.strip()}\n"
            )
            context_blocks.append(block)
        return "\n".join(context_blocks)

    def generate(
        self,
        query: str,
        chunks: List[RerankedChunk],
    ) -> RAGResponse:
        """
        Generates a grounded answer from the provided query and reranked chunks.
        """
        start_time = time.perf_counter()

        # Compile sources list
        sources = [
            {
                "source": os.path.basename(c.source),
                "page": c.page,
                "rank": c.reranked_rank,
                "score": c.rerank_score,
            }
            for c in chunks
        ]

        # Context Insufficiency Check: If no chunks retrieved or empty context
        if not chunks:
            latency = time.perf_counter() - start_time
            return RAGResponse(
                query=query,
                answer="I do not have sufficient information in the provided documents to answer this question.",
                sources=[],
                has_sufficient_context=False,
                model_used=self.model_name,
                latency_seconds=round(latency, 3),
            )

        context_text = self._format_context(chunks)

        # 1. Mock Generator (Deterministic, offline, fast for testing & CI)
        if self.provider == "mock":
            best_chunk = chunks[0]
            clean_snippet = best_chunk.content.replace("\n", " ").strip()
            # Synthesize answer from top grounded passage
            answer = (
                f"Based on the provided documentation, {clean_snippet[:250]}...\n\n"
                f"[Source: {os.path.basename(best_chunk.source)}, Page: {best_chunk.page}]"
            )
            latency = time.perf_counter() - start_time
            return RAGResponse(
                query=query,
                answer=answer,
                sources=sources,
                has_sufficient_context=True,
                model_used="mock-grounded-generator",
                latency_seconds=round(latency, 3),
            )

        # 2. Real LLM Generation (OpenAI / Groq)
        user_prompt = (
            f"Retrieved Context:\n{context_text}\n\n"
            f"User Question: {query}\n\n"
            f"Answer:"
        )

        messages = [
            SystemMessage(content=DEFAULT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            ai_message = self.client.invoke(messages)
            raw_answer = ai_message.content.strip()

            has_sufficient = "not have sufficient information" not in raw_answer.lower()
            latency = time.perf_counter() - start_time

            return RAGResponse(
                query=query,
                answer=raw_answer,
                sources=sources,
                has_sufficient_context=has_sufficient,
                model_used=self.model_name,
                latency_seconds=round(latency, 3),
            )
        except Exception as e:
            latency = time.perf_counter() - start_time
            raise RuntimeError(f"LLM Generation failed ({self.provider}): {e}") from e


if __name__ == "__main__":
    from app.rag.retriever import RAGRetriever
    from app.rag.reranker import get_reranker

    print("--- Testing RAG Answer Generator ---")

    retriever = RAGRetriever(top_k=4)
    reranker = get_reranker(provider="local")
    generator = RAGGenerator(provider="mock")

    test_query = "What is continuous integration and how is code merged?"
    print(f">> Query: '{test_query}'")

    # 1. Retrieve candidates
    candidates = retriever.retrieve(test_query, top_k=4)
    # 2. Rerank
    refined_chunks = reranker.rerank(test_query, candidates, top_k=2)
    # 3. Generate answer
    response = generator.generate(test_query, refined_chunks)

    print(f"\nModel: {response.model_used} (Latency: {response.latency_seconds}s)")
    print(f"Sufficient Context: {response.has_sufficient_context}")
    print(f"Sources Referenced: {len(response.sources)}")
    print("\nGenerated Answer:")
    print("-" * 50)
    print(response.answer)
    print("-" * 50)
