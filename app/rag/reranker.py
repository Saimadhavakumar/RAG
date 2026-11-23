import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
from dotenv import load_dotenv

from app.rag.retriever import RetrievedChunk

load_dotenv()

# Prevent Windows symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


@dataclass
class RerankedChunk:
    """Represents a chunk that has been re-scored and re-ordered by a reranker."""
    content: str
    metadata: Dict[str, Any]
    source: str
    page: int
    original_rank: int
    reranked_rank: int
    retrieval_score: float  # Initial vector distance
    rerank_score: float     # Cross-encoder relevance score (higher is more relevant)


class BaseReranker(ABC):
    """Abstract interface for all RAGGuard reranker providers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int = 3,
    ) -> List[RerankedChunk]:
        """Reranks retrieved candidate chunks based on query-document relevance."""
        pass


class NoOpReranker(BaseReranker):
    """Pass-through reranker that preserves initial vector retrieval order (baseline)."""

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int = 3,
    ) -> List[RerankedChunk]:
        selected = chunks[:top_k]
        return [
            RerankedChunk(
                content=chunk.content,
                metadata=chunk.metadata,
                source=chunk.source,
                page=chunk.page,
                original_rank=chunk.rank,
                reranked_rank=idx + 1,
                retrieval_score=chunk.score,
                rerank_score=1.0 / (1.0 + chunk.score),  # normalized proxy score
            )
            for idx, chunk in enumerate(selected)
        ]


class LocalCrossEncoderReranker(BaseReranker):
    """
    High-precision local cross-encoder reranker.
    Uses 'cross-encoder/ms-marco-MiniLM-L-6-v2' to perform joint query-chunk attention.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.local_dir = Path("data/models/ms-marco-MiniLM-L-6-v2")
        self._ensure_model_downloaded()

        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(str(self.local_dir), device="cpu")

    def _ensure_model_downloaded(self):
        """Downloads model physical files directly to avoid Windows symlink permission errors."""
        if not (self.local_dir / "config.json").exists():
            print("Downloading local Cross-Encoder model into data/models (~80 MB, one-time setup)...")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=self.model_name,
                local_dir=str(self.local_dir),
            )

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int = 3,
    ) -> List[RerankedChunk]:
        if not chunks:
            return []

        # Prepare pairs: [query, document_content]
        pairs = [[query, chunk.content] for chunk in chunks]

        # Compute cross-encoder relevance scores
        scores = self.model.predict(pairs)

        # Attach rerank score to each candidate
        scored_candidates = []
        for chunk, score in zip(chunks, scores):
            scored_candidates.append((chunk, float(score)))

        # Sort descending by cross-encoder score (highest relevance first)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Slice top_k and return structured RerankedChunk
        results = []
        for new_rank, (chunk, rerank_score) in enumerate(scored_candidates[:top_k], start=1):
            results.append(
                RerankedChunk(
                    content=chunk.content,
                    metadata=chunk.metadata,
                    source=chunk.source,
                    page=chunk.page,
                    original_rank=chunk.rank,
                    reranked_rank=new_rank,
                    retrieval_score=chunk.score,
                    rerank_score=round(rerank_score, 4),
                )
            )
        return results


class CohereReranker(BaseReranker):
    """Hosted cloud reranker using Cohere Rerank API (alternative provider)."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY is not set in environment or .env file.")

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int = 3,
    ) -> List[RerankedChunk]:
        import cohere
        client = cohere.ClientV2(api_key=self.api_key)

        docs = [c.content for c in chunks]
        response = client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=docs,
            top_n=top_k,
        )

        results = []
        for new_rank, item in enumerate(response.results, start=1):
            original_chunk = chunks[item.index]
            results.append(
                RerankedChunk(
                    content=original_chunk.content,
                    metadata=original_chunk.metadata,
                    source=original_chunk.source,
                    page=original_chunk.page,
                    original_rank=original_chunk.rank,
                    reranked_rank=new_rank,
                    retrieval_score=original_chunk.score,
                    rerank_score=round(float(item.relevance_score), 4),
                )
            )
        return results


def get_reranker(provider: Optional[str] = None) -> BaseReranker:
    """
    Factory function to initialize the configured reranker provider.
    Supported providers: 'local' (default), 'none', 'cohere'.
    """
    selected = (provider or os.getenv("RERANKER_PROVIDER", "local")).strip().lower()

    if selected == "local":
        return LocalCrossEncoderReranker()
    elif selected == "none":
        return NoOpReranker()
    elif selected == "cohere":
        return CohereReranker()
    else:
        raise ValueError(
            f"Unknown RERANKER_PROVIDER: '{selected}'. Choose 'local', 'none', or 'cohere'."
        )


if __name__ == "__main__":
    from app.rag.retriever import RAGRetriever

    print("--- Testing Two-Stage RAG: Retrieval + Cross-Encoder Reranking ---")

    # Step 1: Initial broad retrieval (Top 6 candidates)
    retriever = RAGRetriever(top_k=6)
    test_query = "What is continuous integration and how is code merged?"
    print(f"\n>> User Query: '{test_query}'")

    candidates = retriever.retrieve(test_query, top_k=6)
    print(f"Stage 1 (Bi-Encoder Retrieval): Retrieved {len(candidates)} candidates.")

    # Step 2: High-precision Reranking (Top 3)
    reranker = get_reranker(provider="local")
    reranked = reranker.rerank(query=test_query, chunks=candidates, top_k=3)

    print("\nStage 2 (Cross-Encoder Reranking): Top 3 Refined Chunks:")
    for chunk in reranked:
        snippet = chunk.content[:160].replace("\n", " ")
        rank_change = (
            f"Promoted (+{chunk.original_rank - chunk.reranked_rank})"
            if chunk.original_rank > chunk.reranked_rank
            else (
                f"Demoted (-{chunk.reranked_rank - chunk.original_rank})"
                if chunk.original_rank < chunk.reranked_rank
                else "Unchanged"
            )
        )
        print(
            f"  [New Rank {chunk.reranked_rank}] Cross-Encoder Score: {chunk.rerank_score:+.4f} | "
            f"Original Rank: {chunk.original_rank} ({rank_change})"
        )
        print(f"      Source: {os.path.basename(chunk.source)} (Page {chunk.page})")
        print(f"      Snippet: {snippet}...\n")
