import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain_core.documents import Document
from app.rag.vector_store import get_vector_store, DEFAULT_COLLECTION_NAME

load_dotenv()


@dataclass
class RetrievedChunk:
    """Represents a single retrieved document chunk with relevance scoring and source attribution."""
    content: str
    metadata: Dict[str, Any]
    score: float  # Distance score (lower is more similar in L2 distance)
    source: str
    page: int
    rank: int


class RAGRetriever:
    """
    Production retriever for RAG applications.
    Handles semantic retrieval, top-k candidate selection, score thresholding,
    and metadata filtering.
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ):
        """
        Args:
            collection_name: ChromaDB collection to retrieve from.
            top_k: Number of candidate chunks to retrieve.
            score_threshold: Optional maximum distance threshold (chunks further than this are filtered out).
        """
        self.collection_name = collection_name
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.vector_store = get_vector_store(collection_name=collection_name)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """
        Executes semantic retrieval for a query.

        Args:
            query: User's query string.
            top_k: Optional override for number of candidates.
            filter_metadata: Optional ChromaDB metadata filter dict (e.g. {"page": 25}).
            score_threshold: Optional override for distance threshold.

        Returns:
            List of structured RetrievedChunk objects sorted by relevance.
        """
        if not query or not query.strip():
            return []

        k = top_k or self.top_k
        threshold = score_threshold or self.score_threshold

        # Query vector store with distance scores and optional metadata filtering
        raw_results = self.vector_store.similarity_search_with_score(
            query=query.strip(),
            k=k,
            filter=filter_metadata,
        )

        retrieved_chunks = []
        rank = 1

        for doc, score in raw_results:
            # If a threshold is set, ignore chunks that are too distant / irrelevant
            if threshold is not None and score > threshold:
                continue

            chunk = RetrievedChunk(
                content=doc.page_content,
                metadata=doc.metadata,
                score=round(float(score), 4),
                source=doc.metadata.get("source", "Unknown"),
                page=int(doc.metadata.get("page", 0)),
                rank=rank,
            )
            retrieved_chunks.append(chunk)
            rank += 1

        return retrieved_chunks


if __name__ == "__main__":
    print("--- Testing RAG Retriever Module ---")

    retriever = RAGRetriever(top_k=3)
    sample_queries = [
        "What is continuous integration?",
        "Explain DevOps culture and benefits",
        "Something completely unrelated like astronomy and black holes",
    ]

    for q in sample_queries:
        print(f"\n>> Query: '{q}'")
        chunks = retriever.retrieve(q, top_k=2)

        if not chunks:
            print("   No relevant chunks found.")
            continue

        for chunk in chunks:
            snippet = chunk.content[:150].replace("\n", " ")
            print(
                f"   [Rank {chunk.rank}] L2 Distance: {chunk.score} | "
                f"Source: {os.path.basename(chunk.source)} (Page {chunk.page})"
            )
            print(f"      Text: {snippet}...")
