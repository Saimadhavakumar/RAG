import os
import hashlib
from typing import List, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_chroma import Chroma
from app.rag.embeddings import get_embedding_model

load_dotenv()

DEFAULT_COLLECTION_NAME = "ragguard_docs"
DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "data/chroma_db")


def _generate_chunk_id(doc: Document, idx: int) -> str:
    """Generates a deterministic unique ID for a document chunk to prevent duplicates."""
    source = Path(doc.metadata.get("source", "doc")).stem
    page = doc.metadata.get("page", 0)
    content_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()[:10]
    return f"{source}_p{page}_c{idx}_{content_hash}"


def get_vector_store(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str = DEFAULT_PERSIST_DIR,
) -> Chroma:
    """
    Initializes and returns a persistent Chroma vector store instance.

    Args:
        collection_name: Name of the Chroma collection.
        persist_directory: Local directory for database persistence.

    Returns:
        Chroma vector store instance.
    """
    embedding_model = get_embedding_model()
    Path(persist_directory).mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_directory,
    )


def index_documents(
    documents: List[Document],
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str = DEFAULT_PERSIST_DIR,
) -> int:
    """
    Indexes a list of Document chunks into the persistent Chroma vector store.
    Uses deterministic chunk IDs so re-indexing is idempotent (no duplicates).

    Args:
        documents: List of chunked Document objects.
        collection_name: Name of the collection.
        persist_directory: Path to persist the vector database.

    Returns:
        Number of documents indexed.
    """
    if not documents:
        raise ValueError("Cannot index an empty document list.")

    vector_store = get_vector_store(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    # Generate deterministic IDs for deduplication
    ids = [_generate_chunk_id(doc, i) for i, doc in enumerate(documents)]

    # Ingest chunks into ChromaDB with deterministic IDs
    vector_store.add_documents(documents=documents, ids=ids)
    return len(documents)


def search_similar(
    query: str,
    k: int = 4,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str = DEFAULT_PERSIST_DIR,
) -> List[Tuple[Document, float]]:
    """
    Searches the vector store for top-k chunks most semantically similar to the query.

    Args:
        query: User question or search query.
        k: Number of candidate chunks to retrieve.
        collection_name: Name of the collection.
        persist_directory: Database storage path.

    Returns:
        List of (Document, similarity_score) tuples.
    """
    vector_store = get_vector_store(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    return vector_store.similarity_search_with_score(query=query, k=k)


if __name__ == "__main__":
    from app.rag.ingestion import load_documents, split_documents

    print("--- Testing ChromaDB Vector Store Integration ---")
    
    # 1. Ingest documents
    raw_docs = load_documents("data/documents")
    print(f"Loaded {len(raw_docs)} pages from data/documents.")

    chunks = split_documents(raw_docs)
    print(f"Generated {len(chunks)} text chunks.")

    # 2. Index into ChromaDB
    print("\nIndexing chunks into ChromaDB (this creates embeddings & stores them on disk)...")
    indexed_count = index_documents(chunks)
    print(f"Successfully indexed {indexed_count} chunks into ChromaDB!")

    # 3. Test Retrieval
    test_query = "What is continuous integration and delivery?"
    print(f"\nRunning test semantic query: '{test_query}'")
    results = search_similar(test_query, k=2)

    print(f"\nRetrieved {len(results)} relevant chunks:")
    for idx, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0)
        snippet = doc.page_content[:200].replace("\n", " ")
        print(f"  [{idx}] Score (L2 Distance): {score:.4f} | Source: {source} (Page {page})")
        print(f"      Content: {snippet}...\n")
