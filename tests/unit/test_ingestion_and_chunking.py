import pytest
from pathlib import Path
from langchain_core.documents import Document
from app.rag.ingestion import load_documents, split_documents


def test_load_documents_empty_dir(tmp_path):
    empty_dir = tmp_path / "empty_docs"
    empty_dir.mkdir()
    docs = load_documents(str(empty_dir))
    assert docs == []


def test_split_documents_preserves_metadata():
    sample_doc = Document(
        page_content="DevOps is a set of practices that combines software development and IT operations. " * 20,
        metadata={"source": "Devops unit-1.pdf", "page": 1},
    )
    chunks = split_documents([sample_doc])
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["source"] == "Devops unit-1.pdf"
        assert chunk.metadata["page"] == 1
        assert len(chunk.page_content) <= 800
