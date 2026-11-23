
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(directory: str):
    documents = []

    for file_path in Path(directory).glob("*.pdf"):
        loader = PyPDFLoader(str(file_path))
        documents.extend(loader.load())

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    return splitter.split_documents(documents)


if __name__ == "__main__":
    docs = load_documents("data/documents")
    chunks = split_documents(docs)

    print(f"Documents loaded: {len(docs)}")
    print(f"Chunks created: {len(chunks)}")

    if chunks:
        print("\nFirst chunk:")
        print(chunks[0].page_content[:500])
