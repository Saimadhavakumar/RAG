import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Prevent Windows symlink permission warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def get_embedding_model(provider: str = None, model_name: str = None):
    """
    Initializes and returns an embedding model.

    Supported providers:
      - 'local': HuggingFace sentence-transformers (100% free, runs on CPU, offline)
      - 'openai': OpenAI embeddings (requires paid OPENAI_API_KEY)
    """
    selected_provider = (
        provider or os.getenv("EMBEDDING_PROVIDER", "local")
    ).strip().lower()

    # 1. Local HuggingFace Embeddings (Free, Offline)
    if selected_provider == "local":
        model_id = (
            model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        ).strip()

        # To completely avoid Windows symlink permission errors in ~/.cache,
        # we download directly to a project-local folder as standard physical files.
        if model_id in ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"):
            local_dir = Path("data/models/all-MiniLM-L6-v2")
            if not (local_dir / "modules.json").exists():
                print("Downloading local embedding model into data/models (~80 MB, one-time setup)...")
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id="sentence-transformers/all-MiniLM-L6-v2",
                    local_dir=str(local_dir),
                )
            target_model = str(local_dir)
        else:
            target_model = model_id

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=target_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    # 2. OpenAI Cloud Embeddings
    elif selected_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key or api_key.startswith("your_"):
            raise ValueError(
                "OPENAI_API_KEY is missing or invalid in your .env file."
            )

        selected_model = (
            model_name or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        ).strip()

        return OpenAIEmbeddings(
            model=selected_model,
            api_key=api_key,
        )

    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER '{selected_provider}'. Choose 'local' or 'openai'."
        )


if __name__ == "__main__":
    import traceback

    try:
        print("Testing Local Embedding Engine...")
        embedding_model = get_embedding_model()
        sample_query = "What is RAGGuard AI?"
        vector = embedding_model.embed_query(sample_query)

        print("\nSUCCESS: Local embedding generated successfully!")
        print(f"Model used: {getattr(embedding_model, 'model_name', 'all-MiniLM-L6-v2')}")
        print(f"Vector dimensions: {len(vector)} (expected: 384 for all-MiniLM-L6-v2)")
        print(f"Sample values (first 3): {vector[:3]}")
    except Exception as err:
        print(f"\nFAILURE: {err}")
        traceback.print_exc()
