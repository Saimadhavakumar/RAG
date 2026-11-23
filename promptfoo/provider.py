import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag.pipeline import RAGPipeline

# Initialize cached pipeline instance
_pipeline = RAGPipeline(retrieval_k=4, rerank_k=2)


def call_api(prompt: str, options=None, context=None):
    """
    Promptfoo Python Provider adapter.
    Executes a user query through the RAGPipeline and returns the generated answer.
    """
    try:
        result = _pipeline.run(prompt)
        return {
            "output": result.answer,
            "metadata": {
                "has_sufficient_context": result.has_sufficient_context,
                "latency_seconds": result.total_latency_seconds,
                "sources": result.sources,
            },
        }
    except Exception as e:
        return {"error": str(e)}
