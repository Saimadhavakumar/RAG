"""RAGGuard Evaluation Engine package."""
from app.evaluation.dataset import EvalSample, get_default_eval_dataset
from app.evaluation.metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
    calculate_faithfulness,
    calculate_answer_correctness,
)

__all__ = [
    "EvalSample",
    "get_default_eval_dataset",
    "calculate_precision_at_k",
    "calculate_recall_at_k",
    "calculate_reciprocal_rank",
    "calculate_faithfulness",
    "calculate_answer_correctness",
]
