import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class EvalSample:
    """Represents a single labelled ground-truth evaluation test case."""
    id: str
    query: str
    ground_truth_answer: str
    expected_sources: List[str]  # e.g., ["Devops unit-3.pdf"]
    expected_page_hint: Optional[int] = None
    expected_keywords: Optional[List[str]] = None
    is_answerable: bool = True
    category: str = "functional"  # 'functional', 'edge_case', 'unanswerable', 'adversarial'


def get_default_eval_dataset() -> List[EvalSample]:
    """
    Returns a curated benchmark dataset tailored to the ingested DevOps documentation.
    Includes normal functional questions, boundary cases, and unanswerable queries.
    """
    return [
        EvalSample(
            id="DEV-001",
            query="What is the definition of DevOps?",
            ground_truth_answer=(
                "DevOps is a combination of cultural philosophies, practices, and tools that "
                "increases an organization's ability to deliver applications and services at high velocity."
            ),
            expected_sources=["Devops unit-3.pdf"],
            expected_page_hint=1,
            expected_keywords=["cultural philosophies", "processes", "tools", "deliver services"],
            is_answerable=True,
            category="functional",
        ),
        EvalSample(
            id="DEV-002",
            query="What is continuous integration and how do developers commit code?",
            ground_truth_answer=(
                "Continuous Integration is a DevOps phase where developers commit code changes to a shared repository. "
                "The code supporting new functionality is continuously integrated with existing code and tested."
            ),
            expected_sources=["Devops unit-3.pdf"],
            expected_page_hint=24,
            expected_keywords=["Continuous Integration", "commit changes", "integrated with the existing code"],
            is_answerable=True,
            category="functional",
        ),
        EvalSample(
            id="DEV-003",
            query="What are the main stages in the DevOps life cycle?",
            ground_truth_answer=(
                "The DevOps life cycle includes stages such as continuous development, continuous integration, "
                "continuous testing, continuous deployment, and continuous monitoring."
            ),
            expected_sources=["Devops unit-3.pdf"],
            expected_page_hint=23,
            expected_keywords=["continuous integration", "DevOps life cycle", "continuous development"],
            is_answerable=True,
            category="functional",
        ),
        EvalSample(
            id="DEV-004",
            query="What is the event horizon of a black hole in astrophysics?",
            ground_truth_answer="I do not have sufficient information in the provided documents to answer this question.",
            expected_sources=[],
            expected_keywords=[],
            is_answerable=False,
            category="unanswerable",
        ),
        EvalSample(
            id="DEV-005",
            query="Explain quantum supremacy and qubit entanglement in quantum computing.",
            ground_truth_answer="I do not have sufficient information in the provided documents to answer this question.",
            expected_sources=[],
            expected_keywords=[],
            is_answerable=False,
            category="unanswerable",
        ),
    ]


def save_eval_dataset_to_json(dataset: List[EvalSample], file_path: str):
    """Saves evaluation samples to a JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in dataset], f, indent=2)


def load_eval_dataset_from_json(file_path: str) -> List[EvalSample]:
    """Loads evaluation samples from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [EvalSample(**item) for item in data]
