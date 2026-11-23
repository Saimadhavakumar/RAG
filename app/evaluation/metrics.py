import re
import math
from typing import List, Tuple, Set


def _normalize_text(text: str) -> List[str]:
    """Tokenizes and normalizes text into lower-cased alphanumeric tokens."""
    return re.findall(r"\b\w+\b", text.lower())


# ---------------------------------------------------------
# 1. Retrieval Metrics (Deterministic IR Metrics)
# ---------------------------------------------------------

def calculate_precision_at_k(
    retrieved_sources: List[str],
    expected_sources: List[str],
    k: int = 3,
) -> float:
    """
    Precision@K: Proportion of the top-k retrieved documents that are relevant.
    Score ranges from 0.0 to 1.0.
    """
    if not expected_sources:
        # If the query was unanswerable/out-of-domain, retrieving 0 is ideal
        return 1.0 if not retrieved_sources else 0.0

    top_k = retrieved_sources[:k]
    if not top_k:
        return 0.0

    expected_set = {s.lower() for s in expected_sources}
    hits = sum(1 for src in top_k if src.lower() in expected_set)
    return round(hits / len(top_k), 4)


def calculate_recall_at_k(
    retrieved_sources: List[str],
    expected_sources: List[str],
    k: int = 3,
) -> float:
    """
    Recall@K: Proportion of expected relevant documents captured in the top-k.
    Score ranges from 0.0 to 1.0.
    """
    if not expected_sources:
        return 1.0 if not retrieved_sources else 0.0

    top_k_set = {s.lower() for s in retrieved_sources[:k]}
    expected_set = {s.lower() for s in expected_sources}

    hits = sum(1 for src in expected_set if src in top_k_set)
    return round(hits / len(expected_set), 4)


def calculate_reciprocal_rank(
    retrieved_sources: List[str],
    expected_sources: List[str],
) -> float:
    """
    Reciprocal Rank (RR): 1 / rank of the first relevant document.
    Returns 1.0 if the first hit is at Rank 1, 0.5 if at Rank 2, 0.0 if not in list.
    """
    if not expected_sources:
        return 1.0 if not retrieved_sources else 0.0

    expected_set = {s.lower() for s in expected_sources}

    for rank, src in enumerate(retrieved_sources, start=1):
        if src.lower() in expected_set:
            return round(1.0 / rank, 4)

    return 0.0


def calculate_ndcg_at_k(
    retrieved_sources: List[str],
    expected_sources: List[str],
    k: int = 3,
) -> float:
    """
    Normalized Discounted Cumulative Gain (NDCG@K):
    Measures ranking quality with higher reward for relevant documents placed higher.
    """
    if not expected_sources:
        return 1.0 if not retrieved_sources else 0.0

    expected_set = {s.lower() for s in expected_sources}
    top_k = retrieved_sources[:k]

    dcg = 0.0
    for i, src in enumerate(top_k):
        rel = 1.0 if src.lower() in expected_set else 0.0
        dcg += rel / math.log2(i + 2)

    # Ideal DCG (all expected hits ranked first)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(expected_sources), k)))

    if idcg == 0.0:
        return 0.0

    return round(dcg / idcg, 4)


# ---------------------------------------------------------
# 2. Answer Quality & Faithfulness (Hallucination Detection)
# ---------------------------------------------------------

def calculate_faithfulness(
    answer: str,
    context: str,
    threshold: float = 0.35,
) -> Tuple[float, List[str]]:
    """
    Measures groundedness / faithfulness of generated answer against retrieved context.
    Detects potential hallucinations by checking whether claims in the answer
    are supported by tokens present in the retrieved context.

    Returns:
        (faithfulness_score [0.0 - 1.0], list_of_unsupported_claims)
    """
    if not answer or not answer.strip():
        return 0.0, ["Empty answer"]

    # Refusal handling: If the answer correctly stated insufficient context, it is 100% faithful
    refusal_phrases = ["not have sufficient information", "no relevant information", "context does not mention"]
    if any(p in answer.lower() for p in refusal_phrases):
        return 1.0, []

    context_tokens = set(_normalize_text(context))
    if not context_tokens:
        return 0.0, ["Context is empty; answer is entirely ungrounded"]

    # Split answer into distinct statements / sentences
    raw_sentences = [s.strip() for s in re.split(r"[.!?\n]", answer) if len(s.strip()) > 15]

    # Filter out citation lines like "[Source: Devops unit-3.pdf, Page: 24]"
    claims = [s for s in raw_sentences if not s.startswith("[Source:")]

    if not claims:
        return 1.0, []

    unsupported_claims = []
    supported_count = 0

    for claim in claims:
        claim_tokens = _normalize_text(claim)
        # Remove common English stop words to focus on meaningful content words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "to", "in", "of", "that", "this", "it", "with", "as"}
        content_tokens = [t for t in claim_tokens if t not in stop_words]

        if not content_tokens:
            supported_count += 1
            continue

        overlap = sum(1 for t in content_tokens if t in context_tokens)
        overlap_ratio = overlap / len(content_tokens)

        if overlap_ratio >= threshold:
            supported_count += 1
        else:
            unsupported_claims.append(claim)

    score = round(supported_count / len(claims), 4)
    return score, unsupported_claims


def calculate_answer_correctness(
    generated_answer: str,
    ground_truth_answer: str,
) -> float:
    """
    Computes token-level F1 score between generated answer and ground truth answer.
    Harmonic mean of token precision and recall.
    """
    gen_tokens = _normalize_text(generated_answer)
    truth_tokens = _normalize_text(ground_truth_answer)

    if not gen_tokens or not truth_tokens:
        return 1.0 if gen_tokens == truth_tokens else 0.0

    gen_set = set(gen_tokens)
    truth_set = set(truth_tokens)

    common = gen_set.intersection(truth_set)
    if not common:
        return 0.0

    precision = len(common) / len(gen_set)
    recall = len(common) / len(truth_set)

    f1 = 2 * (precision * recall) / (precision + recall)
    return round(f1, 4)


def calculate_context_relevance(
    query: str,
    context_chunks: List[str],
) -> float:
    """
    Measures lexical intent overlap between the user's query and the retrieved context chunks.
    """
    query_tokens = set(_normalize_text(query))
    if not query_tokens or not context_chunks:
        return 0.0

    all_context_tokens = set()
    for chunk in context_chunks:
        all_context_tokens.update(_normalize_text(chunk))

    overlap = query_tokens.intersection(all_context_tokens)
    return round(len(overlap) / len(query_tokens), 4)
