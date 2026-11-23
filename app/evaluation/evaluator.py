import os
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from app.rag.pipeline import RAGPipeline, RAGPipelineResult
from app.evaluation.dataset import EvalSample, get_default_eval_dataset
from app.evaluation.metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
    calculate_ndcg_at_k,
    calculate_faithfulness,
    calculate_answer_correctness,
    calculate_context_relevance,
)


@dataclass
class SampleEvaluationResult:
    """Individual test case evaluation output with metric scores and failure evidence."""
    sample_id: str
    query: str
    category: str
    is_answerable: bool
    generated_answer: str
    ground_truth_answer: str
    retrieved_sources: List[str]
    expected_sources: List[str]
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    context_relevance: float
    faithfulness: float
    answer_correctness_f1: float
    unsupported_claims: List[str]
    passed: bool
    failure_reason: Optional[str] = None


@dataclass
class EvaluationReport:
    """Aggregated evaluation summary containing overall QA scores and sample breakdowns."""
    timestamp: str
    total_samples: int
    passed_samples: int
    failed_samples: int
    pass_rate_pct: float
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    mean_ndcg_at_k: float
    mean_faithfulness: float
    mean_answer_correctness_f1: float
    mean_context_relevance: float
    sample_results: List[SampleEvaluationResult]


class RAGEvaluator:
    """
    Independent Evaluation Engine for RAG applications (System B).
    Evaluates retrieval quality, reranking accuracy, answer faithfulness, and hallucination rates.
    """

    def __init__(
        self,
        pipeline: Optional[RAGPipeline] = None,
        k: int = 3,
        min_faithfulness_threshold: float = 0.80,
    ):
        self.pipeline = pipeline or RAGPipeline(retrieval_k=6, rerank_k=k)
        self.k = k
        self.min_faithfulness = min_faithfulness_threshold

    def evaluate_sample(
        self,
        sample: EvalSample,
        result: RAGPipelineResult,
    ) -> SampleEvaluationResult:
        """Evaluates a single RAG pipeline output against ground-truth annotations."""
        retrieved_sources = [os.path.basename(c.source) for c in result.reranked_chunks]
        context_text = "\n".join(c.content for c in result.reranked_chunks)

        # 1. Retrieval Metrics
        p_at_k = calculate_precision_at_k(retrieved_sources, sample.expected_sources, k=self.k)
        r_at_k = calculate_recall_at_k(retrieved_sources, sample.expected_sources, k=self.k)
        mrr = calculate_reciprocal_rank(retrieved_sources, sample.expected_sources)
        ndcg = calculate_ndcg_at_k(retrieved_sources, sample.expected_sources, k=self.k)
        ctx_rel = calculate_context_relevance(sample.query, [c.content for c in result.reranked_chunks])

        # 2. Answer Quality & Faithfulness
        faithfulness, unsupported_claims = calculate_faithfulness(result.answer, context_text)
        f1_correctness = calculate_answer_correctness(result.answer, sample.ground_truth_answer)

        # 3. Pass/Fail Determination
        passed = True
        failure_reasons = []

        if sample.is_answerable:
            if faithfulness < self.min_faithfulness:
                passed = False
                failure_reasons.append(f"Faithfulness below threshold ({faithfulness} < {self.min_faithfulness})")
            if p_at_k == 0.0 and sample.expected_sources:
                passed = False
                failure_reasons.append("Zero retrieval precision for expected source documents")
        else:
            # Unanswerable query must be recognized as having insufficient context
            if result.has_sufficient_context and "not have sufficient information" not in result.answer.lower():
                passed = False
                failure_reasons.append("Model hallucinated answer for an out-of-domain unanswerable question")

        return SampleEvaluationResult(
            sample_id=sample.id,
            query=sample.query,
            category=sample.category,
            is_answerable=sample.is_answerable,
            generated_answer=result.answer,
            ground_truth_answer=sample.ground_truth_answer,
            retrieved_sources=retrieved_sources,
            expected_sources=sample.expected_sources,
            precision_at_k=p_at_k,
            recall_at_k=r_at_k,
            reciprocal_rank=mrr,
            ndcg_at_k=ndcg,
            context_relevance=ctx_rel,
            faithfulness=faithfulness,
            answer_correctness_f1=f1_correctness,
            unsupported_claims=unsupported_claims,
            passed=passed,
            failure_reason="; ".join(failure_reasons) if failure_reasons else None,
        )

    def run_evaluation(
        self,
        dataset: Optional[List[EvalSample]] = None,
    ) -> EvaluationReport:
        """Runs the entire evaluation suite across all benchmark samples."""
        eval_data = dataset or get_default_eval_dataset()
        results: List[SampleEvaluationResult] = []

        print(f"Starting RAGGuard Evaluation across {len(eval_data)} benchmark samples...\n")

        for sample in eval_data:
            pipeline_result = self.pipeline.run(sample.query)
            sample_eval = self.evaluate_sample(sample, pipeline_result)
            results.append(sample_eval)

        # Aggregate summary statistics
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        report = EvaluationReport(
            timestamp=datetime.now().isoformat(),
            total_samples=total,
            passed_samples=passed,
            failed_samples=failed,
            pass_rate_pct=round((passed / total) * 100, 2) if total else 0.0,
            mean_precision_at_k=round(sum(r.precision_at_k for r in results) / total, 4),
            mean_recall_at_k=round(sum(r.recall_at_k for r in results) / total, 4),
            mean_reciprocal_rank=round(sum(r.reciprocal_rank for r in results) / total, 4),
            mean_ndcg_at_k=round(sum(r.ndcg_at_k for r in results) / total, 4),
            mean_faithfulness=round(sum(r.faithfulness for r in results) / total, 4),
            mean_answer_correctness_f1=round(sum(r.answer_correctness_f1 for r in results) / total, 4),
            mean_context_relevance=round(sum(r.context_relevance for r in results) / total, 4),
            sample_results=results,
        )

        return report


if __name__ == "__main__":
    evaluator = RAGEvaluator()
    report = evaluator.run_evaluation()

    print("==================================================")
    print("          RAGGUARD AI EVALUATION REPORT           ")
    print("==================================================")
    print(f"Timestamp: {report.timestamp}")
    print(f"Total Test Cases: {report.total_samples}")
    print(f"Passed:           {report.passed_samples} ({report.pass_rate_pct}%)")
    print(f"Failed:           {report.failed_samples}")
    print("-" * 50)
    print("RETRIEVAL METRICS:")
    print(f"  Mean Precision@3:  {report.mean_precision_at_k:.4f}")
    print(f"  Mean Recall@3:     {report.mean_recall_at_k:.4f}")
    print(f"  Mean MRR:          {report.mean_reciprocal_rank:.4f}")
    print(f"  Mean NDCG@3:       {report.mean_ndcg_at_k:.4f}")
    print(f"  Context Relevance: {report.mean_context_relevance:.4f}")
    print("-" * 50)
    print("ANSWER QUALITY & FAITHFULNESS METRICS:")
    print(f"  Mean Faithfulness (Hallucination check): {report.mean_faithfulness:.4f}")
    print(f"  Mean Answer Correctness (F1):          {report.mean_answer_correctness_f1:.4f}")
    print("==================================================\n")

    print("DETAILED TEST CASE BREAKDOWN:")
    for res in report.sample_results:
        status_tag = "[PASS]" if res.passed else "[FAIL]"
        print(f"  {status_tag} {res.sample_id}: {res.query}")
        print(f"         Precision: {res.precision_at_k} | Recall: {res.recall_at_k} | Faithfulness: {res.faithfulness}")
        if not res.passed:
            print(f"         Failure Reason: {res.failure_reason}")
            if res.unsupported_claims:
                print(f"         Unsupported Claims: {res.unsupported_claims}")
        print()
