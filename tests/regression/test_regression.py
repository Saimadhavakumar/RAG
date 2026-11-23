import pytest
from app.evaluation.evaluator import EvaluationReport, SampleEvaluationResult
from app.regression.regression_runner import compare_with_baseline


def _create_mock_report(pass_rate=100.0, precision=1.0, faithfulness=1.0, sample_passed=True):
    sample = SampleEvaluationResult(
        sample_id="DEV-001",
        query="What is DevOps?",
        category="functional",
        is_answerable=True,
        generated_answer="Answer",
        ground_truth_answer="Answer",
        retrieved_sources=["doc.pdf"],
        expected_sources=["doc.pdf"],
        precision_at_k=precision,
        recall_at_k=1.0,
        reciprocal_rank=1.0,
        ndcg_at_k=1.0,
        context_relevance=1.0,
        faithfulness=faithfulness,
        answer_correctness_f1=1.0,
        unsupported_claims=[],
        passed=sample_passed,
    )
    return EvaluationReport(
        timestamp="2026-11-23T12:00:00",
        total_samples=1,
        passed_samples=1 if sample_passed else 0,
        failed_samples=0 if sample_passed else 1,
        pass_rate_pct=pass_rate,
        mean_precision_at_k=precision,
        mean_recall_at_k=1.0,
        mean_reciprocal_rank=1.0,
        mean_ndcg_at_k=1.0,
        mean_faithfulness=faithfulness,
        mean_answer_correctness_f1=1.0,
        mean_context_relevance=1.0,
        sample_results=[sample],
    )


def test_compare_with_baseline_stable():
    baseline_dict = {
        "timestamp": "2026-11-20T10:00:00",
        "pass_rate_pct": 100.0,
        "mean_precision_at_k": 1.0,
        "mean_recall_at_k": 1.0,
        "mean_reciprocal_rank": 1.0,
        "mean_faithfulness": 1.0,
        "sample_results": [{"sample_id": "DEV-001", "passed": True}],
    }
    current_report = _create_mock_report(pass_rate=100.0, precision=1.0, faithfulness=1.0)
    analysis = compare_with_baseline(current_report, baseline_dict)

    assert analysis.regression_detected is False
    assert len(analysis.new_failures) == 0
    assert analysis.metric_deltas["mean_faithfulness"] == 0.0


def test_compare_with_baseline_detects_faithfulness_drop():
    baseline_dict = {
        "timestamp": "2026-11-20T10:00:00",
        "pass_rate_pct": 100.0,
        "mean_precision_at_k": 1.0,
        "mean_recall_at_k": 1.0,
        "mean_reciprocal_rank": 1.0,
        "mean_faithfulness": 1.0,
        "sample_results": [{"sample_id": "DEV-001", "passed": True}],
    }
    # Current report has faithfulness dropped to 0.80 (-0.20 drop > 0.05 tolerance)
    current_report = _create_mock_report(pass_rate=80.0, precision=1.0, faithfulness=0.80)
    analysis = compare_with_baseline(current_report, baseline_dict, max_allowed_drop=0.05)

    assert analysis.regression_detected is True
    assert any("Faithfulness" in r for r in analysis.regression_reasons)


def test_compare_with_baseline_detects_new_failure():
    baseline_dict = {
        "timestamp": "2026-11-20T10:00:00",
        "sample_results": [{"sample_id": "DEV-001", "passed": True}],
    }
    current_report = _create_mock_report(sample_passed=False)
    analysis = compare_with_baseline(current_report, baseline_dict)

    assert analysis.regression_detected is True
    assert len(analysis.new_failures) == 1
    assert "DEV-001" in analysis.new_failures[0]
