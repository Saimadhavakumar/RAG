import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

from app.evaluation.evaluator import RAGEvaluator, EvaluationReport

DEFAULT_BASELINE_PATH = "data/baselines/baseline_v1.json"


@dataclass
class RegressionAnalysis:
    """Detailed comparison between a current evaluation run and a historical golden baseline."""
    baseline_timestamp: str
    current_timestamp: str
    metric_deltas: Dict[str, float]
    regression_detected: bool
    regression_reasons: List[str]
    new_failures: List[str]
    new_fixes: List[str]


def save_baseline(report: EvaluationReport, baseline_path: str = DEFAULT_BASELINE_PATH):
    """Serializes an evaluation report as a golden baseline for future regression tracking."""
    path = Path(baseline_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)
    print(f"Golden baseline successfully saved to: {baseline_path}")


def load_baseline(baseline_path: str = DEFAULT_BASELINE_PATH) -> Optional[Dict[str, Any]]:
    """Loads a previously saved baseline JSON report if it exists."""
    path = Path(baseline_path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_with_baseline(
    current_report: EvaluationReport,
    baseline_data: Dict[str, Any],
    max_allowed_drop: float = 0.05,
) -> RegressionAnalysis:
    """
    Compares the current evaluation report against a golden baseline.

    Args:
        current_report: Output from current evaluation run.
        baseline_data: Dictionary loaded from golden baseline.
        max_allowed_drop: Maximum acceptable drop in key metrics before flagging a regression (default 5%).

    Returns:
        Structured RegressionAnalysis object.
    """
    deltas = {
        "pass_rate_pct": round(current_report.pass_rate_pct - baseline_data.get("pass_rate_pct", 0.0), 2),
        "mean_precision_at_k": round(current_report.mean_precision_at_k - baseline_data.get("mean_precision_at_k", 0.0), 4),
        "mean_recall_at_k": round(current_report.mean_recall_at_k - baseline_data.get("mean_recall_at_k", 0.0), 4),
        "mean_reciprocal_rank": round(current_report.mean_reciprocal_rank - baseline_data.get("mean_reciprocal_rank", 0.0), 4),
        "mean_faithfulness": round(current_report.mean_faithfulness - baseline_data.get("mean_faithfulness", 0.0), 4),
    }

    regression_detected = False
    regression_reasons = []

    # 1. Metric Drop Checks
    critical_metrics = [
        ("mean_precision_at_k", "Retrieval Precision@K"),
        ("mean_reciprocal_rank", "Mean Reciprocal Rank (MRR)"),
        ("mean_faithfulness", "Answer Faithfulness (Hallucination Rate)"),
    ]

    for key, label in critical_metrics:
        drop = deltas.get(key, 0.0)
        if drop < -max_allowed_drop:
            regression_detected = True
            regression_reasons.append(
                f"Degradation in {label}: Dropped by {abs(drop):.4f} (exceeds tolerance of {max_allowed_drop})"
            )

    # 2. Per-Sample Regression & Fix Tracking
    baseline_sample_map = {
        s["sample_id"]: s["passed"] for s in baseline_data.get("sample_results", [])
    }

    new_failures = []
    new_fixes = []

    for curr in current_report.sample_results:
        sid = curr.sample_id
        if sid in baseline_sample_map:
            was_passed = baseline_sample_map[sid]
            is_passed = curr.passed

            if was_passed and not is_passed:
                regression_detected = True
                new_failures.append(f"{sid}: '{curr.query}' (Was PASSED in baseline, now FAILED)")
            elif not was_passed and is_passed:
                new_fixes.append(f"{sid}: '{curr.query}' (Was FAILED in baseline, now PASSED)")

    if new_failures:
        regression_reasons.append(f"{len(new_failures)} previously passing test cases failed in current run.")

    return RegressionAnalysis(
        baseline_timestamp=baseline_data.get("timestamp", "Unknown"),
        current_timestamp=current_report.timestamp,
        metric_deltas=deltas,
        regression_detected=regression_detected,
        regression_reasons=regression_reasons,
        new_failures=new_failures,
        new_fixes=new_fixes,
    )


def run_regression_check(baseline_path: str = DEFAULT_BASELINE_PATH) -> RegressionAnalysis:
    """Executes evaluation and runs comparison against baseline."""
    evaluator = RAGEvaluator()
    current_report = evaluator.run_evaluation()

    baseline_data = load_baseline(baseline_path)
    if not baseline_data:
        print(f"No existing baseline found at '{baseline_path}'.")
        print("Saving current evaluation run as the new Golden Baseline...")
        save_baseline(current_report, baseline_path)
        # Compare against self as baseline
        baseline_data = asdict(current_report)

    analysis = compare_with_baseline(current_report, baseline_data)
    return analysis


if __name__ == "__main__":
    print("--- Running RAGGuard AI Regression Engine ---")
    analysis = run_regression_check()

    print("\n==================================================")
    print("          RAG REGRESSION ANALYSIS REPORT          ")
    print("==================================================")
    print(f"Baseline Timestamp: {analysis.baseline_timestamp}")
    print(f"Current Timestamp:  {analysis.current_timestamp}")
    print(f"Regression Detected: {'YES - ALERT!' if analysis.regression_detected else 'NO (STABLE)'}")
    print("-" * 50)
    print("METRIC DELTAS (Current vs Baseline):")
    for metric, delta in analysis.metric_deltas.items():
        sign = "+" if delta > 0 else ""
        print(f"  {metric:25}: {sign}{delta}")
    print("-" * 50)

    if analysis.new_failures:
        print("NEW TEST REGRESSIONS (Previously passing, now failing):")
        for fail in analysis.new_failures:
            print(f"  [REGRESSION] {fail}")
        print()

    if analysis.new_fixes:
        print("NEW FIXES (Previously failing, now passing):")
        for fix in analysis.new_fixes:
            print(f"  [IMPROVED] {fix}")
        print()

    if analysis.regression_reasons:
        print("REGRESSION REASONS:")
        for reason in analysis.regression_reasons:
            print(f"  - {reason}")
    else:
        print("Status: Pipeline performance is stable across benchmark metrics.")
    print("==================================================")
