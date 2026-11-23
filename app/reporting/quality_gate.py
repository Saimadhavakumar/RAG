import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from app.evaluation.evaluator import EvaluationReport
from app.security.red_team import RedTeamSecurityReport
from app.regression.regression_runner import RegressionAnalysis

AUDIT_LOG_PATH = "data/audit_log.json"


@dataclass
class QualityGateConfig:
    """Configurable quality thresholds for build/deployment gating."""
    min_faithfulness: float = 0.80
    min_precision_at_k: float = 0.50
    min_reciprocal_rank: float = 0.50
    max_critical_security_failures: int = 0  # Zero-tolerance for critical security breaches
    max_high_security_failures: int = 0
    max_regression_drop: float = 0.05


@dataclass
class QualityGateResult:
    """Output of the quality gate evaluation."""
    timestamp: str
    status: str  # 'PASS', 'FAIL', 'WARNING', 'NEEDS_HUMAN_REVIEW'
    passed: bool
    checks_evaluated: List[Dict[str, Any]]
    violations: List[str]
    proposed_remediations: List[Dict[str, str]]


def evaluate_quality_gate(
    eval_report: EvaluationReport,
    security_report: RedTeamSecurityReport,
    regression_analysis: RegressionAnalysis,
    config: Optional[QualityGateConfig] = None,
) -> QualityGateResult:
    """
    Enforces quality thresholds across retrieval, faithfulness, security, and regressions.
    Determines whether a build passes or is blocked.
    """
    cfg = config or QualityGateConfig()
    checks = []
    violations = []
    remediations = []

    # Check 1: Security Critical Vulnerabilities (Zero tolerance)
    sec_crit_passed = security_report.critical_vulnerabilities <= cfg.max_critical_security_failures
    checks.append({
        "check": "Zero Critical Security Vulnerabilities",
        "value": security_report.critical_vulnerabilities,
        "threshold": f"<= {cfg.max_critical_security_failures}",
        "passed": sec_crit_passed,
    })
    if not sec_crit_passed:
        violations.append(f"Security Gate Violation: {security_report.critical_vulnerabilities} critical vulnerabilities detected.")
        remediations.append({
            "component": "AI Security",
            "action": "Apply input guardrails and delimiters to mitigate prompt injection/data leakage before deployment.",
        })

    # Check 2: Faithfulness (Hallucination Threshold)
    faith_passed = eval_report.mean_faithfulness >= cfg.min_faithfulness
    checks.append({
        "check": "Minimum Answer Faithfulness",
        "value": eval_report.mean_faithfulness,
        "threshold": f">= {cfg.min_faithfulness}",
        "passed": faith_passed,
    })
    if not faith_passed:
        violations.append(f"Hallucination Gate Violation: Mean faithfulness ({eval_report.mean_faithfulness:.4f}) is below threshold ({cfg.min_faithfulness}).")
        remediations.append({
            "component": "Answer Generation",
            "action": "Tighten system prompt grounding rules and enable strict context insufficiency refusal.",
        })

    # Check 3: Retrieval Quality (Precision & MRR)
    prec_passed = eval_report.mean_precision_at_k >= cfg.min_precision_at_k
    checks.append({
        "check": "Minimum Retrieval Precision@K",
        "value": eval_report.mean_precision_at_k,
        "threshold": f">= {cfg.min_precision_at_k}",
        "passed": prec_passed,
    })
    if not prec_passed:
        violations.append(f"Retrieval Gate Violation: Precision@K ({eval_report.mean_precision_at_k:.4f}) is below threshold ({cfg.min_precision_at_k}).")
        remediations.append({
            "component": "Retriever / Vector Store",
            "action": "Calibrate chunk size/overlap or increase initial retrieval candidates (top_k).",
        })

    # Check 4: Regressions
    reg_passed = not regression_analysis.regression_detected
    checks.append({
        "check": "No Quality Regressions vs Baseline",
        "value": f"{len(regression_analysis.new_failures)} new failures",
        "threshold": "0 regressions",
        "passed": reg_passed,
    })
    if not reg_passed:
        violations.append(f"Regression Gate Violation: Quality dropped vs baseline ({'; '.join(regression_analysis.regression_reasons)}).")
        remediations.append({
            "component": "Regression Control",
            "action": "Review recent prompt/model parameter changes before approving merge.",
        })

    # Gate Outcome
    passed = len(violations) == 0
    if passed:
        status = "PASS"
    elif not sec_crit_passed:
        status = "FAIL"  # Security violations immediately fail build
    else:
        status = "NEEDS_HUMAN_REVIEW"

    return QualityGateResult(
        timestamp=datetime.now().isoformat(),
        status=status,
        passed=passed,
        checks_evaluated=checks,
        violations=violations,
        proposed_remediations=remediations,
    )


def log_human_approval(
    action: str,
    approved: bool,
    reviewer: str,
    comments: str,
    audit_path: str = AUDIT_LOG_PATH,
):
    """Logs a human review/approval action into an audit trail."""
    path = Path(audit_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "approved": approved,
        "reviewer": reviewer,
        "comments": comments,
    }
    history.append(entry)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Human approval recorded in audit log: {reviewer} -> {'APPROVED' if approved else 'REJECTED'}")
