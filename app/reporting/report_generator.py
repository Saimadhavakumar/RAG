import os
import json
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

from app.evaluation.evaluator import RAGEvaluator, EvaluationReport
from app.security.red_team import RedTeamRunner, RedTeamSecurityReport
from app.regression.regression_runner import run_regression_check, RegressionAnalysis
from app.reporting.quality_gate import evaluate_quality_gate, QualityGateResult


def generate_html_report(
    eval_report: EvaluationReport,
    sec_report: RedTeamSecurityReport,
    reg_analysis: RegressionAnalysis,
    gate_result: QualityGateResult,
) -> str:
    """Renders a self-contained, responsive HTML test report for RAGGuard AI."""
    status_color = "#10b981" if gate_result.status == "PASS" else ("#ef4444" if gate_result.status == "FAIL" else "#f59e0b")

    checks_html = "".join([
        f"<tr><td>{c['check']}</td><td><b>{c['value']}</b></td><td>{c['threshold']}</td>"
        f"<td><span class='badge {'badge-pass' if c['passed'] else 'badge-fail'}'>{'PASS' if c['passed'] else 'FAIL'}</span></td></tr>"
        for c in gate_result.checks_evaluated
    ])

    samples_html = "".join([
        f"<tr><td>{s.sample_id}</td><td>{s.query}</td><td>{s.precision_at_k:.2f}</td><td>{s.recall_at_k:.2f}</td>"
        f"<td>{s.faithfulness:.2f}</td><td><span class='badge {'badge-pass' if s.passed else 'badge-fail'}'>{'PASS' if s.passed else 'FAIL'}</span></td>"
        f"<td><small>{s.failure_reason or 'None'}</small></td></tr>"
        for s in eval_report.sample_results
    ])

    sec_html = "".join([
        f"<tr><td><b>{f.attack_id}</b></td><td>{f.category.upper()}</td><td>{f.severity}</td>"
        f"<td><span class='badge {'badge-pass' if f.status == 'PASSED' else 'badge-fail'}'>{'BLOCKED' if f.status == 'PASSED' else 'EXPLOITED'}</span></td>"
        f"<td><small>{f.remediation_recommendation}</small></td></tr>"
        for f in sec_report.findings
    ])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>RAGGuard AI — Test & Security Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
    .container {{ max-width: 1100px; margin: 0 auto; }}
    .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 16px; margin-bottom: 24px; }}
    .title {{ font-size: 26px; font-weight: 700; color: #38bdf8; }}
    .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }}
    .card {{ background: #1e293b; border-radius: 8px; padding: 16px; border: 1px solid #334155; }}
    .card-title {{ font-size: 13px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }}
    .card-value {{ font-size: 28px; font-weight: 700; margin-top: 8px; }}
    .gate-banner {{ background: #1e293b; border-left: 6px solid {status_color}; padding: 18px; border-radius: 6px; margin-bottom: 24px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; background: #1e293b; border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid #334155; font-size: 14px; }}
    th {{ background: #0f172a; color: #94a3b8; font-weight: 600; }}
    .badge {{ padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; text-transform: uppercase; }}
    .badge-pass {{ background: #065f46; color: #34d399; }}
    .badge-fail {{ background: #7f1d1d; color: #f87171; }}
    .section-title {{ font-size: 20px; font-weight: 600; margin: 28px 0 12px; color: #e2e8f0; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <div class="title">RAGGuard AI — Quality & Security Report</div>
        <div style="color: #94a3b8; font-size: 14px; margin-top: 4px;">Execution Timestamp: {gate_result.timestamp}</div>
      </div>
      <div>
        <span class="badge" style="background: {status_color}; color: #fff; font-size: 14px; padding: 8px 16px;">
          QUALITY GATE: {gate_result.status}
        </span>
      </div>
    </div>

    <div class="card-grid">
      <div class="card">
        <div class="card-title">Pass Rate</div>
        <div class="card-value" style="color: #38bdf8;">{eval_report.pass_rate_pct}%</div>
      </div>
      <div class="card">
        <div class="card-title">Mean Faithfulness</div>
        <div class="card-value" style="color: #34d399;">{eval_report.mean_faithfulness:.2f}</div>
      </div>
      <div class="card">
        <div class="card-title">Retrieval Precision@3</div>
        <div class="card-value" style="color: #818cf8;">{eval_report.mean_precision_at_k:.2f}</div>
      </div>
      <div class="card">
        <div class="card-title">Security Mitigations</div>
        <div class="card-value" style="color: #f43f5e;">{sec_report.security_score_pct}%</div>
      </div>
    </div>

    <div class="section-title">1. Quality Gate Enforcement</div>
    <table>
      <thead><tr><th>Gate Rule</th><th>Measured Value</th><th>Required Threshold</th><th>Outcome</th></tr></thead>
      <tbody>{checks_html}</tbody>
    </table>

    <div class="section-title">2. AI Red Teaming & Security Findings</div>
    <table>
      <thead><tr><th>Attack ID</th><th>Category</th><th>Severity</th><th>Status</th><th>Remediation Advice</th></tr></thead>
      <tbody>{sec_html}</tbody>
    </table>

    <div class="section-title">3. Detailed Benchmark Test Cases</div>
    <table>
      <thead><tr><th>ID</th><th>Query</th><th>Prec@3</th><th>Recall@3</th><th>Faithfulness</th><th>Status</th><th>Failure Explanation</th></tr></thead>
      <tbody>{samples_html}</tbody>
    </table>
  </div>
</body>
</html>"""
    return html


def run_full_report_generation(reports_dir: str = "reports") -> str:
    """
    Executes the entire RAGGuard QA & Security assessment workflow,
    enforces quality gates, and outputs JSON & HTML reports.
    """
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Run RAG Evaluation
    print("Step 1/4: Running RAG Evaluation Suite...")
    evaluator = RAGEvaluator()
    eval_report = evaluator.run_evaluation()

    # 2. Run AI Red Teaming Security Audit
    print("Step 2/4: Executing AI Red Teaming Security Audit...")
    red_team = RedTeamRunner()
    sec_report = red_team.run_security_audit()

    # 3. Run Regression Engine vs Golden Baseline
    print("Step 3/4: Comparing against Golden Regression Baseline...")
    reg_analysis = run_regression_check()

    # 4. Enforce Quality Gates
    print("Step 4/4: Evaluating Quality Gate Rules...")
    gate_result = evaluate_quality_gate(eval_report, sec_report, reg_analysis)

    # Compile unified JSON payload
    payload = {
        "execution_timestamp": datetime.now().isoformat(),
        "quality_gate": asdict(gate_result),
        "evaluation_summary": asdict(eval_report),
        "security_summary": asdict(sec_report),
        "regression_summary": asdict(reg_analysis),
    }

    # Save JSON files
    json_path = os.path.join(reports_dir, f"ragguard_report_{timestamp_slug}.json")
    latest_json = os.path.join(reports_dir, "latest_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Save HTML files
    html_content = generate_html_report(eval_report, sec_report, reg_analysis, gate_result)
    html_path = os.path.join(reports_dir, f"ragguard_report_{timestamp_slug}.html")
    latest_html = os.path.join(reports_dir, "latest_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    with open(latest_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print("\n==================================================")
    print("        RAGGUARD AI REPORT GENERATION COMPLETE     ")
    print("==================================================")
    print(f"Quality Gate Outcome: {gate_result.status}")
    print(f"JSON Report Saved:    {json_path}")
    print(f"HTML Report Saved:    {html_path}")
    print(f"Latest Shortcut:      {latest_html}")
    print("==================================================")
    return html_path


if __name__ == "__main__":
    run_full_report_generation()
