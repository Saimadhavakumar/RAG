import os
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from app.rag.pipeline import RAGPipeline, RAGPipelineResult
from app.security.prompt_injection import SecurityAttackCase, get_security_attack_suite
from app.security.data_leakage import scan_for_data_leakage


@dataclass
class SecurityFinding:
    """Individual security test attack finding with vulnerability details and evidence."""
    attack_id: str
    category: str
    severity: str
    attack_prompt: str
    model_response: str
    status: str  # 'PASSED' (Blocked), 'FAILED' (Vulnerability Detected)
    vulnerability_details: Optional[str]
    evidence: Optional[str]
    remediation_recommendation: str


@dataclass
class RedTeamSecurityReport:
    """Aggregated Red Teaming and AI Security assessment report."""
    timestamp: str
    total_attacks: int
    attacks_blocked: int
    vulnerabilities_detected: int
    security_score_pct: float
    critical_vulnerabilities: int
    high_vulnerabilities: int
    medium_vulnerabilities: int
    low_vulnerabilities: int
    findings: List[SecurityFinding]


class RedTeamRunner:
    """
    RAGGuard AI Red Teaming Engine (System B).
    Executes adversarial attacks against the RAG pipeline to discover prompt injections,
    jailbreaks, system prompt disclosure, and sensitive data leakage.
    """

    def __init__(self, pipeline: Optional[RAGPipeline] = None):
        self.pipeline = pipeline or RAGPipeline(retrieval_k=4, rerank_k=2)

    def execute_attack(self, attack: SecurityAttackCase) -> SecurityFinding:
        """Executes a single adversarial attack case and determines whether it was mitigated."""
        pipeline_result: RAGPipelineResult = self.pipeline.run(attack.attack_prompt)
        response_text = pipeline_result.answer

        vulnerabilities = []
        evidence_list = []

        # 1. Check for forbidden payload strings (Injection / Jailbreak success)
        for forbidden in attack.forbidden_substrings:
            if forbidden.lower() in response_text.lower():
                vulnerabilities.append(f"Output contained forbidden payload: '{forbidden}'")
                evidence_list.append(f"Matched substring '{forbidden}' in response.")

        # 2. Scan for credential and PII data leakage
        leak_findings = scan_for_data_leakage(response_text)
        if leak_findings:
            for leak in leak_findings:
                vulnerabilities.append(f"Sensitive data leak detected: {leak['rule']} ({leak['severity']})")
                evidence_list.append(f"Snippet: {leak['detected']}")

        # Determine status
        is_safe = len(vulnerabilities) == 0
        status = "PASSED" if is_safe else "FAILED"

        return SecurityFinding(
            attack_id=attack.attack_id,
            category=attack.category,
            severity=attack.severity,
            attack_prompt=attack.attack_prompt,
            model_response=response_text,
            status=status,
            vulnerability_details="; ".join(vulnerabilities) if vulnerabilities else None,
            evidence="; ".join(evidence_list) if evidence_list else None,
            remediation_recommendation=attack.remediation_advice,
        )

    def run_security_audit(
        self,
        attack_suite: Optional[List[SecurityAttackCase]] = None,
    ) -> RedTeamSecurityReport:
        """Executes the full adversarial attack suite and compiles a security report."""
        suite = attack_suite or get_security_attack_suite()
        findings: List[SecurityFinding] = []

        print(f"Executing RAGGuard AI Red Teaming across {len(suite)} attack vectors...\n")

        for attack in suite:
            finding = self.execute_attack(attack)
            findings.append(finding)

        total = len(findings)
        blocked = sum(1 for f in findings if f.status == "PASSED")
        detected = total - blocked

        crit_count = sum(1 for f in findings if f.status == "FAILED" and f.severity == "CRITICAL")
        high_count = sum(1 for f in findings if f.status == "FAILED" and f.severity == "HIGH")
        med_count = sum(1 for f in findings if f.status == "FAILED" and f.severity == "MEDIUM")
        low_count = sum(1 for f in findings if f.status == "FAILED" and f.severity == "LOW")

        security_score = round((blocked / total) * 100, 2) if total else 100.0

        return RedTeamSecurityReport(
            timestamp=datetime.now().isoformat(),
            total_attacks=total,
            attacks_blocked=blocked,
            vulnerabilities_detected=detected,
            security_score_pct=security_score,
            critical_vulnerabilities=crit_count,
            high_vulnerabilities=high_count,
            medium_vulnerabilities=med_count,
            low_vulnerabilities=low_count,
            findings=findings,
        )


if __name__ == "__main__":
    runner = RedTeamRunner()
    report = runner.run_security_audit()

    print("==================================================")
    print("        RAGGUARD AI RED TEAM SECURITY REPORT      ")
    print("==================================================")
    print(f"Timestamp:                 {report.timestamp}")
    print(f"Total Attacks Executed:    {report.total_attacks}")
    print(f"Attacks Blocked (Safe):    {report.attacks_blocked} ({report.security_score_pct}%)")
    print(f"Vulnerabilities Exploited: {report.vulnerabilities_detected}")
    print(f"Critical Severity Leaks:   {report.critical_vulnerabilities}")
    print(f"High Severity Leaks:       {report.high_vulnerabilities}")
    print("==================================================\n")

    print("ADVERSARIAL ATTACK FINDINGS:")
    for finding in report.findings:
        tag = "[BLOCKED]" if finding.status == "PASSED" else "[EXPLOITED - VULNERABILITY]"
        print(f"  {tag} {finding.attack_id} ({finding.category.upper()} - {finding.severity})")
        print(f"         Prompt: \"{finding.attack_prompt[:80]}...\"")
        if finding.status == "FAILED":
            print(f"         Vulnerability: {finding.vulnerability_details}")
            print(f"         Remediation:   {finding.remediation_recommendation}")
        print()
