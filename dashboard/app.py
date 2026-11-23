import os
import sys
import json
from pathlib import Path
from datetime import datetime

import streamlit as st

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag.pipeline import RAGPipeline
from app.reporting.report_generator import run_full_report_generation
from app.reporting.quality_gate import log_human_approval, AUDIT_LOG_PATH
from app.evaluation.metrics import calculate_faithfulness

st.set_page_config(
    page_title="RAGGuard AI — QA & Security Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-pass {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 12px;
    }
    .badge-fail {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 12px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)


def load_latest_report():
    report_file = Path("reports/latest_report.json")
    if not report_file.exists():
        return None
    try:
        with open(report_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# Sidebar
st.sidebar.title("🛡️ RAGGuard AI")
st.sidebar.markdown("**Autonomous RAG QA & Security Platform**")
st.sidebar.divider()

st.sidebar.subheader("System Architecture")
st.sidebar.info("""
- **Target RAG:** Ingestion + ChromaDB
- **Embeddings:** Local CPU (`all-MiniLM-L6-v2`)
- **Reranker:** Cross-Encoder (`ms-marco`)
- **LLM Engine:** Grounded Generator
""")

if st.sidebar.button("⚡ Run Full QA & Security Audit", use_container_width=True):
    with st.spinner("Executing Evaluation, Red Teaming, Regression, and Quality Gates..."):
        run_full_report_generation()
        st.sidebar.success("Audit Complete! Refreshing...")
        st.rerun()

st.sidebar.divider()
st.sidebar.subheader("📂 Document Management")
with st.sidebar.expander("Upload & Index New PDF", expanded=False):
    uploaded_file = st.file_uploader("Select PDF file:", type=["pdf"])
    if uploaded_file is not None:
        if st.button("📥 Ingest Document", use_container_width=True):
            save_dir = Path("data/documents")
            save_dir.mkdir(parents=True, exist_ok=True)
            saved_path = save_dir / uploaded_file.name
            with open(saved_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner(f"Ingesting {uploaded_file.name} into ChromaDB..."):
                from app.rag.ingestion import load_documents, split_documents
                from app.rag.vector_store import index_documents
                raw_docs = load_documents(str(save_dir))
                chunks = split_documents(raw_docs)
                indexed_count = index_documents(chunks)
                st.success(f"Successfully processed {len(raw_docs)} PDFs ({len(chunks)} chunks indexed)!")

st.sidebar.divider()
st.sidebar.caption("RAGGuard AI v3.0 | Industrial QA Automation")

# Main Content
latest_data = load_latest_report()

st.title("🛡️ RAGGuard AI Dashboard")
st.markdown("Automated RAG Evaluation, Quality Gates, AI Red Teaming, and Regression Intelligence.")

tab_overview, tab_sandbox, tab_eval, tab_security, tab_regression, tab_audit = st.tabs([
    "📊 Overview & Quality Gate",
    "🔍 Live RAG Sandbox",
    "📈 Evaluation Metrics",
    "🚨 AI Red Teaming",
    "📉 Regression Tracking",
    "✍️ Human Approval & Audit",
])

# -------------------------------------------------------------
# TAB 1: OVERVIEW & QUALITY GATE
# -------------------------------------------------------------
with tab_overview:
    if not latest_data:
        st.warning("No audit report found. Click 'Run Full QA & Security Audit' in the sidebar to generate one.")
    else:
        gate = latest_data.get("quality_gate", {})
        status = gate.get("status", "UNKNOWN")
        passed = gate.get("passed", False)

        # Quality Gate Banner
        if status == "PASS":
            st.success(f"### ✅ QUALITY GATE: {status}\nAll requirements satisfied. System is verified for production deployment.")
        elif status == "FAIL":
            st.error(f"### 🛑 QUALITY GATE: {status}\nCritical violations detected. Deployment blocked.")
        else:
            st.warning(f"### ⚠️ QUALITY GATE: {status}\nQuality threshold warnings detected. Human approval required.")

        # Top Metric Cards
        eval_sum = latest_data.get("evaluation_summary", {})
        sec_sum = latest_data.get("security_summary", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall Pass Rate", f"{eval_sum.get('pass_rate_pct', 0)}%")
        c2.metric("Mean Faithfulness", f"{eval_sum.get('mean_faithfulness', 0):.2f}")
        c3.metric("Retrieval Precision@3", f"{eval_sum.get('mean_precision_at_k', 0):.2f}")
        c4.metric("Security Mitigations", f"{sec_sum.get('security_score_pct', 0)}%")

        st.divider()

        # Gate Checks Table
        st.subheader("Enforced Quality Gate Rules")
        checks = gate.get("checks_evaluated", [])
        if checks:
            cols = st.columns([3, 2, 2, 2])
            cols[0].markdown("**Rule Description**")
            cols[1].markdown("**Measured Value**")
            cols[2].markdown("**Required Threshold**")
            cols[3].markdown("**Gate Result**")

            for check in checks:
                cols[0].write(check.get("check"))
                cols[1].write(f"`{check.get('value')}`")
                cols[2].write(check.get("threshold"))
                outcome = "PASS" if check.get("passed") else "FAIL"
                color = "green" if check.get("passed") else "red"
                cols[3].markdown(f":{color}[**{outcome}**]")

        # Proposed Remediations
        remediations = gate.get("proposed_remediations", [])
        if remediations:
            st.subheader("Recommended Remediation Actions")
            for rem in remediations:
                st.info(f"**Component: {rem.get('component')}** — {rem.get('action')}")

# -------------------------------------------------------------
# TAB 2: LIVE RAG SANDBOX
# -------------------------------------------------------------
with tab_sandbox:
    st.subheader("Test System A (RAG Pipeline) Interactively")
    st.markdown("Type a question to see real-time retrieval, cross-encoder reranking, and grounded answer generation.")

    sample_questions = [
        "What is continuous integration and how is code merged?",
        "What are the main stages in the DevOps life cycle?",
        "What is quantum entanglement and teleportation?",
    ]
    selected_sample = st.selectbox("Or choose a sample test query:", [""] + sample_questions)

    user_query = st.text_input("Enter your query:", value=selected_sample)

    if st.button("🚀 Execute RAG Query", use_container_width=True):
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Executing retrieval, reranking, and generation..."):
                pipeline = RAGPipeline(retrieval_k=6, rerank_k=3)
                result = pipeline.run(user_query)

                st.subheader("Generated Answer")
                st.write(result.answer)

                # Real-time Faithfulness calculation
                context_str = "\n".join(c.content for c in result.reranked_chunks)
                faith_score, unsupported = calculate_faithfulness(result.answer, context_str)

                c1, c2, c3 = st.columns(3)
                c1.metric("Latency", f"{result.total_latency_seconds}s")
                c2.metric("Faithfulness Score", f"{faith_score:.2f}")
                c3.metric("Sufficient Context", "YES" if result.has_sufficient_context else "NO (Refusal)")

                if unsupported:
                    st.error(f"⚠️ Hallucination Alert: Unsupported claims detected: {unsupported}")

                # Two-Stage Retrieval Breakdown
                st.divider()
                st.subheader("Two-Stage Retrieval Breakdown")

                st.markdown("**Stage 2: Cross-Encoder Refined Chunks (Fed to LLM)**")
                for c in result.reranked_chunks:
                    with st.expander(f"Rank {c.reranked_rank} | Cross-Encoder Score: {c.rerank_score:+.4f} | Source: {os.path.basename(c.source)} (Page {c.page})"):
                        st.caption(f"Original Vector Rank: {c.original_rank} | Distance: {c.retrieval_score:.4f}")
                        st.write(c.content)

# -------------------------------------------------------------
# TAB 3: EVALUATION METRICS
# -------------------------------------------------------------
with tab_eval:
    st.subheader("Automated RAG Evaluation Metrics (Phase 8)")
    if latest_data:
        eval_sum = latest_data.get("evaluation_summary", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision@3", f"{eval_sum.get('mean_precision_at_k', 0):.4f}")
        c2.metric("Recall@3", f"{eval_sum.get('mean_recall_at_k', 0):.4f}")
        c3.metric("Mean Reciprocal Rank (MRR)", f"{eval_sum.get('mean_reciprocal_rank', 0):.4f}")
        c4.metric("NDCG@3", f"{eval_sum.get('mean_ndcg_at_k', 0):.4f}")

        st.divider()
        st.subheader("Benchmark Sample Results")
        samples = eval_sum.get("sample_results", [])
        for s in samples:
            status = "✅ PASS" if s.get("passed") else "❌ FAIL"
            with st.expander(f"{status} | {s.get('sample_id')}: {s.get('query')}"):
                c1, c2, c3 = st.columns(3)
                c1.write(f"**Precision@3:** {s.get('precision_at_k')}")
                c2.write(f"**Recall@3:** {s.get('recall_at_k')}")
                c3.write(f"**Faithfulness:** {s.get('faithfulness')}")
                st.write(f"**Generated Answer:** {s.get('generated_answer')}")
                st.write(f"**Ground Truth:** {s.get('ground_truth_answer')}")
                if s.get("failure_reason"):
                    st.error(f"Failure Explanation: {s.get('failure_reason')}")

# -------------------------------------------------------------
# TAB 4: AI RED TEAMING
# -------------------------------------------------------------
with tab_security:
    st.subheader("AI Security & Red Teaming (Phases 10 & 11)")
    if latest_data:
        sec_sum = latest_data.get("security_summary", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Attacks Executed", sec_sum.get("total_attacks", 0))
        c2.metric("Attacks Blocked", sec_sum.get("attacks_blocked", 0))
        c3.metric("Vulnerabilities Detected", sec_sum.get("vulnerabilities_detected", 0))

        st.divider()
        st.subheader("Adversarial Findings")
        findings = sec_sum.get("findings", [])
        for f in findings:
            is_blocked = f.get("status") == "PASSED"
            tag = "🛡️ BLOCKED" if is_blocked else "🚨 VULNERABILITY DETECTED"
            with st.expander(f"{tag} | {f.get('attack_id')} ({f.get('category').upper()} - {f.get('severity')})"):
                st.write(f"**Adversarial Prompt:** `{f.get('attack_prompt')}`")
                st.write(f"**Model Response:** {f.get('model_response')}")
                st.info(f"**Remediation Advice:** {f.get('remediation_recommendation')}")

# -------------------------------------------------------------
# TAB 5: REGRESSION TRACKING
# -------------------------------------------------------------
with tab_regression:
    st.subheader("Regression Intelligence (Phase 12)")
    if latest_data:
        reg_sum = latest_data.get("regression_summary", {})
        is_reg = reg_sum.get("regression_detected", False)
        if is_reg:
            st.error("🚨 REGRESSION ALERT DETECTED vs Golden Baseline!")
        else:
            st.success("✅ System performance is stable with zero regressions vs Golden Baseline.")

        st.divider()
        st.subheader("Metric Deltas vs Golden Baseline")
        deltas = reg_sum.get("metric_deltas", {})
        for metric, delta in deltas.items():
            sign = "+" if delta > 0 else ""
            st.write(f"- **{metric}:** `{sign}{delta}`")

# -------------------------------------------------------------
# TAB 6: HUMAN APPROVAL & AUDIT LOG
# -------------------------------------------------------------
with tab_audit:
    st.subheader("Human-in-the-Loop Remediation Approval (Part 13)")
    st.markdown("Record human sign-off on remediation suggestions before promoting prompt or model changes to production.")

    with st.form("approval_form"):
        action = st.text_input("Action to Review:", "Approve Production Deployment of DevOps RAG Pipeline v1")
        reviewer = st.text_input("Reviewer Name:", "QA_Engineer_Alex")
        decision = st.radio("Decision:", ["Approve", "Reject"])
        comments = st.text_area("Audit Comments / Evidence:", "All 22 Pytest tests passed, zero security leaks, and Quality Gate passed.")
        submitted = st.form_submit_button("Submit Audit Record")

        if submitted:
            approved = decision == "Approve"
            log_human_approval(action=action, approved=approved, reviewer=reviewer, comments=comments)
            st.success(f"Audit record logged successfully: {reviewer} -> {decision}!")
            st.rerun()

    st.divider()
    st.subheader("Permanent Audit Trail")
    audit_path = Path(AUDIT_LOG_PATH)
    if audit_path.exists():
        try:
            with open(audit_path, "r", encoding="utf-8") as f:
                logs = json.load(f)
            st.table(logs)
        except Exception:
            st.write("Audit log is currently empty.")
    else:
        st.write("No audit entries recorded yet.")
