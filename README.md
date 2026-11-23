# 🛡️ RAGGuard AI

> **Autonomous RAG Evaluation, QA Automation & Security Testing Platform**  
> An industry-grade platform to evaluate, test, monitor, and secure Retrieval-Augmented Generation (RAG) applications against retrieval failures, hallucinations, prompt injection attacks, sensitive data leakage, and regressions.

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python)](https://www.python.org/)
[![Pytest Tests](https://img.shields.io/badge/Pytest-22%2F22%20Passed-success?logo=pytest)](https://docs.pytest.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Executive Summary & Problem Statement

Traditional software testing relies heavily on deterministic expected outputs. Generative AI and RAG applications introduce entirely new testing challenges:

1. **Non-Deterministic Outputs:** The same question can produce slightly different responses.
2. **Retrieval Failures:** Semantically similar documents are not always relevant documents.
3. **Hallucinations & Ungrounded Answers:** Models can fabricate answers that sound convincing but are unsupported by retrieved context.
4. **Prompt Injection & Jailbreak Attacks:** Attackers can manipulate model behavior through direct or indirect prompt injection inside documents.
5. **Sensitive Information Disclosure:** Credentials, API keys, or private records can leak into generated responses.
6. **Silent Regressions:** Tweak a prompt, change an embedding model, or alter chunk size, and performance can silently collapse.

**RAGGuard AI** solves these challenges through measurable evaluation metrics, automated QA suites, adversarial red teaming, regression detection, and configurable quality gates.

---

## 🏛️ Two-System Architecture

RAGGuard AI enforces strict architectural separation between the application being tested and its testing framework:

```
┌─────────────────────────────────────────────────────────────┐
│          SYSTEM A: Target RAG Application Under Test        │
│                                                             │
│  [PDFs] ──> [Ingestion & Chunking] ──> [Embeddings]         │
│                                              │              │
│                                              ▼              │
│  [User Query] ──> [Retriever] ──> [ChromaDB Vector Store]   │
│                         │                                   │
│                         ▼ (Top 6 Candidates)                │
│                 [Cross-Encoder Reranker]                    │
│                         │                                   │
│                         ▼ (Top 3 Refined)                   │
│               [Grounded Generator] ──> [RAG Response]       │
└──────────────────────────────┬──────────────────────────────┘
                               │ Audited & Tested by
┌──────────────────────────────▼──────────────────────────────┐
│        SYSTEM B: RAGGuard QA & Security Testing Platform     │
│                                                             │
│  ├── 1. RAG Evaluation Engine (Precision, Recall, MRR,      │
│  │      NDCG, Faithfulness / Hallucination Detection)       │
│  ├── 2. Pytest QA Automation (22 Unit & Integration Tests)  │
│  ├── 3. Promptfoo CLI Integration (Prompt & Model Sweeps)   │
│  ├── 4. AI Red Teaming (Prompt Injection & Leakage Scans)   │
│  ├── 5. Regression Engine (Deltas vs. Golden Baseline)      │
│  ├── 6. Quality Gates & Human Approval (Build Enforcement)  │
│  ├── 7. Automated Reports (Interactive HTML + Machine JSON) │
│  ├── 8. FastAPI REST Backend (/api/v1/...)                  │
│  └── 9. Streamlit Dashboard (Multi-tab Command Center)      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Differentiators

| Capability | Basic RAG Demos | RAGGuard AI Platform |
| :--- | :--- | :--- |
| **Retrieval Architecture** | Single-stage vector search | **Two-Stage Retrieval** (Bi-Encoder search + Cross-Encoder reranking) |
| **Hallucination Detection** | None | **Automated Faithfulness Scoring** (Token-level claim grounding) |
| **Out-of-Domain Handling** | Tries to answer and hallucinates | **Context Insufficiency Checks** (Truthful refusal enforcement) |
| **AI Security Testing** | Ignored | **OWASP Top 10 Red Teaming** (Direct/indirect injections, jailbreaks) |
| **Regression Tracking** | Manual spot-checking | **Golden Baseline Comparison** with metric delta degradation alerts |
| **Build Gating** | None | **Automated Quality Gate** (`PASS`/`FAIL` blocking CI/CD pipelines) |
| **Human in the Loop** | None | **Audit-logged remediation sign-offs** ([data/audit_log.json](file:///e:/RAGGUARD/data/audit_log.json)) |

---

## 🛠️ Technology Stack

- **Core:** Python 3.11+, Pydantic v2, `python-dotenv`
- **RAG & Vector Database:** LangChain, ChromaDB, `all-MiniLM-L6-v2` (Local CPU embeddings) / OpenAI `text-embedding-3-small`
- **Reranker:** `sentence-transformers` Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) / Cohere Rerank API
- **QA Automation:** Pytest (22 automated test cases)
- **LLM Evaluation:** Custom IR metrics, token F1 correctness, Promptfoo integration
- **AI Security:** Real-time data leakage regex scanner, adversarial red team suite
- **Backend & Delivery:** FastAPI, Uvicorn, Streamlit, GitHub Actions CI/CD

---

## 📥 Quickstart & Setup (Windows CMD)

### 1. Clone & Activate Virtual Environment
```cmd
git clone https://github.com/your-username/RAGGuard-AI.git
cd RAGGuard-AI

python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Dependencies
```cmd
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```cmd
copy .env.example .env
```
*(By default, RAGGuard AI is configured to run 100% locally and free on CPU using `all-MiniLM-L6-v2` and our mock generator, with optional OpenAI/Cohere API support).*

---

## 🧪 Execution & Verification Commands

### Run Full Pytest Test Suite (22 Tests)
```cmd
pytest
```

### Run RAG Evaluation Engine (Phase 8)
Computes Precision@3, Recall@3, MRR, NDCG@3, and Faithfulness across benchmark queries:
```cmd
python -m app.evaluation.evaluator
```

### Run AI Red Teaming Security Audit (Phase 11)
Attacks the RAG application with prompt injections, jailbreaks, and credential harvesting:
```cmd
python -m app.security.red_team
```

### Run Regression Engine vs. Golden Baseline (Phase 12)
Detects quality degradation against `data/baselines/baseline_v1.json`:
```cmd
python -m app.regression.regression_runner
```

### Generate Full QA & Security Reports (Phases 13 & 14)
Generates machine JSON and opens the HTML visual report:
```cmd
python -m app.reporting.report_generator
```
*(Open [reports/latest_report.html](file:///e:/RAGGUARD/reports/latest_report.html) in your browser).*

### Launch Interactive Streamlit Dashboard (Phase 16)
```cmd
streamlit run dashboard\app.py
```
*(Access web command center at `http://localhost:8501`).*

### Launch FastAPI Swagger Backend (Phase 15)
```cmd
uvicorn app.api.main:app --reload
```
*(Access interactive Swagger API docs at `http://127.0.0.1:8000/docs`).*

---

## 📊 Complete Architecture Documentation

Detailed sequence diagrams and dataflow charts are available in [docs/architecture/ARCHITECTURE.md](file:///e:/RAGGUARD/docs/architecture/ARCHITECTURE.md):
1. Overall System Architecture
2. Document Ingestion & Storage Pipeline
3. Two-Stage RAG Query Execution Flow
4. QA Automation Execution Flow (Pytest)
5. Promptfoo Evaluation Workflow
6. AI Red Teaming & Threat Model Flow
7. Regression Testing Workflow
8. Quality Gate & Human Approval Workflow
9. CI/CD GitHub Actions Pipeline

---

## 🎓 Technical Interview Defense & Key Design Decisions

### Q1: Why did you implement Two-Stage Retrieval instead of standard vector search?
> **Answer:** Standard vector search uses bi-encoders that encode queries and passages into separate vectors. While blazingly fast ($O(1)$ to $O(\log N)$ in ChromaDB HNSW), bi-encoders miss token-level interactions. In our tests, vector retrieval placed the most direct answer at Rank 3. Our Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) performed joint query-passage self-attention and promoted it to Rank 1 ($+3.78$ relevance score), resolving the *"Lost in the Middle"* problem.

### Q2: How does RAGGuard AI mathematically detect hallucinations?
> **Answer:** We built a deterministic token-level Faithfulness metric (`calculate_faithfulness`). It splits the model answer into individual claims, strips boilerplate/citations, and calculates the content-token recall against the retrieved context. Any claim with overlap below $0.35$ is flagged as an unsupported claim / hallucination candidate.

### Q3: How does your Quality Gate stop bad deployments in CI/CD?
> **Answer:** In `.github/workflows/ci.yml`, the pipeline runs `app.reporting.report_generator`. The Quality Gate evaluates 4 rules:
> 1. Zero critical security vulnerabilities (zero-tolerance).
> 2. Minimum Faithfulness $\ge 0.80$.
> 3. Minimum Precision@3 $\ge 0.50$.
> 4. Zero metric drops $> 5\%$ vs. our Golden Baseline.
> If any critical threshold fails, the build exits with an error code and blocks deployment.

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
