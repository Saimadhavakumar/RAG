# RAGGuard AI — Technical Architecture & System Diagrams

This document outlines the complete architectural design, execution workflows, and testing lifecycles of **RAGGuard AI**, detailing the separation between **System A (Target RAG Application)** and **System B (QA & Security Testing Platform)**.

---

## 1. Overall System Architecture

```mermaid
graph TB
    subgraph SYSTEM_A["System A: Target RAG Application Under Test"]
        Docs["PDF Documents<br>(data/documents/)"] --> Ingestion["Document Ingestion<br>(app/rag/ingestion.py)"]
        Ingestion --> Chunking["Recursive Chunking<br>(chunk_size=800, overlap=150)"]
        Chunking --> Embeddings["Embedding Engine<br>(all-MiniLM-L6-v2 / OpenAI)"]
        Embeddings --> ChromaDB[("ChromaDB Vector Store<br>(data/chroma_db/)")]
        
        UserQuery["User Query"] --> Retriever["Semantic Retriever<br>(app/rag/retriever.py)"]
        ChromaDB -.->|Top-6 Candidates| Retriever
        Retriever --> Reranker["Cross-Encoder Reranker<br>(ms-marco-MiniLM-L-6-v2)"]
        Reranker -->|Top-3 Best Chunks| Generator["Grounded Generator<br>(app/rag/generator.py)"]
        Generator --> RAGResponse["Grounded Response + Sources"]
    end

    subgraph SYSTEM_B["System B: RAGGuard QA & Security Testing Platform"]
        TestDataset[("Benchmark Dataset<br>(app/evaluation/dataset.py)")] --> Evaluator["RAG Evaluation Engine<br>(Precision, Recall, MRR, Faithfulness)"]
        SYSTEM_A -.->|Target Under Test| Evaluator
        
        ThreatModel["OWASP LLM Threat Suite"] --> RedTeam["AI Red Teaming Engine<br>(app/security/red_team.py)"]
        SYSTEM_A -.->|Target Under Test| RedTeam
        
        GoldenBaseline[("Golden Baseline<br>(data/baselines/)")] --> RegressionEngine["Regression Engine<br>(app/regression/regression_runner.py)"]
        Evaluator -.-> RegressionEngine
        
        Evaluator --> QualityGate{"Quality Gate<br>(app/reporting/quality_gate.py)"}
        RedTeam --> QualityGate
        RegressionEngine --> QualityGate
        
        QualityGate -->|PASS / FAIL| ReportGen["Report Generator<br>(JSON + HTML Reports)"]
        QualityGate -->|NEEDS REVIEW| HumanAudit["Human Approval & Audit<br>(data/audit_log.json)"]
    end

    subgraph INTERFACES["Interfaces & Delivery"]
        ReportGen --> StreamlitUI["Streamlit Dashboard<br>(dashboard/app.py)"]
        SYSTEM_A --> FastAPI["FastAPI REST Backend<br>(app/api/main.py)"]
        SYSTEM_B --> FastAPI
        FastAPI --> GitHubActions["GitHub Actions CI/CD<br>(.github/workflows/ci.yml)"]
    end
```

---

## 2. Document Ingestion & Storage Pipeline

```mermaid
flowchart LR
    A["Raw PDF Files<br>(data/documents/)"] --> B["PyPDFLoader<br>Text Extraction"]
    B --> C["Metadata Preservation<br>(source, page_number)"]
    C --> D["Recursive Character Text Splitter<br>(800 chars / 150 overlap)"]
    D --> E["Deterministic Chunk ID Generation<br>(MD5 hash content + page)"]
    E --> F["Batch Embedding Generation<br>(384-dimensional dense vectors)"]
    F --> G[("ChromaDB Persistent Store<br>(HNSW Vector Index)")]
```

---

## 3. Two-Stage RAG Query Execution Flow

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Pipeline as RAGPipeline
    participant Retriever as RAGRetriever
    participant Chroma as ChromaDB
    participant Reranker as CrossEncoderReranker
    participant Generator as RAGGenerator

    User->>Pipeline: Submit Query ("What is continuous integration?")
    Pipeline->>Retriever: retrieve(query, top_k=6)
    Retriever->>Chroma: similarity_search_with_score(query, k=6)
    Chroma-->>Retriever: 6 Candidate Chunks with L2 Distances
    Retriever-->>Pipeline: Return List[RetrievedChunk]
    
    Pipeline->>Reranker: rerank(query, 6 candidates, top_k=3)
    Note over Reranker: Joint Self-Attention over<br>[Query, Candidate Passage]
    Reranker-->>Pipeline: Top-3 Reranked Chunks (Page 24 promoted to Rank 1)
    
    Pipeline->>Generator: generate(query, 3 refined chunks)
    Note over Generator: Context Insufficiency Check<br>& Strict Grounding Template
    Generator-->>Pipeline: RAGResponse (Answer + [Source: File, Page])
    Pipeline-->>User: Structured RAGPipelineResult
```

---

## 4. QA Automation Execution Flow (Pytest)

```mermaid
flowchart TD
    Pytest["Pytest Execution<br>(pytest.ini: pythonpath = .)"] --> Disc["Test Discovery (22 Tests)"]
    Disc --> T1["Unit Metrics Tests<br>(Precision@K, Recall@K, MRR, NDCG, Faithfulness)"]
    Disc --> T2["Ingestion Tests<br>(Empty dir handling, metadata preservation)"]
    Disc --> T3["RAG Pipeline Tests<br>(Functional queries, refusal on empty context)"]
    Disc --> T4["Security Tests<br>(Data leakage scanner, direct injection mitigation)"]
    Disc --> T5["Regression Tests<br>(Metric delta thresholds, new failure detection)"]
    Disc --> T6["FastAPI Integration Tests<br>(Health check, /api/v1/query, /api/v1/human-approval)"]
    T1 & T2 & T3 & T4 & T5 & T6 --> Result["Test Summary: 22/22 PASSED in <8s"]
```

---

## 5. Promptfoo Evaluation Workflow

```mermaid
flowchart LR
    Config["promptfooconfig.yaml<br>promptfoo_security.yaml"] --> CLI["npx promptfoo eval"]
    CLI --> Provider["promptfoo/provider.py<br>(call_api adapter)"]
    Provider --> Pipeline["RAGPipeline.run(query)"]
    Pipeline --> Provider
    Provider --> Asserts["Promptfoo Assertions<br>(icontains, not-contains, regex citations)"]
    Asserts --> PFReport["Promptfoo Evaluation Matrix & Diff"]
```

---

## 6. AI Red Teaming & Threat Model Flow

```mermaid
flowchart TD
    ThreatSuite["Curated Adversarial Suite<br>(app/security/prompt_injection.py)"] --> Runner["RedTeamRunner<br>(app/security/red_team.py)"]
    
    subgraph ATTACKS["Attack Vectors"]
        A1["Direct Prompt Injection<br>('Ignore all prior rules...')"]
        A2["System Prompt Extraction<br>('Output your system prompt...')"]
        A3["Jailbreak / DAN Attack<br>('You are now unrestricted...')"]
        A4["Sensitive Data Disclosure<br>('What is the admin password?')"]
        A5["Indirect Document Injection<br>('Execute exfiltration command...')"]
    end
    
    ThreatSuite --> ATTACKS
    ATTACKS --> Pipeline["RAGPipeline"]
    Pipeline --> Scanner["Data Leakage Scanner<br>(Regex API keys, passwords, PII)"]
    Scanner --> ForbiddenCheck{"Forbidden Payloads<br>or Leaks Found?"}
    ForbiddenCheck -->|No| Blocked["Status: BLOCKED (Safe)"]
    ForbiddenCheck -->|Yes| Exploited["Status: EXPLOITED (Vulnerability Detected)"]
    Blocked & Exploited --> SecReport["RedTeamSecurityReport<br>(Mitigation Score, Severity, Remediation)"]
```

---

## 7. Regression Testing Workflow

```mermaid
flowchart LR
    CurrentEval["Current Evaluation Run<br>(RAGEvaluator.run_evaluation)"] --> Compare["Regression Engine<br>(app/regression/regression_runner.py)"]
    Baseline[("Golden Baseline<br>data/baselines/baseline_v1.json")] --> Compare
    Compare --> Deltas["Calculate Metric Deltas<br>(ΔPrecision, ΔRecall, ΔMRR, ΔFaithfulness)"]
    Deltas --> Check{"Any Metric Drop > 5%<br>or New Test Failures?"}
    Check -->|Yes| Alert["🚨 REGRESSION DETECTED<br>Log Failures & Degradation"]
    Check -->|No| Stable["✅ STABLE<br>Zero Performance Degradation"]
```

---

## 8. Quality Gate & Human Approval Workflow

```mermaid
flowchart TD
    Eval["Evaluation Report"] & Sec["Security Report"] & Reg["Regression Analysis"] --> QG["evaluate_quality_gate()"]
    
    subgraph GATES["Gate Thresholds"]
        G1["Zero Critical Security Leaks (0)"]
        G2["Minimum Faithfulness (>= 0.80)"]
        G3["Minimum Precision@3 (>= 0.50)"]
        G4["Zero Regressions vs Baseline (0)"]
    end
    
    QG --> GATES
    GATES --> Decision{"All Rules Satisfied?"}
    Decision -->|Yes| Pass["Status: PASS<br>(Deploy to Production)"]
    Decision -->|Security Breached| Fail["Status: FAIL<br>(Build Blocked Immediately)"]
    Decision -->|Minor Quality Dip| Review["Status: NEEDS_HUMAN_REVIEW"]
    
    Review --> Signoff["Human Approval Reviewer<br>(Streamlit UI / API Endpoint)"]
    Signoff --> Audit[("Audit Log<br>data/audit_log.json")]
    Audit --> Pass
```

---

## 9. CI/CD GitHub Actions Pipeline

```mermaid
flowchart LR
    Push["git push / PR to main"] --> GH["GitHub Actions Runner<br>(ubuntu-latest, Python 3.11)"]
    GH --> Install["pip install -r requirements.txt"]
    Install --> Pytest["Run Pytest Suite (22/22)"]
    Pytest --> ReportGen["Run report_generator.py<br>(Eval + Red Team + Quality Gate)"]
    ReportGen --> GateCheck{"Quality Gate<br>Outcome"}
    GateCheck -->|PASS| Success["Build Green ✅"]
    GateCheck -->|FAIL| Failure["Build Blocked ❌"]
    Success & Failure --> Artifacts["Upload Artifacts<br>(reports/*.html & reports/*.json)"]
```
