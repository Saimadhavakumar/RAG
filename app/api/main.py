import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from app.rag.pipeline import RAGPipeline, RAGPipelineResult
from app.rag.ingestion import load_documents, split_documents
from app.rag.vector_store import index_documents
from app.evaluation.evaluator import RAGEvaluator
from app.security.red_team import RedTeamRunner
from app.regression.regression_runner import run_regression_check
from app.reporting.report_generator import run_full_report_generation
from app.reporting.quality_gate import log_human_approval

app = FastAPI(
    title="RAGGuard AI — RAG Evaluation & Security Testing API",
    description="Industry-grade QA Automation, Evaluation, and AI Red Teaming platform for RAG applications.",
    version="1.0.0",
)

# Global pipeline instance
_pipeline = RAGPipeline()


# ---------------------------------------------------------
# Request & Response Schemas (Pydantic v2)
# ---------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., example="What is continuous integration?")
    retrieval_k: Optional[int] = Field(default=6, ge=1, le=20)
    rerank_k: Optional[int] = Field(default=3, ge=1, le=10)


class QueryResponse(BaseModel):
    query: str
    answer: str
    has_sufficient_context: bool
    sources: List[Dict[str, Any]]
    model_used: str
    reranker_used: str
    total_latency_seconds: float


class IngestRequest(BaseModel):
    directory: str = Field(default="data/documents")


class IngestResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_indexed: int


class HumanApprovalRequest(BaseModel):
    action: str = Field(..., example="Deploy prompt update v2")
    approved: bool = Field(...)
    reviewer: str = Field(..., example="QA_Lead_Alex")
    comments: str = Field(..., example="Verified zero hallucinations and passing security score.")


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------

@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "RAGGuard AI Platform",
        "version": "1.0.0",
        "docs_url": "/docs",
    }


@app.post("/api/v1/query", response_model=QueryResponse, tags=["RAG Application"])
def execute_rag_query(req: QueryRequest):
    """Executes a query through the full RAG pipeline (Retrieval -> Reranking -> Grounded Generation)."""
    try:
        pipeline = RAGPipeline(retrieval_k=req.retrieval_k, rerank_k=req.rerank_k)
        result: RAGPipelineResult = pipeline.run(req.query)
        return QueryResponse(
            query=result.query,
            answer=result.answer,
            has_sufficient_context=result.has_sufficient_context,
            sources=result.sources,
            model_used=result.model_used,
            reranker_used=result.reranker_used,
            total_latency_seconds=result.total_latency_seconds,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}",
        )


@app.post("/api/v1/ingest", response_model=IngestResponse, tags=["RAG Application"])
def ingest_documents_endpoint(req: IngestRequest):
    """Loads PDF documents, performs chunking, and indexes embeddings into ChromaDB."""
    try:
        raw_docs = load_documents(req.directory)
        if not raw_docs:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No PDF documents found in directory: {req.directory}",
            )
        chunks = split_documents(raw_docs)
        count = index_documents(chunks)
        return IngestResponse(
            status="success",
            documents_loaded=len(raw_docs),
            chunks_indexed=count,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(e)}",
        )


@app.post("/api/v1/evaluate", tags=["QA Platform"])
def run_evaluation_endpoint():
    """Runs the RAG evaluation engine across benchmark test cases and computes metrics."""
    try:
        evaluator = RAGEvaluator()
        report = evaluator.run_evaluation()
        return asdict(report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.post("/api/v1/red-team", tags=["AI Security"])
def run_red_team_endpoint():
    """Executes adversarial attack vectors (prompt injection, jailbreak, sensitive disclosure)."""
    try:
        runner = RedTeamRunner()
        report = runner.run_security_audit()
        return asdict(report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Red Team audit failed: {str(e)}")


@app.post("/api/v1/regression", tags=["QA Platform"])
def run_regression_endpoint():
    """Executes regression analysis against historical Golden Baseline."""
    try:
        analysis = run_regression_check()
        return asdict(analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regression analysis failed: {str(e)}")


@app.post("/api/v1/reports/generate", tags=["Reporting & Gates"])
def generate_full_report_endpoint():
    """Executes the complete evaluation, security audit, and quality gate check, generating JSON and HTML reports."""
    try:
        html_path = run_full_report_generation()
        latest_json = Path("reports/latest_report.json")
        data = {}
        if latest_json.exists():
            with open(latest_json, "r", encoding="utf-8") as f:
                data = json.load(f)
        return {
            "status": "success",
            "html_report_path": html_path,
            "report_data": data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@app.get("/api/v1/reports/latest", tags=["Reporting & Gates"])
def get_latest_report():
    """Retrieves the latest generated JSON report."""
    path = Path("reports/latest_report.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="No report found. Generate one first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/v1/human-approval", tags=["Reporting & Gates"])
def record_human_approval(req: HumanApprovalRequest):
    """Records human approval or rejection of suggested fixes into an audit log."""
    try:
        log_human_approval(
            action=req.action,
            approved=req.approved,
            reviewer=req.reviewer,
            comments=req.comments,
        )
        return {"status": "recorded", "approval": req.dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record approval: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.main:app", host="127.0.0.1", port=8000, reload=True)
