import pytest
from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_rag_query_api():
    payload = {
        "query": "What is continuous integration?",
        "retrieval_k": 4,
        "rerank_k": 2,
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "continuous integration" in data["answer"].lower()
    assert len(data["sources"]) > 0
    assert data["has_sufficient_context"] is True


def test_human_approval_api(tmp_path):
    payload = {
        "action": "Approve Prompt Guardrail Update v1",
        "approved": True,
        "reviewer": "QA_Lead_Alex",
        "comments": "Audited zero hallucination risks.",
    }
    response = client.post("/api/v1/human-approval", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "recorded"
    assert data["approval"]["reviewer"] == "QA_Lead_Alex"


def test_latest_report_api():
    response = client.get("/api/v1/reports/latest")
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        data = response.json()
        assert "quality_gate" in data
