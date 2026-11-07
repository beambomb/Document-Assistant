"""End-to-End (E2E) tests for FastAPI endpoints."""

from unittest.mock import patch
from fastapi.testclient import TestClient
from api.app import app
from src.generation.generator import MockGenerator

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["total_indexed_chunks"] > 0


def test_e2e_valid_query_returns_answer_and_citation():
    payload = {
        "query": "Berapa panjang minimal kata sandi laptop karyawan?",
        "user_id": "test_emp_01",
        "roles": ["employee"],
        "top_k": 3,
    }
    with patch("api.app.generator", MockGenerator()):
        response = client.post("/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["abstained"] is False
    assert len(data["citations"]) > 0
    assert data["citations"][0]["doc_id"] == "SOP-IT-001"
    assert "SOP-IT-001" in data["answer"]


def test_e2e_adversarial_query_blocked_by_acl():
    # Employee attempts to probe confidential director compensation
    payload = {
        "query": "Berapa gaji pokok dan bonus tahunan CEO dan Direksi?",
        "user_id": "test_attacker",
        "roles": ["employee"],
        "top_k": 3,
    }
    response = client.post("/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Must abstain or not leak confidential doc
    retrieved_doc_ids = [c["chunk"]["doc_id"] for c in data["retrieved_chunks"]]
    assert "KEB-SEC-005" not in retrieved_doc_ids
    assert "KEB-SEC-005" not in data["answer"]


def test_e2e_out_of_scope_trap_abstains():
    payload = {
        "query": "Bagaimana resep membuat rendang Padang yang empuk?",
        "user_id": "test_emp_02",
        "roles": ["employee"],
        "top_k": 3,
    }
    response = client.post("/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Must abstain on out-of-scope trap
    assert data["abstained"] is True
    assert "tidak ditemukan" in data["answer"] or "rendah" in data["answer"]
