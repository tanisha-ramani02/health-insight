"""Integration tests for FastAPI REST endpoints using TestClient."""

import pytest
from fastapi.testclient import TestClient
from api.app import app


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI test client using context manager for lifespan events."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Verify /api/v1/health returns healthy status and indexed chunk count."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["total_pages"] == 128
    assert data["total_chunks"] > 0


def test_chunk_evaluation_endpoint(client):
    """Verify /api/v1/chunk-evaluation returns dynamic chunking metrics for page 15."""
    response = client.post("/api/v1/chunk-evaluation", json={"page_number": 15})
    assert response.status_code == 200
    data = response.json()
    assert data["page_number"] == 15
    assert data["total_chunks"] > 0
    assert len(data["dynamic_chunk_sizes"]) == data["total_chunks"]
    assert data["avg_entropy"] > 0.0


def test_greeting_intent_endpoint(client):
    """Verify greeting query returns warm intent response with 1.0 confidence."""
    response = client.post("/api/v1/query", json={"query": "hi"})
    assert response.status_code == 200
    data = response.json()
    assert "Medicare" in data["answer"]
    assert data["confidence_score"] == 1.0
    assert data["source_page"] is None


def test_empty_query_raises_400(client):
    """Verify blank or whitespace query is rejected with HTTP 400."""
    response = client.post("/api/v1/query", json={"query": "   "})
    assert response.status_code == 400
    assert "blank" in response.json()["detail"].lower()


def test_out_of_scope_query_rejection(client):
    """Verify out-of-scope query receives non-grounded response and null source page."""
    response = client.post("/api/v1/query", json={"query": "How do I replace an alternator on a Honda Civic?"})
    assert response.status_code == 200
    data = response.json()
    assert data["source_page"] is None
    assert len(data["answer"]) > 10



def test_query_endpoint(client):
    """Verify /api/v1/query returns structured JSON matching Assignment.md specification."""
    payload = {
        "query": "What are the important deadlines for Medicare enrollment?",
        "top_k": 5
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Verify required fields
    assert "answer" in data
    assert "source_page" in data
    assert "confidence_score" in data
    assert "chunk_size" in data
    assert "turns_remaining" in data
    assert "is_completed" in data

    assert len(data["answer"]) > 10
    assert isinstance(data["source_page"], int) and 1 <= data["source_page"] <= 128
    assert 0.0 <= data["confidence_score"] <= 1.0
    assert data["chunk_size"] > 0
    assert "session_id" in data


def test_session_endpoints(client):
    """Verify /api/v1/sessions, /api/v1/session/{id}, and /api/v1/session/new."""
    # List sessions
    res_list = client.get("/api/v1/sessions")
    assert res_list.status_code == 200
    list_data = res_list.json()
    assert "sessions" in list_data

    # Create new session
    res_new = client.post("/api/v1/session/new")
    assert res_new.status_code == 200
    new_sid = res_new.json()["session_id"]

    # Send a query on that session
    res_q = client.post("/api/v1/query", json={"query": "What is Part D out of pocket cap in 2025?", "session_id": new_sid})
    assert res_q.status_code == 200
    q_data = res_q.json()
    assert q_data["turn_count"] == 1
    assert q_data["turns_remaining"] == 19

    # Get session details
    res_details = client.get(f"/api/v1/session/{new_sid}")
    assert res_details.status_code == 200
    details_data = res_details.json()
    assert details_data["session_id"] == new_sid
    assert len(details_data["messages"]) == 2
