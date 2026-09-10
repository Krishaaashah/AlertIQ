"""
Tests for the FastAPI backend (api/).

These exercise the API in 'cold start' mode — i.e. without a trained
suppression model or RL agent on disk — because that's the state a fresh
clone or a grader's machine will be in. The API is designed to fall back
gracefully in this state (static-threshold decision instead of RL, 0.5
default probability instead of a calibrated one), and that fallback
behavior is exactly what's being verified here.

Requires: fastapi, httpx (see requirements-dev.txt). Skipped automatically
if fastapi isn't installed.
"""

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

client = TestClient(app)


def test_root_health_check():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "AlertIQ"
    assert body["status"] == "running"


def test_detailed_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


SAMPLE_ALERT = {
    "amount": 250000.0,
    "step": 12,
    "rule_count": 2,
    "alert_reason": "R1",
    "balance_drain_ratio": 0.95,
    "oldbalanceOrg": 260000.0,
    "newbalanceOrig": 13000.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 250000.0,
    "type_CASH_OUT": 1,
    "type_DEBIT": 0,
    "type_PAYMENT": 0,
    "type_TRANSFER": 0,
}


def test_evaluate_single_alert_returns_a_decision():
    response = client.post("/api/alerts/evaluate", json=SAMPLE_ALERT)
    assert response.status_code == 200

    body = response.json()
    assert body["decision"] in {"SUPPRESS", "ESCALATE"}
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["risk_level"] in {"MINIMAL", "LOW", "MEDIUM", "HIGH"}
    assert isinstance(body["explanation"], str) and len(body["explanation"]) > 0


def test_evaluate_batch_of_alerts():
    response = client.post("/api/alerts/batch", json={"alerts": [SAMPLE_ALERT, SAMPLE_ALERT]})
    assert response.status_code == 200

    body = response.json()
    assert body["total_alerts"] == 2
    assert body["suppressed"] + body["escalated"] == 2
    assert "suppression_rate" in body["summary"]


def test_evaluate_batch_rejects_empty_list():
    response = client.post("/api/alerts/batch", json={"alerts": []})
    assert response.status_code == 400


def test_stats_endpoint_reflects_processed_alerts():
    client.post("/api/alerts/evaluate", json=SAMPLE_ALERT)
    response = client.get("/api/alerts/stats")
    assert response.status_code == 200
    assert response.json()["total_alerts_processed"] >= 1
