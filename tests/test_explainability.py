"""Tests for src/explainability.py — audit-readable decision explanations."""

import numpy as np

from src.explainability import ExplainabilityEngine


def test_high_risk_escalation_is_labeled_high():
    engine = ExplainabilityEngine()
    state = np.array([0.85, 1.0, 0.9, 0.05, 0.5], dtype=np.float32)
    explanation = engine.explain_decision(action=1, state=state, alert_data={"alert_reason": "R1"})

    assert explanation["decision"] == "ESCALATE"
    assert explanation["risk_level"] == "HIGH"
    assert 0.0 <= explanation["confidence_score"] <= 1.0
    assert "High-value transaction" in " ".join(explanation["risk_factors"])


def test_low_risk_suppression_is_labeled_minimal_or_low():
    engine = ExplainabilityEngine()
    state = np.array([0.02, 0.0, 0.1, 0.01, 0.3], dtype=np.float32)
    explanation = engine.explain_decision(action=0, state=state, alert_data={"alert_reason": "none"})

    assert explanation["decision"] == "SUPPRESS"
    assert explanation["risk_level"] in {"MINIMAL", "LOW"}


def test_drift_override_forces_drift_explanation_template():
    engine = ExplainabilityEngine()
    state = np.array([0.4, 0.5, 0.5, 0.05, 0.5], dtype=np.float32)
    explanation = engine.explain_decision(action=1, state=state, drift_active=True)

    assert explanation["drift_override"] is True
    assert "drift" in explanation["explanation"].lower()


def test_state_features_are_all_present_and_bounded():
    engine = ExplainabilityEngine()
    state = np.array([0.5, 0.5, 0.5, 0.02, 0.5], dtype=np.float32)
    explanation = engine.explain_decision(action=1, state=state)

    expected_keys = {
        "fraud_probability", "benign_probability", "rule_intensity",
        "amount_severity", "system_fraud_rate", "workload_level",
    }
    assert expected_keys <= set(explanation["state_features"].keys())
    for value in explanation["state_features"].values():
        assert 0.0 <= value <= 1.0
