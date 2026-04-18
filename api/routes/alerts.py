"""
Alert evaluation endpoints.

POST /api/alerts/evaluate  — Evaluate a single alert
POST /api/alerts/batch     — Evaluate a batch of alerts
GET  /api/alerts/stats     — Get system statistics
"""

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import (
    AlertFeatures, AlertDecisionResponse, BatchAlertRequest,
    BatchAlertResponse, SystemStats, DecisionEnum, RiskLevel, DriftSeverity
)

router = APIRouter()


def _get_app_state():
    """Get the global app state."""
    from api.main import app_state
    return app_state


def _evaluate_single_alert(alert: AlertFeatures, app_state) -> AlertDecisionResponse:
    """Core evaluation logic for a single alert."""
    from src.explainability import ExplainabilityEngine
    
    # Build feature DataFrame for the suppression model
    alert_dict = alert.model_dump()
    alert_df = pd.DataFrame([alert_dict])
    
    # Get fraud probability from suppression model
    if app_state.suppression_model is not None:
        try:
            probs = app_state.suppression_model.predict_proba(alert_df)
            prob_fraud = float(probs[0][1])
            prob_benign = float(probs[0][0])
        except Exception:
            # If model fails (feature mismatch), use a safe default
            prob_fraud = 0.5
            prob_benign = 0.5
    else:
        prob_fraud = 0.5
        prob_benign = 0.5
    
    # Normalize features for RL state
    max_amount = 1e7  # Reasonable max for PaySim
    amount_normalized = np.log1p(alert.amount) / np.log1p(max_amount)
    rule_count_normalized = alert.rule_count / 4.0
    fraud_rate = 0.013  # Typical PaySim alert fraud rate
    workload = 0.5  # Default mid-range workload
    
    state = np.array([
        prob_fraud,
        rule_count_normalized,
        amount_normalized,
        fraud_rate,
        workload
    ], dtype=np.float32)
    
    # Get RL decision
    if app_state.rl_agent is not None:
        action = app_state.rl_agent.select_action(state, training=False)
    else:
        # Fallback: static threshold
        action = 0 if prob_benign > 0.90 else 1
    
    # Check drift override
    drift_active = False
    if app_state.drift_detector is not None:
        bias = app_state.drift_detector.get_fallback_action_bias()
        if bias > 0 and np.random.random() < bias:
            action = 1  # Force escalation
            drift_active = True
    
    # Generate explanation
    bandit = app_state.rl_agent if app_state.rl_agent_type == 'bandit' else None
    explainer = ExplainabilityEngine(bandit_agent=bandit)
    explanation = explainer.explain_decision(
        action=action,
        state=state,
        alert_data=alert_dict,
        drift_active=drift_active
    )
    
    # Track
    app_state.alerts_processed += 1
    
    return AlertDecisionResponse(
        decision=DecisionEnum.SUPPRESS if action == 0 else DecisionEnum.ESCALATE,
        confidence=explanation['confidence_score'],
        risk_level=RiskLevel(explanation['risk_level']),
        fraud_probability=round(prob_fraud, 4),
        explanation=explanation['explanation'],
        risk_factors=explanation['risk_factors'],
        state_features=explanation['state_features'],
        drift_override=drift_active,
    )


@router.post("/evaluate", response_model=AlertDecisionResponse)
async def evaluate_alert(alert: AlertFeatures):
    """Evaluate a single alert and return suppress/escalate decision with explanation."""
    app_state = _get_app_state()
    return _evaluate_single_alert(alert, app_state)


@router.post("/batch", response_model=BatchAlertResponse)
async def evaluate_batch(request: BatchAlertRequest):
    """Evaluate a batch of alerts."""
    app_state = _get_app_state()
    
    if not request.alerts:
        raise HTTPException(status_code=400, detail="No alerts provided")
    
    decisions = []
    for alert in request.alerts:
        decision = _evaluate_single_alert(alert, app_state)
        decisions.append(decision)
    
    suppressed = sum(1 for d in decisions if d.decision == DecisionEnum.SUPPRESS)
    escalated = sum(1 for d in decisions if d.decision == DecisionEnum.ESCALATE)
    
    return BatchAlertResponse(
        total_alerts=len(decisions),
        suppressed=suppressed,
        escalated=escalated,
        decisions=decisions,
        summary={
            "suppression_rate": round(suppressed / len(decisions), 4) if decisions else 0,
            "escalation_rate": round(escalated / len(decisions), 4) if decisions else 0,
            "avg_fraud_probability": round(
                np.mean([d.fraud_probability for d in decisions]), 4
            ) if decisions else 0,
        }
    )


@router.get("/stats", response_model=SystemStats)
async def get_stats():
    """Get current system statistics."""
    app_state = _get_app_state()
    
    drift_status = DriftSeverity.STABLE
    if app_state.drift_detector and app_state.drift_detector.drift_history:
        latest = app_state.drift_detector.drift_history[-1]
        drift_status = DriftSeverity(latest['overall_severity'])
    
    return SystemStats(
        total_alerts_processed=app_state.alerts_processed,
        fraud_rate=0.013,
        suppression_rate=0.0,
        escalation_rate=0.0,
        model_loaded=app_state.suppression_model is not None,
        rl_agent_loaded=app_state.rl_agent is not None,
        drift_status=drift_status,
    )
