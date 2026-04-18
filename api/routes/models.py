"""
Model information endpoints.

GET /api/models/info            — Get loaded model metadata
GET /api/models/comparison      — Get RL vs baseline comparison
GET /api/models/training-curves — Serve training curve image path
"""

import os
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.config import RL_OUTPUT_DIR, OUTPUT_DIR, SUPPRESSION_MODEL_PATH

router = APIRouter()


def _get_app_state():
    from api.main import app_state
    return app_state


@router.get("/info")
async def model_info():
    """Get information about loaded models."""
    app_state = _get_app_state()
    
    # Suppression model info
    suppression_info = {
        "loaded": app_state.suppression_model is not None,
        "type": "CalibratedClassifierCV (Logistic Regression + Platt Scaling)",
        "path": SUPPRESSION_MODEL_PATH,
        "cost_sensitive": True,
        "missed_fraud_cost_weight": 50,
    }
    
    # RL agent info
    rl_info = {
        "loaded": app_state.rl_agent is not None,
        "type": app_state.rl_agent_type or "none",
        "state_dim": 5,
        "action_dim": 2,
        "actions": ["SUPPRESS", "ESCALATE"],
        "state_features": [
            "prob_fraud", "rule_count_norm", "amount_norm", "fraud_rate", "workload"
        ],
    }
    
    if app_state.rl_agent is not None and hasattr(app_state.rl_agent, 'epsilon'):
        rl_info['epsilon'] = app_state.rl_agent.epsilon
    
    # Feature importance for bandit
    if app_state.rl_agent_type == 'bandit' and hasattr(app_state.rl_agent, 'get_feature_importance'):
        importance = app_state.rl_agent.get_feature_importance()
        rl_info['feature_importance'] = {
            'suppress_weights': importance['SUPPRESS'].tolist(),
            'escalate_weights': importance['ESCALATE'].tolist(),
            'feature_names': importance['feature_names'],
        }
    
    # Drift detector info
    drift_info = {
        "initialized": app_state.drift_detector is not None,
    }
    if app_state.drift_detector is not None:
        drift_info.update(app_state.drift_detector.get_drift_summary())
    
    return {
        "suppression_model": suppression_info,
        "rl_agent": rl_info,
        "drift_detector": drift_info,
    }


@router.get("/comparison")
async def model_comparison():
    """Get RL vs baseline policy comparison results."""
    comparison_path = os.path.join(RL_OUTPUT_DIR, 'policy_comparison.csv')
    
    if not os.path.exists(comparison_path):
        raise HTTPException(
            status_code=404,
            detail="No comparison results found. Run Phase 4 (RL training) first."
        )
    
    df = pd.read_csv(comparison_path)
    records = df.to_dict(orient='records')
    
    # Find best policy
    rl_row = [r for r in records if r.get('Policy') == 'RL Agent']
    static_rows = [r for r in records if r.get('Policy') == 'Static Threshold']
    
    best_policy = "RL Agent"
    recommendation = "RL agent provides adaptive, context-aware decision making."
    
    if rl_row and static_rows:
        rl_reward = rl_row[0].get('Avg Reward', 0)
        best_static = max(static_rows, key=lambda x: x.get('Avg Reward', 0))
        if best_static.get('Avg Reward', 0) > rl_reward:
            best_policy = f"Static Threshold ({best_static.get('Parameter', 'N/A')})"
            recommendation = (
                "Static threshold currently outperforms RL agent. "
                "Consider retraining with more episodes or adjusting reward structure."
            )
    
    return {
        "comparison_data": records,
        "best_policy": best_policy,
        "recommendation": recommendation,
    }


@router.get("/training-curves")
async def training_curves():
    """Serve training curves image."""
    chart_path = os.path.join(RL_OUTPUT_DIR, 'training_curves.png')
    if not os.path.exists(chart_path):
        raise HTTPException(status_code=404, detail="Training curves not found.")
    return FileResponse(chart_path, media_type="image/png")


@router.get("/policy-comparison-chart")
async def policy_comparison_chart():
    """Serve policy comparison chart."""
    chart_path = os.path.join(RL_OUTPUT_DIR, 'policy_comparison.png')
    if not os.path.exists(chart_path):
        raise HTTPException(status_code=404, detail="Policy comparison chart not found.")
    return FileResponse(chart_path, media_type="image/png")


@router.get("/evaluation-report")
async def evaluation_report():
    """Get evaluation report text."""
    report_path = os.path.join(RL_OUTPUT_DIR, 'evaluation_report.txt')
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Evaluation report not found.")
    
    with open(report_path, 'r') as f:
        report_text = f.read()
    
    return {"report": report_text}
