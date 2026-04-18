"""
Drift detection endpoints.

POST /api/drift/check   — Run drift detection on test data
GET  /api/drift/status  — Get current drift status
"""

import os
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import DriftCheckResponse, DriftSeverity
from src.config import OUTPUT_DIR

router = APIRouter()


def _get_app_state():
    from api.main import app_state
    return app_state


@router.post("/check", response_model=DriftCheckResponse)
async def check_drift():
    """Run drift detection comparing training data vs test data."""
    app_state = _get_app_state()
    
    if app_state.drift_detector is None:
        # Try to initialize
        train_path = os.path.join(OUTPUT_DIR, 'train_feedback.csv')
        if not os.path.exists(train_path):
            raise HTTPException(
                status_code=400,
                detail="No training data available. Run Phase 2 first."
            )
        from src.drift_detector import DriftDetector
        train_df = pd.read_csv(train_path)
        app_state.drift_detector = DriftDetector(train_df)
    
    # Use test data as target
    test_path = os.path.join(OUTPUT_DIR, 'test_feedback.csv')
    if not os.path.exists(test_path):
        raise HTTPException(
            status_code=400,
            detail="No test data available. Run Phase 2 first."
        )
    
    target_df = pd.read_csv(test_path)
    report = app_state.drift_detector.check_drift(target_df)
    
    return DriftCheckResponse(
        overall_severity=DriftSeverity(report['overall_severity']),
        avg_psi=report['avg_psi'],
        max_psi=report['max_psi'],
        avg_kl_divergence=report['avg_kl_divergence'],
        total_features_monitored=report['total_features_monitored'],
        features_drifted=report['features_drifted'],
        recommendation=report['recommendation'],
        fallback_active=report['fallback_active'],
        feature_reports=report['feature_reports'],
    )


@router.post("/simulate")
async def simulate_drift(intensity: float = 0.3):
    """Simulate drift on test data and run detection (for demo purposes)."""
    app_state = _get_app_state()
    
    if app_state.drift_detector is None:
        train_path = os.path.join(OUTPUT_DIR, 'train_feedback.csv')
        if not os.path.exists(train_path):
            raise HTTPException(status_code=400, detail="No training data available.")
        from src.drift_detector import DriftDetector
        train_df = pd.read_csv(train_path)
        app_state.drift_detector = DriftDetector(train_df)
    
    test_path = os.path.join(OUTPUT_DIR, 'test_feedback.csv')
    if not os.path.exists(test_path):
        raise HTTPException(status_code=400, detail="No test data available.")
    
    from src.drift_detector import simulate_drift as sim_drift
    target_df = pd.read_csv(test_path)
    drifted_df = sim_drift(target_df, drift_intensity=intensity)
    
    report = app_state.drift_detector.check_drift(drifted_df)
    
    return {
        "drift_intensity": intensity,
        "overall_severity": report['overall_severity'],
        "avg_psi": report['avg_psi'],
        "max_psi": report['max_psi'],
        "features_drifted": report['features_drifted'],
        "recommendation": report['recommendation'],
        "fallback_active": report['fallback_active'],
    }


@router.get("/status")
async def drift_status():
    """Get current drift monitoring status."""
    app_state = _get_app_state()
    
    if app_state.drift_detector is None:
        return {
            "initialized": False,
            "message": "Drift detector not initialized. Run drift check first."
        }
    
    summary = app_state.drift_detector.get_drift_summary()
    summary['initialized'] = True
    summary['fallback_bias'] = app_state.drift_detector.get_fallback_action_bias()
    
    return summary
