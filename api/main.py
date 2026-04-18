"""
AlertIQ — FastAPI Application Entry Point.

Main application with CORS, startup model loading, and route registration.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    API_TITLE, API_VERSION, OUTPUT_DIR, RL_OUTPUT_DIR,
    SUPPRESSION_MODEL_PATH, MODEL_DIR
)


# ─────────────────────────────────────────────
#  Global State (loaded on startup)
# ─────────────────────────────────────────────
class AppState:
    """Container for loaded models and system state."""
    suppression_model = None
    rl_agent = None
    rl_agent_type = None
    drift_detector = None
    train_df = None
    test_df = None
    eval_df = None
    pipeline_status = "IDLE"
    pipeline_message = ""
    alerts_processed = 0
    decisions_log = []


app_state = AppState()


# ─────────────────────────────────────────────
#  Lifespan (startup / shutdown)
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    print("="*60)
    print("  AlertIQ API -- Starting Up")
    print("="*60)
    
    # Load suppression model if available
    if os.path.exists(SUPPRESSION_MODEL_PATH):
        try:
            app_state.suppression_model = joblib.load(SUPPRESSION_MODEL_PATH)
            print(f"  [OK] Suppression model loaded from {SUPPRESSION_MODEL_PATH}")
        except Exception as e:
            print(f"  [FAIL] Failed to load suppression model: {e}")
    else:
        print(f"  [WARN] Suppression model not found at {SUPPRESSION_MODEL_PATH}")
    
    # Load RL agent if available
    dqn_path = os.path.join(RL_OUTPUT_DIR, 'dqn_agent.pkl')
    bandit_path = os.path.join(RL_OUTPUT_DIR, 'bandit_agent.pkl')
    
    if os.path.exists(dqn_path):
        try:
            from src.rl_trainer import load_policy
            app_state.rl_agent = load_policy(dqn_path, agent_type='dqn', device='cpu')
            app_state.rl_agent_type = 'dqn'
            print(f"  [OK] DQN agent loaded from {dqn_path}")
        except Exception as e:
            print(f"  [FAIL] Failed to load DQN agent: {e}")
    elif os.path.exists(bandit_path):
        try:
            from src.rl_trainer import load_policy
            app_state.rl_agent = load_policy(bandit_path, agent_type='bandit', device='cpu')
            app_state.rl_agent_type = 'bandit'
            print(f"  [OK] Bandit agent loaded from {bandit_path}")
        except Exception as e:
            print(f"  [FAIL] Failed to load Bandit agent: {e}")
    else:
        print("  [WARN] No RL agent found. Train one first with run_phase4_rl.py")
    
    # Load test data for drift detection baseline
    test_data_path = os.path.join(OUTPUT_DIR, 'test_feedback.csv')
    train_data_path = os.path.join(OUTPUT_DIR, 'train_feedback.csv')
    if os.path.exists(train_data_path):
        try:
            app_state.train_df = pd.read_csv(train_data_path)
            from src.drift_detector import DriftDetector
            app_state.drift_detector = DriftDetector(app_state.train_df)
            print(f"  [OK] Drift detector initialized with training data")
        except Exception as e:
            print(f"  [FAIL] Failed to initialize drift detector: {e}")
    
    if os.path.exists(test_data_path):
        app_state.test_df = pd.read_csv(test_data_path)
        print(f"  [OK] Test data loaded ({len(app_state.test_df)} rows)")
    
    print("="*60)
    print("  AlertIQ API -- Ready")
    print("="*60)
    
    yield  # App is running
    
    # Shutdown
    print("AlertIQ API -- Shutting down")


# ─────────────────────────────────────────────
#  FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=(
        "Risk-Constrained Reinforcement Learning Framework for Adaptive AML Alert Governance. "
        "Evaluates suspicious transaction alerts and decides whether to suppress (auto-dismiss) "
        "or escalate (route to human analyst) using RL-optimized policies."
    ),
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for chart serving
os.makedirs(RL_OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/static/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


# ─────────────────────────────────────────────
#  Health Check
# ─────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "service": "AlertIQ",
        "version": API_VERSION,
        "status": "running",
        "models": {
            "suppression_model": app_state.suppression_model is not None,
            "rl_agent": app_state.rl_agent is not None,
            "rl_agent_type": app_state.rl_agent_type,
            "drift_detector": app_state.drift_detector is not None,
        }
    }


@app.get("/health", tags=["Health"])
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "components": {
            "suppression_model": "loaded" if app_state.suppression_model else "not loaded",
            "rl_agent": f"loaded ({app_state.rl_agent_type})" if app_state.rl_agent else "not loaded",
            "drift_detector": "active" if app_state.drift_detector else "not initialized",
            "training_data": "available" if app_state.train_df is not None else "not available",
            "test_data": "available" if app_state.test_df is not None else "not available",
        },
        "alerts_processed": app_state.alerts_processed,
    }


# ─────────────────────────────────────────────
#  Register Routers
# ─────────────────────────────────────────────
from api.routes.alerts import router as alerts_router
from api.routes.pipeline import router as pipeline_router
from api.routes.drift import router as drift_router
from api.routes.models import router as models_router

app.include_router(alerts_router, prefix="/api/alerts", tags=["Alerts"])
app.include_router(pipeline_router, prefix="/api/pipeline", tags=["Pipeline"])
app.include_router(drift_router, prefix="/api/drift", tags=["Drift Detection"])
app.include_router(models_router, prefix="/api/models", tags=["Models"])
