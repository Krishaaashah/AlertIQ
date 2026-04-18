"""
Pipeline execution endpoints.

POST /api/pipeline/run      — Trigger pipeline execution
GET  /api/pipeline/status   — Get pipeline status
GET  /api/pipeline/results  — Get latest results
"""

import os
import threading
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import (
    PipelineRunRequest, PipelineStatusResponse, PipelineResultsResponse,
    PipelineStatus
)
from src.config import OUTPUT_DIR, RL_OUTPUT_DIR, RAW_DATA_PATH

router = APIRouter()


def _get_app_state():
    from api.main import app_state
    return app_state


def _run_pipeline_worker(phases: list, rl_episodes: int,
                          train_dqn: bool, train_bandit: bool):
    """Background worker to run the pipeline phases."""
    app_state = _get_app_state()
    
    try:
        if "phase2" in phases:
            app_state.pipeline_status = "RUNNING"
            app_state.pipeline_message = "Phase 2: Generating alerts and simulating analysts..."
            
            from src.data_loader import load_and_clean
            from src.alert_rules import generate_alerts
            from src.analyst_simulator import simulate_analysts
            from src.dataset_builder import build_and_split
            
            df = load_and_clean(RAW_DATA_PATH)
            df_alerts = generate_alerts(df)
            df_feedback = simulate_analysts(df_alerts)
            build_and_split(df_feedback, OUTPUT_DIR)
            
            app_state.pipeline_message = "Phase 2 complete."
        
        if "phase3" in phases:
            app_state.pipeline_message = "Phase 3: Training suppression model..."
            
            import joblib
            from src.suppression_model import load_and_prepare_data, train_suppression_model
            from src.gating_evaluator import evaluate_thresholds
            from src.config import SUPPRESSION_MODEL_PATH, MODEL_DIR
            
            train_path = os.path.join(OUTPUT_DIR, 'train_feedback.csv')
            test_path = os.path.join(OUTPUT_DIR, 'test_feedback.csv')
            
            train_df, test_df = load_and_prepare_data(train_path, test_path)
            model, eval_df = train_suppression_model(train_df, test_df)
            
            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump(model, SUPPRESSION_MODEL_PATH)
            evaluate_thresholds(eval_df, output_dir=OUTPUT_DIR)
            
            app_state.suppression_model = model
            app_state.pipeline_message = "Phase 3 complete."
        
        if "phase4" in phases:
            app_state.pipeline_message = "Phase 4: Training RL agents..."
            
            from src.rl_pipeline import RLDecisionOptimizer
            
            optimizer = RLDecisionOptimizer(
                data_dir=OUTPUT_DIR, outputs_dir=RL_OUTPUT_DIR
            )
            results = optimizer.run_full_pipeline(
                num_episodes_dqn=rl_episodes if train_dqn else 0,
                num_episodes_bandit=rl_episodes if train_bandit else 0,
                train_dqn=train_dqn,
                train_bandit=train_bandit
            )
            
            # Reload RL agent
            from src.rl_trainer import load_policy
            dqn_path = os.path.join(RL_OUTPUT_DIR, 'dqn_agent.pkl')
            bandit_path = os.path.join(RL_OUTPUT_DIR, 'bandit_agent.pkl')
            
            if train_dqn and os.path.exists(dqn_path):
                app_state.rl_agent = load_policy(dqn_path, agent_type='dqn', device='cpu')
                app_state.rl_agent_type = 'dqn'
            elif train_bandit and os.path.exists(bandit_path):
                app_state.rl_agent = load_policy(bandit_path, agent_type='bandit', device='cpu')
                app_state.rl_agent_type = 'bandit'
            
            app_state.pipeline_message = "Phase 4 complete."
        
        if "phase5" in phases:
            app_state.pipeline_message = "Phase 5: Running drift detection..."
            
            from src.drift_detector import DriftDetector
            train_path = os.path.join(OUTPUT_DIR, 'train_feedback.csv')
            test_path = os.path.join(OUTPUT_DIR, 'test_feedback.csv')
            
            if os.path.exists(train_path) and os.path.exists(test_path):
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)
                detector = DriftDetector(train_df)
                detector.check_drift(test_df)
                app_state.drift_detector = detector
            
            app_state.pipeline_message = "Phase 5 complete."
        
        app_state.pipeline_status = "COMPLETED"
        app_state.pipeline_message = "All requested phases completed successfully."
        
    except Exception as e:
        app_state.pipeline_status = "FAILED"
        app_state.pipeline_message = f"Pipeline failed: {str(e)}"


@router.post("/run", response_model=PipelineStatusResponse)
async def run_pipeline(request: PipelineRunRequest):
    """Trigger pipeline execution (runs in background)."""
    app_state = _get_app_state()
    
    if app_state.pipeline_status == "RUNNING":
        raise HTTPException(status_code=409, detail="Pipeline is already running")
    
    # Start pipeline in background thread
    thread = threading.Thread(
        target=_run_pipeline_worker,
        args=(request.phases, request.rl_episodes, request.train_dqn, request.train_bandit),
        daemon=True
    )
    thread.start()
    
    app_state.pipeline_status = "RUNNING"
    app_state.pipeline_message = "Pipeline started..."
    
    return PipelineStatusResponse(
        status=PipelineStatus.RUNNING,
        current_phase=request.phases[0] if request.phases else None,
        progress=0.0,
        message="Pipeline execution started in background."
    )


@router.get("/status", response_model=PipelineStatusResponse)
async def pipeline_status():
    """Get current pipeline status."""
    app_state = _get_app_state()
    
    return PipelineStatusResponse(
        status=PipelineStatus(app_state.pipeline_status),
        message=app_state.pipeline_message,
    )


@router.get("/results", response_model=PipelineResultsResponse)
async def pipeline_results():
    """Get latest pipeline results."""
    app_state = _get_app_state()
    
    phase2_done = os.path.exists(os.path.join(OUTPUT_DIR, 'train_feedback.csv'))
    phase3_done = os.path.exists(os.path.join(OUTPUT_DIR, 'threshold_report.csv'))
    phase4_done = os.path.exists(os.path.join(RL_OUTPUT_DIR, 'policy_comparison.csv'))
    
    comparison = None
    if phase4_done:
        try:
            comp_df = pd.read_csv(os.path.join(RL_OUTPUT_DIR, 'policy_comparison.csv'))
            comparison = comp_df.to_dict(orient='records')
        except Exception:
            pass
    
    metrics = None
    if phase3_done:
        try:
            thresh_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'threshold_report.csv'))
            metrics = {
                'threshold_analysis': thresh_df.to_dict(orient='records')
            }
        except Exception:
            pass
    
    return PipelineResultsResponse(
        phase2_complete=phase2_done,
        phase3_complete=phase3_done,
        phase4_complete=phase4_done,
        phase5_complete=app_state.drift_detector is not None,
        metrics=metrics,
        comparison=comparison,
    )
