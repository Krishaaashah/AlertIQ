"""
Pydantic schemas for AlertIQ API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


# ─────────────────────────────────────────────
#  Enums
# ─────────────────────────────────────────────

class DecisionEnum(str, Enum):
    SUPPRESS = "SUPPRESS"
    ESCALATE = "ESCALATE"


class RiskLevel(str, Enum):
    MINIMAL = "MINIMAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class DriftSeverity(str, Enum):
    STABLE = "STABLE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class PipelineStatus(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# ─────────────────────────────────────────────
#  Alert Evaluation
# ─────────────────────────────────────────────

class AlertFeatures(BaseModel):
    """Features for a single alert to be evaluated."""
    amount: float = Field(..., description="Transaction amount")
    step: int = Field(0, description="Time step")
    rule_count: int = Field(1, description="Number of triggered alert rules (0-4)")
    alert_reason: str = Field("R1", description="Primary alert rule (R1/R2/R3/R4)")
    balance_drain_ratio: float = Field(0.0, description="Ratio of balance drained (0-1)")
    oldbalanceOrg: float = Field(0.0, description="Sender's original balance")
    newbalanceOrig: float = Field(0.0, description="Sender's new balance after txn")
    oldbalanceDest: float = Field(0.0, description="Receiver's original balance")
    newbalanceDest: float = Field(0.0, description="Receiver's new balance after txn")
    type_CASH_OUT: int = Field(0, description="Binary: is CASH_OUT type")
    type_DEBIT: int = Field(0, description="Binary: is DEBIT type")
    type_PAYMENT: int = Field(0, description="Binary: is PAYMENT type")
    type_TRANSFER: int = Field(0, description="Binary: is TRANSFER type")


class AlertDecisionResponse(BaseModel):
    """Response for a single alert evaluation."""
    decision: DecisionEnum
    confidence: float = Field(..., description="Confidence score (0-1)")
    risk_level: RiskLevel
    fraud_probability: float
    explanation: str
    risk_factors: List[str]
    state_features: Dict[str, float]
    drift_override: bool = False


class BatchAlertRequest(BaseModel):
    """Request for batch alert evaluation."""
    alerts: List[AlertFeatures]


class BatchAlertResponse(BaseModel):
    """Response for batch alert evaluation."""
    total_alerts: int
    suppressed: int
    escalated: int
    decisions: List[AlertDecisionResponse]
    summary: Dict[str, Any]


# ─────────────────────────────────────────────
#  System Stats
# ─────────────────────────────────────────────

class SystemStats(BaseModel):
    """Current system statistics."""
    total_alerts_processed: int = 0
    fraud_rate: float = 0.0
    suppression_rate: float = 0.0
    escalation_rate: float = 0.0
    model_loaded: bool = False
    rl_agent_loaded: bool = False
    drift_status: DriftSeverity = DriftSeverity.STABLE


# ─────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────

class PipelineRunRequest(BaseModel):
    """Request to trigger pipeline execution."""
    phases: List[str] = Field(
        default=["phase2", "phase3", "phase4"],
        description="Pipeline phases to run"
    )
    rl_episodes: int = Field(default=50, description="RL training episodes")
    train_dqn: bool = Field(default=True)
    train_bandit: bool = Field(default=True)


class PipelineStatusResponse(BaseModel):
    """Pipeline execution status."""
    status: PipelineStatus
    current_phase: Optional[str] = None
    progress: float = 0.0
    message: str = ""


class PipelineResultsResponse(BaseModel):
    """Pipeline execution results."""
    phase2_complete: bool = False
    phase3_complete: bool = False
    phase4_complete: bool = False
    phase5_complete: bool = False
    metrics: Optional[Dict[str, Any]] = None
    comparison: Optional[List[Dict[str, Any]]] = None


# ─────────────────────────────────────────────
#  Drift Detection
# ─────────────────────────────────────────────

class DriftCheckResponse(BaseModel):
    """Response from drift check."""
    overall_severity: DriftSeverity
    avg_psi: float
    max_psi: float
    avg_kl_divergence: float
    total_features_monitored: int
    features_drifted: int
    recommendation: str
    fallback_active: bool
    feature_reports: List[Dict[str, Any]]


# ─────────────────────────────────────────────
#  Model Info
# ─────────────────────────────────────────────

class ModelInfoResponse(BaseModel):
    """Information about loaded models."""
    suppression_model: Dict[str, Any]
    rl_agent: Dict[str, Any]
    drift_detector: Dict[str, Any]


class ComparisonResponse(BaseModel):
    """RL vs baseline policy comparison results."""
    comparison_data: List[Dict[str, Any]]
    best_policy: str
    recommendation: str
