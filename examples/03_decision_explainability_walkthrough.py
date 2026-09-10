"""
Example 3 — Decision + explainability walkthrough (Phases 3-6)

Builds a small evaluation frame (as if a calibrated ML model had already
scored each alert with prob_fraud), runs each alert through the adaptive
threshold policy — the same interpretable baseline the RL agent is
benchmarked against — and prints an audit-style explanation for each
decision using the real ExplainabilityEngine.

This intentionally avoids requiring a trained DQN checkpoint (which needs
torch and a full training run) so it can run anywhere in a few seconds.
To exercise the actual DQN policy instead of the baseline, train one first
with `python run_phase4_rl.py` and see `run_phase4_rl.py` for how the
trained agent is loaded and queried.

Run:
    python examples/03_decision_explainability_walkthrough.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.rl_environment import AdaptiveThresholdPolicy
from src.explainability import ExplainabilityEngine

# A handful of illustrative alerts spanning the risk spectrum. In production
# these prob_fraud values come from the Phase 3 calibrated suppression model
# (see src/suppression_model.py); here they're set explicitly for clarity.
DEMO_ALERTS = [
    {"label": "Small routine payment",     "prob_fraud": 0.02, "rule_count": 1, "amount": 850.0,
     "alert_reason": "R1", "balance_drain_ratio": 0.05},
    {"label": "Large but explainable transfer", "prob_fraud": 0.18, "rule_count": 1, "amount": 45000.0,
     "alert_reason": "R1", "balance_drain_ratio": 0.10},
    {"label": "Burst of rapid transactions", "prob_fraud": 0.42, "rule_count": 2, "amount": 12000.0,
     "alert_reason": "R2", "balance_drain_ratio": 0.30},
    {"label": "Near-total balance drain",   "prob_fraud": 0.91, "rule_count": 3, "amount": 98000.0,
     "alert_reason": "R4", "balance_drain_ratio": 0.97},
]


def build_state(alert: dict) -> np.ndarray:
    """Build the 5-D RL state vector the same way rl_environment.py does."""
    amount_log = np.log1p(alert["amount"])
    max_log_amount = np.log1p(3_000_000)  # rough ceiling matching PaySim's range
    amount_norm = min(amount_log / max_log_amount, 1.0)

    return np.array([
        alert["prob_fraud"],
        alert["rule_count"] / 4.0,
        amount_norm,
        0.013,   # typical system fraud rate
        0.5,     # mid-range workload
    ], dtype=np.float32)


def main():
    policy = AdaptiveThresholdPolicy(base_threshold=0.90)
    explainer = ExplainabilityEngine()

    print(f"{'Alert':38s} {'ProbFraud':>10s}  {'Decision':>10s}  {'Risk':>8s}")
    print("-" * 74)

    for alert in DEMO_ALERTS:
        state = build_state(alert)
        action = policy.decide(state)
        explanation = explainer.explain_decision(action=action, state=state, alert_data=alert)

        print(f"{alert['label']:38s} {alert['prob_fraud']*100:>9.1f}%  "
              f"{explanation['decision']:>10s}  {explanation['risk_level']:>8s}")

    print("\nFull explanation for the highest-risk alert:\n")
    alert = DEMO_ALERTS[-1]
    state = build_state(alert)
    action = policy.decide(state)
    explanation = explainer.explain_decision(action=action, state=state, alert_data=alert)
    print(f"  Decision:    {explanation['decision']}")
    print(f"  Confidence:  {explanation['confidence_score']:.1%}")
    print(f"  Risk level:  {explanation['risk_level']}")
    print(f"  Explanation: {explanation['explanation']}")
    print(f"  Risk factors:")
    for factor in explanation["risk_factors"]:
        print(f"    - {factor}")


if __name__ == "__main__":
    main()
