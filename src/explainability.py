"""
Phase 6 — Explainability Engine.

Generates human-readable explanations for each suppress/escalate decision.
Critical for AML compliance — regulators require that automated decisions
can be justified and audited.

Explanation strategies:
  1. Feature-based reasoning (probability, amount, rule triggers)
  2. Policy-based reasoning (why the RL agent chose this action)
  3. Bandit weight interpretation (for contextual bandit agent)
  4. Risk factor enumeration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional


class ExplainabilityEngine:
    """
    Generates structured, human-readable explanations for alert decisions.
    
    Designed for regulatory compliance (Basel III / FATF guidelines) where
    automated AML decisions must be explainable and auditable.
    """
    
    # Explanation templates for each scenario
    TEMPLATES = {
        'suppress_low_risk': (
            "SUPPRESSED — Low-risk alert. {reasons}. "
            "Model confidence: {confidence:.1f}%. "
            "This alert matches historical patterns of false positives."
        ),
        'suppress_moderate': (
            "SUPPRESSED — Moderate confidence. {reasons}. "
            "Model confidence: {confidence:.1f}%. "
            "Alert profile is consistent with benign activity, but flagged for audit trail."
        ),
        'escalate_high_risk': (
            "ESCALATED — High fraud risk detected. {reasons}. "
            "Fraud probability: {fraud_prob:.1f}%. "
            "Immediate analyst review required."
        ),
        'escalate_caution': (
            "ESCALATED — Precautionary. {reasons}. "
            "Fraud probability: {fraud_prob:.1f}%. "
            "Risk level warrants human verification despite moderate indicators."
        ),
        'escalate_drift': (
            "ESCALATED — Safety override. System drift detected. "
            "Normal suppression suspended. All borderline alerts routed to analyst "
            "until model stability is confirmed."
        ),
    }
    
    # Human-readable alert reason mappings
    RULE_DESCRIPTIONS = {
        'R1': 'High-value transaction (top 5% by amount)',
        'R2': 'Burst activity (multiple rapid transactions)',
        'R3': 'High-risk transaction type (CASH_OUT/TRANSFER)',
        'R4': 'Severe balance drain (>90% of account balance)',
        'none': 'No specific rule trigger'
    }
    
    def __init__(self, bandit_agent=None):
        """
        Initialize explainability engine.
        
        Args:
            bandit_agent: Optional ContextualBanditAgent for weight-based explanations
        """
        self.bandit_agent = bandit_agent
    
    def explain_decision(self, action: int, state: np.ndarray,
                         alert_data: Dict[str, Any] = None,
                         drift_active: bool = False) -> Dict[str, Any]:
        """
        Generate a structured explanation for a suppress/escalate decision.
        
        Args:
            action: 0 = SUPPRESS, 1 = ESCALATE
            state: [prob_fraud, rule_count_norm, amount_norm, fraud_rate, workload]
            alert_data: Optional raw alert data for richer context
            drift_active: Whether drift fallback mode is active
            
        Returns:
            Structured explanation dictionary
        """
        prob_fraud = state[0]
        rule_count_norm = state[1]
        amount_norm = state[2]
        fraud_rate = state[3]
        workload = state[4]
        
        prob_benign = 1.0 - prob_fraud
        
        # Build risk factors
        risk_factors = self._identify_risk_factors(
            prob_fraud, rule_count_norm, amount_norm, fraud_rate, alert_data
        )
        
        # Build reasons string
        reasons = "; ".join(risk_factors) if risk_factors else "Standard profile"
        
        # Select explanation template
        if drift_active and action == 1:
            template_key = 'escalate_drift'
        elif action == 0:  # SUPPRESS
            if prob_benign > 0.95:
                template_key = 'suppress_low_risk'
            else:
                template_key = 'suppress_moderate'
        else:  # ESCALATE
            if prob_fraud > 0.5:
                template_key = 'escalate_high_risk'
            else:
                template_key = 'escalate_caution'
        
        # Generate explanation text
        explanation_text = self.TEMPLATES[template_key].format(
            reasons=reasons,
            confidence=prob_benign * 100,
            fraud_prob=prob_fraud * 100
        )
        
        # Build structured response
        explanation = {
            'decision': 'SUPPRESS' if action == 0 else 'ESCALATE',
            'explanation': explanation_text,
            'confidence_score': round(float(prob_benign if action == 0 else prob_fraud), 4),
            'risk_level': self._classify_risk(prob_fraud),
            'risk_factors': risk_factors,
            'state_features': {
                'fraud_probability': round(float(prob_fraud), 4),
                'benign_probability': round(float(prob_benign), 4),
                'rule_intensity': round(float(rule_count_norm), 4),
                'amount_severity': round(float(amount_norm), 4),
                'system_fraud_rate': round(float(fraud_rate), 4),
                'workload_level': round(float(workload), 4),
            },
            'drift_override': drift_active,
        }
        
        # Add bandit feature importance if available
        if self.bandit_agent is not None:
            explanation['feature_importance'] = self._get_bandit_reasoning(action)
        
        return explanation
    
    def _identify_risk_factors(self, prob_fraud: float, rule_count_norm: float,
                                amount_norm: float, fraud_rate: float,
                                alert_data: Dict = None) -> List[str]:
        """Identify human-readable risk factors from state and alert data."""
        factors = []
        
        # Fraud probability assessment
        if prob_fraud > 0.7:
            factors.append(f"Very high fraud probability ({prob_fraud:.1%})")
        elif prob_fraud > 0.3:
            factors.append(f"Elevated fraud probability ({prob_fraud:.1%})")
        else:
            factors.append(f"Low fraud probability ({prob_fraud:.1%})")
        
        # Rule trigger intensity
        if rule_count_norm > 0.75:
            factors.append("Multiple alert rules triggered (high suspicion)")
        elif rule_count_norm > 0.25:
            factors.append("Some alert rules triggered")
        
        # Transaction amount
        if amount_norm > 0.8:
            factors.append("Unusually large transaction amount")
        elif amount_norm > 0.5:
            factors.append("Above-average transaction amount")
        
        # System fraud rate context
        if fraud_rate > 0.05:
            factors.append(f"Elevated system fraud rate ({fraud_rate:.2%})")
        
        # Alert-specific reasons
        if alert_data:
            alert_reason = alert_data.get('alert_reason', 'none')
            if alert_reason in self.RULE_DESCRIPTIONS:
                factors.append(self.RULE_DESCRIPTIONS[alert_reason])
            
            # Balance drain detection
            if alert_data.get('balance_drain_ratio', 0) > 0.8:
                factors.append(f"Significant balance drain ({alert_data['balance_drain_ratio']:.0%})")
        
        return factors
    
    def _classify_risk(self, prob_fraud: float) -> str:
        """Classify overall risk level."""
        if prob_fraud > 0.7:
            return "HIGH"
        elif prob_fraud > 0.3:
            return "MEDIUM"
        elif prob_fraud > 0.1:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _get_bandit_reasoning(self, action: int) -> Dict[str, Any]:
        """Extract feature importance from the contextual bandit agent."""
        if not hasattr(self.bandit_agent, 'get_feature_importance'):
            return {}
        
        importance = self.bandit_agent.get_feature_importance()
        action_name = 'SUPPRESS' if action == 0 else 'ESCALATE'
        weights = importance.get(action_name, np.zeros(5))
        feature_names = importance.get('feature_names', 
                                        ['prob_fraud', 'rule_count', 'amount', 'fraud_rate', 'workload'])
        
        # Sort by absolute importance
        sorted_indices = np.argsort(np.abs(weights))[::-1]
        
        return {
            'action': action_name,
            'top_factors': [
                {
                    'feature': feature_names[i],
                    'weight': round(float(weights[i]), 4),
                    'direction': 'positive' if weights[i] > 0 else 'negative'
                }
                for i in sorted_indices
            ]
        }
    
    def explain_batch(self, actions: np.ndarray, states: np.ndarray,
                      alert_data_list: List[Dict] = None,
                      drift_active: bool = False) -> List[Dict[str, Any]]:
        """
        Generate explanations for a batch of decisions.
        
        Args:
            actions: Array of actions (0/1)
            states: Array of state vectors
            alert_data_list: Optional list of alert data dicts
            drift_active: Whether drift fallback is active
            
        Returns:
            List of explanation dictionaries
        """
        explanations = []
        for i in range(len(actions)):
            alert_data = alert_data_list[i] if alert_data_list else None
            exp = self.explain_decision(
                action=int(actions[i]),
                state=states[i],
                alert_data=alert_data,
                drift_active=drift_active
            )
            explanations.append(exp)
        return explanations
    
    def generate_audit_record(self, decision_explanation: Dict[str, Any],
                               alert_id: str = None) -> Dict[str, Any]:
        """
        Generate a compliance-ready audit record for a decision.
        
        Args:
            decision_explanation: Output from explain_decision()
            alert_id: Optional alert identifier
            
        Returns:
            Audit record suitable for regulatory logging
        """
        from datetime import datetime
        
        return {
            'audit_id': f"AUD-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'alert_id': alert_id or 'N/A',
            'timestamp': datetime.now().isoformat(),
            'decision': decision_explanation['decision'],
            'risk_level': decision_explanation['risk_level'],
            'confidence': decision_explanation['confidence_score'],
            'explanation_text': decision_explanation['explanation'],
            'risk_factors': decision_explanation['risk_factors'],
            'model_features': decision_explanation['state_features'],
            'drift_override': decision_explanation.get('drift_override', False),
            'reviewable': True,
            'system': 'AlertIQ v1.0'
        }
