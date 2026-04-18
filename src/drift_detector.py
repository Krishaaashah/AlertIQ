"""
Phase 5 — Drift Detection & Safety Monitoring.

Detects distribution shift between training data and incoming data using:
  - Population Stability Index (PSI)
  - KL Divergence (Kullback-Leibler)

When drift is detected, the system activates a fallback mode that biases
the RL agent toward escalation to prevent missed fraud during unstable periods.

Industry Reference:
  PSI < 0.10 → No significant shift
  PSI 0.10–0.25 → Moderate shift (warning)
  PSI > 0.25 → Significant shift (critical — retrain required)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from datetime import datetime

from .config import (
    DRIFT_PSI_WARNING, DRIFT_PSI_CRITICAL,
    DRIFT_KL_WARNING, DRIFT_KL_CRITICAL,
    DRIFT_NUM_BINS, DRIFT_FALLBACK_BIAS
)


class DriftDetector:
    """
    Monitors distribution drift between reference (training) and target (live) data.
    
    Uses PSI and KL divergence to detect when the model's input distribution
    has shifted enough to warrant retraining or conservative fallback behavior.
    """
    
    def __init__(self, reference_df: pd.DataFrame, feature_columns: List[str] = None,
                 num_bins: int = DRIFT_NUM_BINS):
        """
        Initialize drift detector with reference (training) data.
        
        Args:
            reference_df: Training/reference DataFrame
            feature_columns: Columns to monitor (auto-detect numerical if None)
            num_bins: Number of bins for histogram comparison
        """
        self.num_bins = num_bins
        self.reference_df = reference_df.copy()
        
        # Auto-detect numerical columns if not specified
        if feature_columns is None:
            self.feature_columns = [
                c for c in reference_df.columns
                if pd.api.types.is_numeric_dtype(reference_df[c])
                and c not in ['isFraud', 'analyst_decision']
            ]
        else:
            self.feature_columns = feature_columns
        
        # Pre-compute reference distributions
        self.reference_distributions = {}
        for col in self.feature_columns:
            values = reference_df[col].dropna().values
            if len(values) > 0:
                hist, bin_edges = np.histogram(values, bins=num_bins, density=True)
                self.reference_distributions[col] = {
                    'hist': hist,
                    'bin_edges': bin_edges
                }
        
        # Drift history
        self.drift_history = []
    
    @staticmethod
    def _compute_psi(reference_hist: np.ndarray, target_hist: np.ndarray,
                     epsilon: float = 1e-6) -> float:
        """
        Compute Population Stability Index between two distributions.
        
        PSI = Σ (P_target - P_ref) * ln(P_target / P_ref)
        
        Args:
            reference_hist: Reference distribution (normalized histogram)
            target_hist: Target distribution (normalized histogram)
            epsilon: Small constant to prevent log(0)
            
        Returns:
            PSI value
        """
        # Normalize to proportions
        ref = reference_hist / (reference_hist.sum() + epsilon)
        tgt = target_hist / (target_hist.sum() + epsilon)
        
        # Add epsilon to prevent division by zero
        ref = np.clip(ref, epsilon, None)
        tgt = np.clip(tgt, epsilon, None)
        
        psi = np.sum((tgt - ref) * np.log(tgt / ref))
        return float(psi)
    
    @staticmethod
    def _compute_kl_divergence(reference_hist: np.ndarray, target_hist: np.ndarray,
                                epsilon: float = 1e-6) -> float:
        """
        Compute KL Divergence: D_KL(target || reference).
        
        Args:
            reference_hist: Reference distribution
            target_hist: Target distribution
            epsilon: Small constant to prevent log(0)
            
        Returns:
            KL divergence value
        """
        ref = reference_hist / (reference_hist.sum() + epsilon)
        tgt = target_hist / (target_hist.sum() + epsilon)
        
        ref = np.clip(ref, epsilon, None)
        tgt = np.clip(tgt, epsilon, None)
        
        kl = np.sum(tgt * np.log(tgt / ref))
        return float(kl)
    
    def check_drift(self, target_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for distribution drift between reference and target data.
        
        Args:
            target_df: New/live data to compare against reference
            
        Returns:
            Drift report with per-feature and aggregate metrics
        """
        feature_reports = []
        psi_values = []
        kl_values = []
        
        for col in self.feature_columns:
            if col not in target_df.columns or col not in self.reference_distributions:
                continue
            
            ref_info = self.reference_distributions[col]
            target_values = target_df[col].dropna().values
            
            if len(target_values) == 0:
                continue
            
            # Compute target histogram using same bin edges as reference
            target_hist, _ = np.histogram(target_values, bins=ref_info['bin_edges'], density=True)
            ref_hist = ref_info['hist']
            
            # Compute metrics
            psi = self._compute_psi(ref_hist, target_hist)
            kl = self._compute_kl_divergence(ref_hist, target_hist)
            
            psi_values.append(psi)
            kl_values.append(kl)
            
            # Per-feature severity
            if psi > DRIFT_PSI_CRITICAL:
                severity = "CRITICAL"
            elif psi > DRIFT_PSI_WARNING:
                severity = "WARNING"
            else:
                severity = "STABLE"
            
            feature_reports.append({
                'feature': col,
                'psi': round(psi, 6),
                'kl_divergence': round(kl, 6),
                'severity': severity
            })
        
        # Aggregate metrics
        avg_psi = np.mean(psi_values) if psi_values else 0.0
        max_psi = np.max(psi_values) if psi_values else 0.0
        avg_kl = np.mean(kl_values) if kl_values else 0.0
        
        # Overall drift severity
        if max_psi > DRIFT_PSI_CRITICAL:
            overall_severity = "CRITICAL"
        elif max_psi > DRIFT_PSI_WARNING:
            overall_severity = "WARNING"
        else:
            overall_severity = "STABLE"
        
        # Count drifted features
        drifted_features = [f for f in feature_reports if f['severity'] != 'STABLE']
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_severity': overall_severity,
            'avg_psi': round(avg_psi, 6),
            'max_psi': round(max_psi, 6),
            'avg_kl_divergence': round(avg_kl, 6),
            'total_features_monitored': len(feature_reports),
            'features_drifted': len(drifted_features),
            'feature_reports': feature_reports,
            'recommendation': self._get_recommendation(overall_severity),
            'fallback_active': overall_severity == "CRITICAL"
        }
        
        # Store in history
        self.drift_history.append(report)
        
        return report
    
    def _get_recommendation(self, severity: str) -> str:
        """Get actionable recommendation based on drift severity."""
        if severity == "CRITICAL":
            return (
                "CRITICAL DRIFT DETECTED. Input distribution has shifted significantly. "
                "Activating conservative fallback mode (bias toward escalation). "
                "Immediate model retraining recommended. Suppress rate will be reduced "
                f"to {(1 - DRIFT_FALLBACK_BIAS) * 100:.0f}% of normal."
            )
        elif severity == "WARNING":
            return (
                "MODERATE DRIFT DETECTED. Some features show distribution changes. "
                "Monitor closely and schedule retraining within 1-2 cycles. "
                "Current model predictions may be less reliable."
            )
        else:
            return "NO SIGNIFICANT DRIFT. Model operating within expected distribution bounds."
    
    def get_fallback_action_bias(self) -> float:
        """
        Get the current action bias for the RL agent.
        
        When drift is critical, returns a probability of forcing escalation
        to prevent missed fraud during unstable periods.
        
        Returns:
            Probability (0.0 to 1.0) of overriding suppress → escalate
        """
        if not self.drift_history:
            return 0.0  # No drift check has been run
        
        latest = self.drift_history[-1]
        if latest['overall_severity'] == "CRITICAL":
            return DRIFT_FALLBACK_BIAS
        elif latest['overall_severity'] == "WARNING":
            return DRIFT_FALLBACK_BIAS * 0.3  # Mild bias
        else:
            return 0.0
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get a summary of all drift checks performed."""
        if not self.drift_history:
            return {'message': 'No drift checks performed yet.', 'total_checks': 0}
        
        return {
            'total_checks': len(self.drift_history),
            'latest_severity': self.drift_history[-1]['overall_severity'],
            'latest_avg_psi': self.drift_history[-1]['avg_psi'],
            'latest_max_psi': self.drift_history[-1]['max_psi'],
            'critical_count': sum(1 for h in self.drift_history if h['overall_severity'] == 'CRITICAL'),
            'warning_count': sum(1 for h in self.drift_history if h['overall_severity'] == 'WARNING'),
            'stable_count': sum(1 for h in self.drift_history if h['overall_severity'] == 'STABLE'),
        }


def simulate_drift(reference_df: pd.DataFrame, drift_intensity: float = 0.3,
                   seed: int = 42) -> pd.DataFrame:
    """
    Simulate distribution drift by perturbing reference data.
    
    Useful for testing the drift detector without real production data.
    
    Args:
        reference_df: Original reference data
        drift_intensity: How much to shift (0.0 = no drift, 1.0 = extreme)
        seed: Random seed
        
    Returns:
        Drifted DataFrame
    """
    np.random.seed(seed)
    drifted = reference_df.copy()
    
    numerical_cols = [
        c for c in drifted.columns
        if pd.api.types.is_numeric_dtype(drifted[c])
        and c not in ['isFraud']
    ]
    
    for col in numerical_cols:
        col_std = drifted[col].std()
        if col_std > 0:
            # Add Gaussian noise proportional to drift intensity
            noise = np.random.normal(0, col_std * drift_intensity, size=len(drifted))
            drifted[col] = drifted[col] + noise
    
    return drifted


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    ref = pd.DataFrame({
        'amount': np.random.lognormal(10, 2, 1000),
        'step': np.random.randint(1, 100, 1000),
        'prob_fraud': np.random.uniform(0, 0.3, 1000),
        'isFraud': np.random.choice([0, 1], 1000, p=[0.95, 0.05])
    })
    
    detector = DriftDetector(ref)
    
    # Test with no drift
    report = detector.check_drift(ref)
    print(f"No-drift test: {report['overall_severity']} (PSI={report['avg_psi']:.4f})")
    
    # Test with simulated drift
    drifted = simulate_drift(ref, drift_intensity=0.5)
    report = detector.check_drift(drifted)
    print(f"Drifted test: {report['overall_severity']} (PSI={report['avg_psi']:.4f})")
    print(f"Recommendation: {report['recommendation']}")
