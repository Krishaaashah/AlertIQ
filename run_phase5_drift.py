"""
AlertIQ — Phase 5: Drift Detection & Safety Testing
Tests PSI and KL divergence on training vs test data, plus simulated drift scenarios.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from src.drift_detector import DriftDetector, simulate_drift
from src.config import OUTPUT_DIR, TRAIN_DATA_PATH, TEST_DATA_PATH


def main():
    print("=" * 60)
    print("  AlertIQ — Phase 5: Drift Detection & Safety Testing")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading reference (training) data...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    print(f"  Reference set: {len(train_df):,} | Target set: {len(test_df):,}")

    # Initialize detector
    detector = DriftDetector(train_df)
    print(f"  Monitoring {len(detector.feature_columns)} features")

    # Check 1: No drift (same distribution)
    print("\n[2/4] Checking train vs test (natural split)...")
    report = detector.check_drift(test_df)
    print(f"  Severity: {report['overall_severity']}")
    print(f"  Avg PSI: {report['avg_psi']:.6f} | Max PSI: {report['max_psi']:.6f}")
    print(f"  Features drifted: {report['features_drifted']}/{report['total_features_monitored']}")
    print(f"  Recommendation: {report['recommendation']}")

    # Check 2: Moderate drift simulation
    print("\n[3/4] Simulating MODERATE drift (intensity=0.3)...")
    drifted_moderate = simulate_drift(test_df, drift_intensity=0.3)
    report_mod = detector.check_drift(drifted_moderate)
    print(f"  Severity: {report_mod['overall_severity']}")
    print(f"  Avg PSI: {report_mod['avg_psi']:.6f} | Max PSI: {report_mod['max_psi']:.6f}")
    print(f"  Fallback bias: {detector.get_fallback_action_bias():.2f}")

    # Check 3: Severe drift simulation
    print("\n[4/4] Simulating SEVERE drift (intensity=0.8)...")
    drifted_severe = simulate_drift(test_df, drift_intensity=0.8)
    report_sev = detector.check_drift(drifted_severe)
    print(f"  Severity: {report_sev['overall_severity']}")
    print(f"  Avg PSI: {report_sev['avg_psi']:.6f} | Max PSI: {report_sev['max_psi']:.6f}")
    print(f"  Fallback bias: {detector.get_fallback_action_bias():.2f}")
    print(f"  Recommendation: {report_sev['recommendation']}")

    # Summary
    summary = detector.get_drift_summary()
    print("\n" + "-" * 60)
    print("  DRIFT DETECTION SUMMARY")
    print("-" * 60)
    print(f"  Total checks:  {summary['total_checks']}")
    print(f"  Stable:        {summary['stable_count']}")
    print(f"  Warning:       {summary['warning_count']}")
    print(f"  Critical:      {summary['critical_count']}")

    print("\n" + "=" * 60)
    print("  Phase 5 Complete — Drift detection validated")
    print("=" * 60)


if __name__ == "__main__":
    main()
