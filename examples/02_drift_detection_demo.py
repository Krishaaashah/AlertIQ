"""
Example 2 — Drift detection and safety fallback (Phase 5)

Shows the DriftDetector comparing a 'live' batch of transactions against a
reference distribution, first with no drift, then with simulated drift, and
what the resulting fallback recommendation looks like.

Run:
    python examples/02_drift_detection_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.drift_detector import DriftDetector, simulate_drift

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "dataset", "sample_paysim.csv")


def main():
    if not os.path.exists(DATA_PATH):
        os.system(f"{sys.executable} dataset/generate_sample.py")

    reference = pd.read_csv(DATA_PATH)
    detector = DriftDetector(reference)

    print("=== Scenario A: incoming batch matches training distribution ===")
    stable_report = detector.check_drift(reference)
    print(f"Severity: {stable_report['overall_severity']}")
    print(f"Avg PSI:  {stable_report['avg_psi']:.4f}")
    print(f"Recommendation: {stable_report['recommendation']}\n")

    print("=== Scenario B: incoming batch has drifted (simulated) ===")
    drifted_batch = simulate_drift(reference, drift_intensity=2.5, seed=99)
    drifted_report = detector.check_drift(drifted_batch)
    print(f"Severity: {drifted_report['overall_severity']}")
    print(f"Avg PSI:  {drifted_report['avg_psi']:.4f}")
    print(f"Max PSI:  {drifted_report['max_psi']:.4f}")
    print(f"Fallback active: {drifted_report['fallback_active']}")
    print(f"Recommendation: {drifted_report['recommendation']}\n")

    print("Per-feature detail (drifted scenario):")
    for feat in drifted_report["feature_reports"]:
        print(f"  {feat['feature']:>20s}  PSI={feat['psi']:.4f}  [{feat['severity']}]")

    print(f"\nRL action bias after this check: {detector.get_fallback_action_bias():.2f}")
    print("(0.0 = no override, higher = stronger bias toward forced escalation)")


if __name__ == "__main__":
    main()
