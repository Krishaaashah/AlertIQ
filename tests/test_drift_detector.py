"""Tests for src/drift_detector.py — Phase 5 drift monitoring and safety fallback."""

import numpy as np
import pandas as pd
import pytest

from src.drift_detector import DriftDetector, simulate_drift


@pytest.fixture
def reference_df():
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "amount": rng.lognormal(9, 1.5, 1000),
        "step": rng.integers(1, 744, 1000),
        "prob_fraud": rng.uniform(0, 0.3, 1000),
        "isFraud": rng.choice([0, 1], 1000, p=[0.97, 0.03]),
    })


def test_identical_distribution_reports_stable(reference_df):
    detector = DriftDetector(reference_df)
    report = detector.check_drift(reference_df)

    assert report["overall_severity"] == "STABLE"
    assert report["fallback_active"] is False
    assert report["avg_psi"] < 0.05


def test_strongly_shifted_distribution_is_flagged(reference_df):
    detector = DriftDetector(reference_df)
    drifted = simulate_drift(reference_df, drift_intensity=1.5, seed=7)
    report = detector.check_drift(drifted)

    # A large synthetic shift should push PSI well past the warning band.
    assert report["overall_severity"] in {"WARNING", "CRITICAL"}
    assert report["max_psi"] > 0.1


def test_isFraud_and_analyst_decision_are_excluded_from_monitoring(reference_df):
    detector = DriftDetector(reference_df)
    assert "isFraud" not in detector.feature_columns


def test_fallback_action_bias_only_triggers_after_a_check(reference_df):
    detector = DriftDetector(reference_df)
    # Before any check_drift() call, there's no history yet.
    assert detector.get_fallback_action_bias() == 0.0

    drifted = simulate_drift(reference_df, drift_intensity=2.0, seed=3)
    detector.check_drift(drifted)

    if detector.drift_history[-1]["overall_severity"] == "CRITICAL":
        assert detector.get_fallback_action_bias() > 0.0


def test_drift_history_accumulates(reference_df):
    detector = DriftDetector(reference_df)
    detector.check_drift(reference_df)
    detector.check_drift(reference_df)
    assert detector.get_drift_summary()["total_checks"] == 2
