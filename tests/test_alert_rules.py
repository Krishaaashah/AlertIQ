"""Tests for src/alert_rules.py — the Phase 1 rule-based alert engine."""

from src.alert_rules import generate_alerts


def test_generate_alerts_returns_expected_columns(encoded_sample_df):
    alerts = generate_alerts(encoded_sample_df)

    for col in ["R1", "R2", "R3", "R4", "rule_count", "alert_flag", "alert_reason"]:
        assert col in alerts.columns


def test_generate_alerts_only_keeps_flagged_rows(encoded_sample_df):
    alerts = generate_alerts(encoded_sample_df)

    # generate_alerts() filters down to alert_flag == 1 rows only
    assert (alerts["alert_flag"] == 1).all()
    assert len(alerts) <= len(encoded_sample_df)


def test_high_value_rule_flags_top_quantile(encoded_sample_df):
    alerts = generate_alerts(encoded_sample_df)

    # Every R1-flagged alert should be at or above the 95th percentile of
    # amount in the ORIGINAL (pre-filter) data.
    threshold = encoded_sample_df["amount"].quantile(0.95)
    r1_alerts = alerts[alerts["R1"] == 1]
    assert (r1_alerts["amount"] >= threshold - 1e-6).all()


def test_rule_engine_does_not_drop_fraud(encoded_sample_df):
    """
    Sanity check matching the project's headline claim: the rule layer is
    tuned to be a high-recall (not high-precision) filter, so it should
    almost never let fraud through un-flagged on realistic data.
    """
    alerts = generate_alerts(encoded_sample_df)

    total_fraud = encoded_sample_df["isFraud"].sum()
    caught_fraud = alerts["isFraud"].sum()

    if total_fraud > 0:
        recall = caught_fraud / total_fraud
        assert recall >= 0.9, f"Rule engine fraud recall dropped to {recall:.2%}"


def test_alert_reason_is_a_known_rule(encoded_sample_df):
    alerts = generate_alerts(encoded_sample_df)
    assert set(alerts["alert_reason"].unique()) <= {"R1", "R2", "R3", "R4", "none"}
