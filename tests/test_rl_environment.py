"""
Tests for src/rl_environment.py — the Phase 3/4 decision environment and the
static-threshold baseline policy it's benchmarked against.

These tests deliberately avoid importing src.rl_agent (the DQN itself),
since that module requires torch, which is a heavier optional dependency.
The environment and reward logic tested here are torch-free by design.
"""

import numpy as np

from src.rl_environment import AlertSuppressionEnvironment, AdaptiveThresholdPolicy


def test_reward_matches_asymmetric_spec(synthetic_eval_df):
    """
    Values are asserted against src/config.py (the single source of truth
    used at runtime), not against the older numbers quoted in this module's
    docstring — config.py currently uses -500 / +1 / +5 / -0.5.
    """
    from src.config import (
        RL_REWARD_SUPPRESS_FRAUD, RL_REWARD_SUPPRESS_BENIGN,
        RL_REWARD_ESCALATE_FRAUD, RL_REWARD_ESCALATE_BENIGN,
    )

    env = AlertSuppressionEnvironment(synthetic_eval_df)

    # Suppressing real fraud must be the worst outcome by a wide margin.
    assert env._compute_reward(action=0, is_fraud=True) == RL_REWARD_SUPPRESS_FRAUD
    # Suppressing a benign alert is the desired, rewarded outcome.
    assert env._compute_reward(action=0, is_fraud=False) == RL_REWARD_SUPPRESS_BENIGN
    # Escalating real fraud is correct and rewarded.
    assert env._compute_reward(action=1, is_fraud=True) == RL_REWARD_ESCALATE_FRAUD
    # Escalating a benign alert costs a little analyst time, but is safe.
    assert env._compute_reward(action=1, is_fraud=False) == RL_REWARD_ESCALATE_BENIGN

    # The core safety property of the whole system: missing fraud must be
    # penalized far more harshly than any other outcome is rewarded.
    penalty = abs(RL_REWARD_SUPPRESS_FRAUD)
    best_reward = max(RL_REWARD_SUPPRESS_BENIGN, RL_REWARD_ESCALATE_FRAUD)
    assert penalty > best_reward * 10


def test_state_vector_shape_and_bounds(synthetic_eval_df):
    env = AlertSuppressionEnvironment(synthetic_eval_df)
    state = env.reset()

    assert state.shape == (5,)
    prob_fraud, rule_count_norm, amount_norm, fraud_rate, workload = state
    assert 0.0 <= prob_fraud <= 1.0
    assert 0.0 <= rule_count_norm <= 1.0
    assert 0.0 <= amount_norm <= 1.0
    assert 0.0 <= workload <= 1.0


def test_episode_runs_to_completion_and_summarizes(synthetic_eval_df):
    env = AlertSuppressionEnvironment(synthetic_eval_df)
    env.reset()

    done = False
    steps = 0
    while not done:
        # Trivial policy: always escalate. We only care that the episode
        # terminates cleanly and bookkeeping is internally consistent.
        _, _, done, _ = env.step(action=1)
        steps += 1
        assert steps <= len(synthetic_eval_df) + 1  # safety valve against infinite loops

    summary = env.get_episode_summary()
    assert summary["total_alerts"] == len(synthetic_eval_df)
    assert summary["escalate_count"] == len(synthetic_eval_df)
    assert summary["suppress_count"] == 0
    # Always-escalate should catch 100% of fraud (safe, high-workload baseline).
    assert summary["fraud_catch_rate"] == 1.0


def test_balanced_sampling_targets_50_50_when_max_steps_set(synthetic_eval_df):
    env = AlertSuppressionEnvironment(synthetic_eval_df, max_steps_per_episode=20)
    env.reset()
    sampled = env._episode_df

    assert len(sampled) == 20
    fraud_share = sampled["isFraud"].mean()
    # With only 10 real fraud rows in the fixture, balanced replay sampling
    # should land close to 50%, not the raw ~5% base rate.
    assert fraud_share > 0.3


def test_adaptive_threshold_policy_suppresses_low_risk():
    policy = AdaptiveThresholdPolicy(base_threshold=0.90)
    # prob_fraud=0.01 -> prob_benign=0.99, comfortably above threshold
    low_risk_state = np.array([0.01, 0.0, 0.1, 0.02, 0.5], dtype=np.float32)
    assert policy.decide(low_risk_state) == 0  # SUPPRESS


def test_adaptive_threshold_policy_escalates_high_risk():
    policy = AdaptiveThresholdPolicy(base_threshold=0.90)
    high_risk_state = np.array([0.8, 1.0, 0.9, 0.05, 0.5], dtype=np.float32)
    assert policy.decide(high_risk_state) == 1  # ESCALATE


def test_adaptive_threshold_rises_with_workload():
    policy = AdaptiveThresholdPolicy(base_threshold=0.90)
    low_workload_threshold = policy.adjust_threshold(workload_level=0.0)
    high_workload_threshold = policy.adjust_threshold(workload_level=1.0)
    # Under heavier workload the policy should suppress more, i.e. raise
    # the bar for escalation.
    assert high_workload_threshold >= low_workload_threshold
