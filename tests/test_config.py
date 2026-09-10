"""
Sanity checks on src/config.py.

These aren't testing business logic so much as guarding invariants that the
rest of the system silently assumes — e.g. that missing fraud is always
penalized far more than any other action is rewarded. A well-meaning future
edit to config.py could otherwise quietly break the system's safety
guarantees without any other test noticing.
"""

from src import config


def test_missed_fraud_is_the_dominant_penalty():
    rewards = [
        config.RL_REWARD_SUPPRESS_BENIGN,
        config.RL_REWARD_ESCALATE_FRAUD,
        config.RL_REWARD_ESCALATE_BENIGN,
    ]
    assert config.RL_REWARD_SUPPRESS_FRAUD < 0
    assert abs(config.RL_REWARD_SUPPRESS_FRAUD) > max(rewards) * 10


def test_drift_thresholds_are_ordered():
    assert 0 < config.DRIFT_PSI_WARNING < config.DRIFT_PSI_CRITICAL
    assert 0 < config.DRIFT_KL_WARNING < config.DRIFT_KL_CRITICAL


def test_train_split_is_a_valid_fraction():
    assert 0 < config.TRAIN_SPLIT < 1


def test_cost_sensitive_weight_penalizes_missed_fraud_over_false_alarms():
    assert config.MISSED_FRAUD_COST > 1


def test_epsilon_decays_from_exploration_to_exploitation():
    assert config.RL_EPSILON_START > config.RL_EPSILON_END
    assert 0 < config.RL_EPSILON_END < 1
