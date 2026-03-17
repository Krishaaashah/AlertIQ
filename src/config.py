# Global configuration and thresholds

RANDOM_SEED = 42

# Alert rule thresholds
HIGH_VALUE_QUANTILE = 0.95          # R1: flag top 5% by amount
BURST_TXN_THRESHOLD = 3           # R2: flag accounts with >= 3 txns in a step (per account)
BALANCE_DROP_QUANTILE = 0.95       # R4: flag top 5% balance drops

# Analyst simulation parameters
JUNIOR_ACCURACY = 0.995
SENIOR_ACCURACY = 0.999
JUNIOR_WEIGHT_INITIAL = 0.7
SENIOR_WEIGHT_INITIAL = 1.0

JUNIOR_PROB = 0.7
SENIOR_PROB = 0.3

# Trust Scoring Parameters
# True Positive: +2, True Negative: +1, False Positive: -1, False Negative: -5 (heavy penalty for missing fraud)
TRUST_REWARD_TP = 2.0
TRUST_REWARD_TN = 1.0
TRUST_PENALTY_FP = -1.0
TRUST_PENALTY_FN = -5.0

# Train-test split
TRAIN_SPLIT = 0.7

# Cost-sensitive model parameters
MISSED_FRAUD_COST = 50      # Cost ratio: missing 1 fraud = 50× cost of 1 false alarm
