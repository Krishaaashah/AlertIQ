# =============================================================================
# AlertIQ — Global Configuration
# =============================================================================
# All tunable parameters for the entire pipeline live here.
# Organized by phase for clarity.
# =============================================================================

import os

RANDOM_SEED = 42

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "rl_outputs")

RAW_DATA_PATH = os.path.join(DATA_DIR, "PaySim.csv")
TRAIN_DATA_PATH = os.path.join(OUTPUT_DIR, "train_feedback.csv")
TEST_DATA_PATH = os.path.join(OUTPUT_DIR, "test_feedback.csv")
SUPPRESSION_MODEL_PATH = os.path.join(MODEL_DIR, "suppression_model.pkl")

# =============================================================================
# PHASE 2 — Alert Rule Thresholds
# =============================================================================
HIGH_VALUE_QUANTILE = 0.95          # R1: flag top 5% by amount
BURST_TXN_THRESHOLD = 3            # R2: flag accounts with >= 3 txns in a step
BALANCE_DROP_QUANTILE = 0.95       # R4: flag top 5% balance drops

# =============================================================================
# PHASE 2 — Analyst Simulation Parameters
# =============================================================================
JUNIOR_ACCURACY = 0.995
SENIOR_ACCURACY = 0.999
JUNIOR_WEIGHT_INITIAL = 0.7
SENIOR_WEIGHT_INITIAL = 1.0

JUNIOR_PROB = 0.7
SENIOR_PROB = 0.3

# Trust Scoring Rewards
# TP: +2, TN: +1, FP: -1, FN: -5 (heavy penalty for missing fraud)
TRUST_REWARD_TP = 2.0
TRUST_REWARD_TN = 1.0
TRUST_PENALTY_FP = -1.0
TRUST_PENALTY_FN = -5.0

# =============================================================================
# PHASE 2 — Train-Test Split
# =============================================================================
TRAIN_SPLIT = 0.7

# =============================================================================
# PHASE 3 — Cost-Sensitive Suppression Model
# =============================================================================
MISSED_FRAUD_COST = 50              # Cost ratio: missing 1 fraud = 50x false alarm
CALIBRATION_METHOD = "sigmoid"      # Platt Scaling for probability calibration
CALIBRATION_CV = 5                  # Cross-validation folds for calibration

# =============================================================================
# PHASE 4 — RL Environment (Reward Structure — Asymmetric)
# =============================================================================
RL_REWARD_SUPPRESS_FRAUD = -500.0   # CATASTROPHIC: suppressed real fraud (must dominate benign suppression gains)
RL_REWARD_SUPPRESS_BENIGN = 1.0     # GOOD: suppressed false positive
RL_REWARD_ESCALATE_FRAUD = 5.0      # GOOD: escalated real fraud (bonus for catching fraud)
RL_REWARD_ESCALATE_BENIGN = -0.5    # ACCEPTABLE: analyst time cost

# =============================================================================
# PHASE 4 — RL Agent Hyperparameters
# =============================================================================
RL_AGENT_TYPE = "dqn"               # 'dqn' or 'bandit'
RL_LEARNING_RATE = 0.001
RL_GAMMA = 0.99                     # Discount factor
RL_EPSILON_START = 1.0              # Initial exploration rate
RL_EPSILON_END = 0.05               # Minimum exploration rate
RL_EPSILON_DECAY = 0.995            # Exploration decay rate
RL_BATCH_SIZE = 64
RL_REPLAY_BUFFER_SIZE = 50000
RL_UPDATE_FREQUENCY = 4             # Target network update freq (episodes)

# =============================================================================
# PHASE 4 — RL Training Configuration
# =============================================================================
RL_TRAINING_EPISODES = 100
RL_EVALUATION_EPISODES = 3          # Full dataset per episode, so keep low
RL_MAX_STEPS_PER_EPISODE = 10000    # Subsample per episode (None = full dataset)
RL_BASELINE_THRESHOLDS = [0.80, 0.85, 0.90, 0.95, 0.99]
RL_WORKLOAD_SCALING = 1.0           # Factor to scale workload impact

# ─── EDA / Evaluation Outputs ──────────────────────────────────────────────
EDA_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "eda")
EVAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "evaluation")

# =============================================================================
# PHASE 5 — Drift Detection & Safety
# =============================================================================
DRIFT_PSI_WARNING = 0.1             # PSI > 0.1 → warning
DRIFT_PSI_CRITICAL = 0.25           # PSI > 0.25 → critical shift
DRIFT_KL_WARNING = 0.05             # KL divergence warning threshold
DRIFT_KL_CRITICAL = 0.15            # KL divergence critical threshold
DRIFT_NUM_BINS = 10                 # Bins for PSI/KL computation
DRIFT_FALLBACK_BIAS = 0.8           # On critical drift, escalate 80% of alerts

# =============================================================================
# PHASE 6 — API Configuration
# =============================================================================
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "AlertIQ — AML Alert Governance API"
API_VERSION = "1.0.0"
