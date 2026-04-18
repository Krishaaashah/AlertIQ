"""
AlertIQ — Phase 4: RL Decision Policy Training
Trains DQN and Contextual Bandit agents → Compares vs static baselines → Generates reports.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rl_pipeline import RLDecisionOptimizer
from src.config import OUTPUT_DIR, RL_OUTPUT_DIR, RL_TRAINING_EPISODES


def main():
    print("=" * 60)
    print("  AlertIQ — Phase 4: RL Decision Policy Training")
    print("=" * 60)

    optimizer = RLDecisionOptimizer(
        data_dir=OUTPUT_DIR,
        outputs_dir=RL_OUTPUT_DIR,
    )

    results = optimizer.run_full_pipeline(
        num_episodes_dqn=RL_TRAINING_EPISODES,
        num_episodes_bandit=RL_TRAINING_EPISODES,
        train_dqn=True,
        train_bandit=True,
    )

    print("\n" + "=" * 60)
    print("  Phase 4 Complete")
    print(f"  Results: {RL_OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
