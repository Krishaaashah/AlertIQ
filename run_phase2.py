"""
AlertIQ — Phase 2: Data Pipeline
Loads PaySim data → Generates rule-based alerts → Simulates analyst feedback → Splits train/test.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_and_clean
from src.alert_rules import generate_alerts
from src.analyst_simulator import simulate_analysts
from src.dataset_builder import build_and_split
from src.config import RAW_DATA_PATH, OUTPUT_DIR


def main():
    print("=" * 60)
    print("  AlertIQ — Phase 2: Data & Alert Simulation Pipeline")
    print("=" * 60)

    print("\n[1/4] Loading and cleaning PaySim data...")
    df = load_and_clean(RAW_DATA_PATH)
    print(f"  Loaded {len(df):,} transactions")

    print("\n[2/4] Generating rule-based alerts...")
    df_alerts = generate_alerts(df)
    print(f"  Generated {len(df_alerts):,} alerts")

    print("\n[3/4] Simulating analyst feedback...")
    df_feedback = simulate_analysts(df_alerts)

    print("\n[4/4] Building train/test datasets...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    build_and_split(df_feedback, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("  Phase 2 Complete — Files saved to outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
