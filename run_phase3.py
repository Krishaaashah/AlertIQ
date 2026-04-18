"""
AlertIQ — Phase 3: Suppression Model Training
Trains cost-sensitive logistic regression with Platt calibration → Evaluates thresholds.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
from src.suppression_model import load_and_prepare_data, train_suppression_model
from src.gating_evaluator import evaluate_thresholds
from src.config import TRAIN_DATA_PATH, TEST_DATA_PATH, SUPPRESSION_MODEL_PATH, OUTPUT_DIR, MODEL_DIR


def main():
    print("=" * 60)
    print("  AlertIQ — Phase 3: Suppression Model Training")
    print("=" * 60)

    print(f"\n[1/3] Loading data...")
    train_df, test_df = load_and_prepare_data(TRAIN_DATA_PATH, TEST_DATA_PATH)
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")

    print("\n[2/3] Training cost-sensitive model with Platt calibration...")
    model, eval_df = train_suppression_model(train_df, test_df)

    print(f"\n[3/3] Saving model to {SUPPRESSION_MODEL_PATH}...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, SUPPRESSION_MODEL_PATH)

    # Evaluate thresholds
    results_df = evaluate_thresholds(eval_df, output_dir=OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("  Phase 3 Complete — Model saved, threshold report generated")
    print("=" * 60)


if __name__ == "__main__":
    main()
