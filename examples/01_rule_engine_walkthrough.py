"""
Example 1 — Rule-based alert generation (Phase 1)

Loads the small synthetic sample shipped in dataset/sample_paysim.csv,
runs it through the exact same feature loading and rule-engine code used
by the full pipeline, and prints a breakdown of what got flagged and why.

Run:
    python examples/01_rule_engine_walkthrough.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_and_clean
from src.alert_rules import generate_alerts

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "dataset", "sample_paysim.csv")


def main():
    if not os.path.exists(DATA_PATH):
        print("Sample dataset not found. Generating it now...")
        os.system(f"{sys.executable} dataset/generate_sample.py")

    print(f"Loading transactions from {DATA_PATH}")
    df = load_and_clean(DATA_PATH)
    print(f"  {len(df):,} transactions loaded\n")

    alerts = generate_alerts(df)

    print("\n--- Summary ---")
    print(f"Total transactions:   {len(df):,}")
    print(f"Alerts generated:     {len(alerts):,} ({len(alerts) / len(df):.1%} of traffic)")
    print(f"Fraud in raw data:    {int(df['isFraud'].sum())}")
    print(f"Fraud caught by rules:{int(alerts['isFraud'].sum())}")

    if df["isFraud"].sum() > 0:
        recall = alerts["isFraud"].sum() / df["isFraud"].sum()
        print(f"Rule-layer fraud recall: {recall:.2%}")

    print("\nTop 5 highest-amount alerts:")
    cols = ["amount", "rule_count", "alert_reason", "isFraud"]
    print(alerts.sort_values("amount", ascending=False)[cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
