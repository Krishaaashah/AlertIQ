"""
Generate a small, synthetic PaySim-shaped sample dataset.

The full PaySim dataset (~6.36M rows) is too large to ship inside a Git
repository and is distributed by its authors on Kaggle. This script creates
a small, entirely synthetic dataset that mirrors the *schema* and rough
*statistical shape* of PaySim (transaction types, amount distribution,
~0.1-0.4% fraud rate, fraud concentrated in CASH_OUT/TRANSFER) so that:

  - `tests/` can run without downloading anything
  - `examples/` can demonstrate the full pipeline end-to-end in seconds
  - New contributors can explore the code before committing to the full
    Kaggle download

This is NOT real financial data and NOT a substitute for the full dataset
when training or evaluating the actual models reported in the README.

Usage:
    python dataset/generate_sample.py
"""

import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_ROWS = 3000
FRAUD_RATE = 0.0025  # slightly enriched vs. real PaySim (0.0013) so a 3k-row
                      # sample still contains enough fraud cases to be useful

TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
TYPE_PROBS = [0.22, 0.35, 0.03, 0.34, 0.06]
FRAUD_ELIGIBLE_TYPES = {"CASH_OUT", "TRANSFER"}  # matches real PaySim behavior


def generate_sample(n_rows: int = N_ROWS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    step = rng.integers(1, 744, size=n_rows)  # PaySim spans 744 hourly steps
    txn_type = rng.choice(TYPES, size=n_rows, p=TYPE_PROBS)
    amount = rng.lognormal(mean=9.0, sigma=1.8, size=n_rows).round(2)

    name_orig = np.array([f"C{rng.integers(10**8, 10**9)}" for _ in range(n_rows)])
    name_dest = np.array([f"C{rng.integers(10**8, 10**9)}" for _ in range(n_rows)])

    old_balance_org = rng.lognormal(mean=9.5, sigma=1.5, size=n_rows).round(2)
    # Outgoing transaction types reduce sender balance; PAYMENT/CASH_IN differ slightly
    new_balance_orig = np.clip(old_balance_org - amount, 0, None).round(2)

    old_balance_dest = rng.lognormal(mean=8.5, sigma=1.8, size=n_rows).round(2)
    new_balance_dest = (old_balance_dest + amount * rng.uniform(0.7, 1.0, n_rows)).round(2)

    is_fraud = np.zeros(n_rows, dtype=int)
    fraud_eligible_idx = np.where(np.isin(txn_type, list(FRAUD_ELIGIBLE_TYPES)))[0]
    n_fraud = max(1, int(n_rows * FRAUD_RATE))
    fraud_idx = rng.choice(fraud_eligible_idx, size=min(n_fraud, len(fraud_eligible_idx)), replace=False)
    is_fraud[fraud_idx] = 1

    # Fraudulent transactions in PaySim characteristically drain the sender's
    # balance almost completely, and skew toward larger amounts.
    amount[fraud_idx] = rng.lognormal(mean=11.5, sigma=1.0, size=len(fraud_idx)).round(2)
    old_balance_org[fraud_idx] = np.maximum(amount[fraud_idx], old_balance_org[fraud_idx])
    new_balance_orig[fraud_idx] = (old_balance_org[fraud_idx] - amount[fraud_idx]).round(2)
    new_balance_orig[fraud_idx] = np.clip(new_balance_orig[fraud_idx], 0, None)

    is_flagged_fraud = np.zeros(n_rows, dtype=int)
    very_large = amount > 200_000
    is_flagged_fraud[very_large & (is_fraud == 1)] = rng.choice(
        [0, 1], size=int((very_large & (is_fraud == 1)).sum()), p=[0.9, 0.1]
    ) if (very_large & (is_fraud == 1)).sum() > 0 else is_flagged_fraud[very_large & (is_fraud == 1)]

    df = pd.DataFrame({
        "step": step,
        "type": txn_type,
        "amount": amount,
        "nameOrig": name_orig,
        "oldbalanceOrg": old_balance_org,
        "newbalanceOrig": new_balance_orig,
        "nameDest": name_dest,
        "oldbalanceDest": old_balance_dest,
        "newbalanceDest": new_balance_dest,
        "isFraud": is_fraud,
        "isFlaggedFraud": is_flagged_fraud,
    })

    return df


if __name__ == "__main__":
    df = generate_sample()
    out_path = os.path.join(os.path.dirname(__file__), "sample_paysim.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} synthetic rows to {out_path}")
    print(f"Synthetic fraud rate: {df['isFraud'].mean():.4%}")
    print(df['type'].value_counts())
