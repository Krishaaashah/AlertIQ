import pandas as pd
import numpy as np
from .config import HIGH_VALUE_QUANTILE, BALANCE_DROP_QUANTILE, BURST_TXN_THRESHOLD

def generate_alerts(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    
    # Rule R1: High-value transaction
    high_value_threshold = df['amount'].quantile(HIGH_VALUE_QUANTILE)
    df['R1'] = (df['amount'] > high_value_threshold).astype(int)
    
    # Group by nameOrig AND step to count transactions per account per hour
    df['txn_count'] = df.groupby(['nameOrig', 'step'])['amount'].transform('count')
    df['R2'] = (df['txn_count'] >= BURST_TXN_THRESHOLD).astype(int)
    
    # Rule R3: Fraud-prone transaction types (CASH_OUT and TRANSFER)
    # These are the ONLY types where fraud occurs in PaySim
    type_cols = [col for col in df.columns if col.startswith('type_')]
    
    # Identify CASH_OUT and TRANSFER columns
    r3_cols = [c for c in type_cols if 'CASH_OUT' in c or 'TRANSFER' in c]
    if r3_cols:
        # R3: Flag high-risk types ONLY IF amount > 200,000
        df['R3'] = ((df[r3_cols].sum(axis=1) > 0) & (df['amount'] > 200_000)).astype(int)
    else:
        # Fallback: if drop_first removed CASH_OUT (alphabetically after CASH_IN),
        # CASH_OUT should still exist. But handle edge case.
        df['R3'] = 0
    
    # Rule R4: Sudden balance drain — sender loses most/all of their balance
    # Only flag when there IS a meaningful balance to drain
    df['balance_drop'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balance_drain_ratio'] = np.where(
        df['oldbalanceOrg'] > 0,
        df['balance_drop'] / df['oldbalanceOrg'],
        0
    )
    # Flag where >= 90% of balance was drained AND amount > 10,000
    df['R4'] = ((df['balance_drain_ratio'] >= 0.9) & (df['amount'] > 10_000)).astype(int)
    
    # Count how many rules triggered per transaction
    df['rule_count'] = df[['R1','R2','R3','R4']].sum(axis=1)
    
    # Final alert flag: at least 1 rule triggered
    df['alert_flag'] = (df['rule_count'] > 0).astype(int)
    
    # Store alert reason (first triggered rule for primary reason)
    rule_cols = ['R1','R2','R3','R4']
    df['alert_reason'] = df[rule_cols].apply(
        lambda row: rule_cols[row.values.argmax()] if row.sum() > 0 else 'none',
        axis=1
    )
    
    # Keep only alerts
    df = df[df['alert_flag'] == 1].reset_index(drop=True)
    
    print(f"  Alert rate: {len(df)} alerts from original data")
    print(f"  R1 (high-value): {df['R1'].sum():,}")
    print(f"  R2 (burst txns): {df['R2'].sum():,}")
    print(f"  R3 (risky type): {df['R3'].sum():,}")
    print(f"  R4 (balance drain): {df['R4'].sum():,}")
    
    return df
