import pandas as pd
from .config import TRAIN_SPLIT

def build_and_split(df: pd.DataFrame, output_path: str):
    
    # Drop intermediate rule columns and ID columns (keep useful engineered features)
    # Keep: rule_count, balance_drain_ratio, alert_reason (used by model)
    drop_cols = ['R1', 'R2', 'R3', 'R4', 'txn_count', 'balance_drop', 'alert_flag', 'nameOrig', 'nameDest']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Time-based split using row order as proxy
    split_index = int(len(df) * TRAIN_SPLIT)
    
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    
    train_df.to_csv(f"{output_path}/train_feedback.csv", index=False)
    test_df.to_csv(f"{output_path}/test_feedback.csv", index=False)
    
    print("Train size:", train_df.shape)
    print("Test size:", test_df.shape)
    
    # Recall check
    baseline_recall = (df[df['isFraud']==1]['analyst_decision'] == 'escalate').mean()
    print("Baseline fraud recall:", round(baseline_recall, 4))
    
    # Fraud distribution check
    train_fraud = train_df['isFraud'].mean()
    test_fraud = test_df['isFraud'].mean()
    print(f"Train fraud rate: {train_fraud:.4f}, Test fraud rate: {test_fraud:.4f}")
    
    return train_df, test_df
