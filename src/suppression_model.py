import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

try:
    from .config import MISSED_FRAUD_COST
except ImportError:
    MISSED_FRAUD_COST = 50

def load_and_prepare_data(train_path: str, test_path: str):
    """Load the dataset previously split by dataset_builder.py."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def train_suppression_model(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Train a cost-sensitive logistic regression model to predict false positive alerts.
    Target: isFraud (1 = Fraud, 0 = False Positive)
    """
    print("\n--- Feature Engineering & Modeling ---")
    
    # 1. Remove leakage columns
    leakage_cols = ['analyst_decision', 'analyst_id', 'analyst_weight', 'analyst_type', 'isFraud']
    
    y_train = train_df['isFraud']
    y_test = test_df['isFraud']
    
    X_train = train_df.drop(columns=[c for c in leakage_cols if c in train_df.columns])
    X_test = test_df.drop(columns=[c for c in leakage_cols if c in test_df.columns])
    
    # 2. Select relevant transaction features and categorical data
    categorical_cols = ['alert_reason'] if 'alert_reason' in X_train.columns else []
    
    # Capture all numerical features (exclude categoricals)
    numerical_cols = [c for c in X_train.columns if c not in categorical_cols and pd.api.types.is_numeric_dtype(X_train[c])]
    
    print(f"Applying One-Hot Encoding to: {categorical_cols}")
    print(f"Applying Normalization to {len(numerical_cols)} numerical features.")
    
    # 3. Pipeline creation: Normalize numericals, Encode categoricals
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    # 4. Cost-Sensitive Modeling
    # Heavy penalty for false negatives (misclassifying 1 as 0, thereby suppressing true fraud)
    class_weight = {0: 1, 1: MISSED_FRAUD_COST}
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight=class_weight, random_state=42, max_iter=1000))
    ])
    
    print(f"Training Logistic Regression with class weights: {class_weight}")
    model.fit(X_train, y_train)
    
    # 5. Confidence Score Generation
    print("Generating calibrated probability scores...")
    # predict_proba returns [prob_benign, prob_fraud]
    test_probs = model.predict_proba(X_test)
    
    eval_df = test_df.copy()
    eval_df['prob_benign'] = test_probs[:, 0]
    eval_df['prob_fraud'] = test_probs[:, 1]
    
    return model, eval_df

if __name__ == "__main__":
    train_path = "data/train_feedback.csv"
    test_path = "data/test_feedback.csv"
    import os
    if os.path.exists(train_path) and os.path.exists(test_path):
        train_df, test_df = load_and_prepare_data(train_path, test_path)
        model, eval_df = train_suppression_model(train_df, test_df)
    else:
        print("Data files not found. Ensure dataset builder has produced train_feedback.csv and test_feedback.csv")
