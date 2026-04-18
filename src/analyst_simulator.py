import numpy as np
import pandas as pd
from .config import (
    JUNIOR_ACCURACY, SENIOR_ACCURACY,
    JUNIOR_WEIGHT_INITIAL, SENIOR_WEIGHT_INITIAL,
    JUNIOR_PROB, SENIOR_PROB,
    TRUST_REWARD_TP, TRUST_REWARD_TN,
    TRUST_PENALTY_FP, TRUST_PENALTY_FN,
    RANDOM_SEED
)

class AnalystSimulator:
    """Simulates a team of analysts handling alerts, tracking their historical Trust Scores."""
    
    def __init__(self, num_analysts=50):
        np.random.seed(RANDOM_SEED)
        
        # Initialize analyst pool
        self.analysts = {}
        for i in range(num_analysts):
            atype = np.random.choice(['junior', 'senior'], p=[JUNIOR_PROB, SENIOR_PROB])
            self.analysts[f"A_{i}"] = {
                'type': atype,
                'accuracy': JUNIOR_ACCURACY if atype == 'junior' else SENIOR_ACCURACY,
                'trust_score': JUNIOR_WEIGHT_INITIAL if atype == 'junior' else SENIOR_WEIGHT_INITIAL,
                'total_reviews': 0
            }
            
    def simulate_feedback(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        analyst_ids = list(self.analysts.keys())
        
        # Assign random analyst to each alert
        df['analyst_id'] = np.random.choice(analyst_ids, size=len(df))
        
        decisions = []
        trust_weights = []
        
        # We simulate chronologically (assuming df is sorted by time step)
        for _, row in df.iterrows():
            aid = row['analyst_id']
            fraud = row['isFraud']
            
            analyst = self.analysts[aid]
            
            # Record current trust weight (before taking action)
            trust_weights.append(analyst['trust_score'])
            
            # Make decision
            correct = np.random.rand() < analyst['accuracy']
            
            if fraud == 1:
                decision = 'escalate' if correct else 'dismiss'
                # Update trust based on outcome
                if decision == 'escalate':
                    analyst['trust_score'] += TRUST_REWARD_TP  # True Positive (Caught Fraud)
                else:
                    analyst['trust_score'] += TRUST_PENALTY_FN # False Negative (Missed Fraud!)
            else:
                decision = 'dismiss' if correct else 'escalate'
                if decision == 'dismiss':
                    analyst['trust_score'] += TRUST_REWARD_TN  # True Negative (Correctly dismissed benign)
                else:
                    analyst['trust_score'] += TRUST_PENALTY_FP # False Positive (Wasted time escalating benign)
            
            # Normalize trust score to prevent negative weights or exploding values
            # We decay it slightly to make recent behavior matter more, and bound it
            analyst['trust_score'] = max(0.1, min(10.0, analyst['trust_score'] * 0.9999))
            analyst['total_reviews'] += 1
            
            decisions.append(decision)
            
        df['analyst_decision'] = decisions
        df['analyst_weight'] = trust_weights
        
        # Attach static type for reference
        df['analyst_type'] = df['analyst_id'].map({k: v['type'] for k, v in self.analysts.items()})
        
        return df

def simulate_analysts(df: pd.DataFrame) -> pd.DataFrame:
    """Wrapper function to maintain compatibility with phase2.py"""
    simulator = AnalystSimulator(num_analysts=50) # Team of 50 analysts
    return simulator.simulate_feedback(df)
