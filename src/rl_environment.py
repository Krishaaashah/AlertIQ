"""
RL Environment for Alert Suppression Optimization.

This module defines a multi-armed contextual bandit environment where an RL agent
learns to make suppression decisions (Suppress vs. Escalate) based on:
  - Suppression confidence (prob_fraud from ML model)
  - Alert context (transaction features, rule triggers)
  - System state (fraud rate, workload level)

State Space:
  - prob_fraud: Model's predicted probability of fraud [0, 1]
  - rule_count: Number of alert rules triggered [0, 4]
  - amount_normalized: Transaction amount normalized [0, 1]
  - fraud_rate_context: Current system fraud rate [0, 1]
  - workload_level: Current System workload [0, 1]

Action Space:
  - 0: SUPPRESS (prevent analyst review, assume safe)
  - 1: ESCALATE (send to analyst for review)

Reward Function (Asymmetric):
  - Suppress fraud: -50 (catastrophic penalty)
  - Suppress benign: +1 (desired outcome)
  - Escalate fraud: +1 (correctly routed)
  - Escalate benign: -0.1 (small cost - analyst time)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

from .config import (
    RL_REWARD_SUPPRESS_FRAUD, RL_REWARD_SUPPRESS_BENIGN,
    RL_REWARD_ESCALATE_FRAUD, RL_REWARD_ESCALATE_BENIGN
)


class AlertSuppressionEnvironment:
    """
    RL Environment for alert suppression decisions.
    
    Provides state observations and computes rewards based on decisions and outcomes.
    """
    
    def __init__(self, eval_df: pd.DataFrame, fraud_rate: float = None, 
                 workload_scaling: float = 1.0, max_steps_per_episode: int = None):
        """
        Initialize the environment.
        
        Args:
            eval_df: DataFrame with columns [prob_fraud, isFraud, rule_count, amount, ...]
            fraud_rate: Current system fraud rate (auto-calculated if None)
            workload_scaling: Factor to scale workload impact (default 1.0)
            max_steps_per_episode: Max alerts to process per episode (None = all)
        """
        self.eval_df = eval_df.copy()
        self.fraud_rate = fraud_rate or eval_df['isFraud'].mean()
        self.workload_scaling = workload_scaling
        self.max_steps_per_episode = max_steps_per_episode
        
        # Normalize numerical features for state representation
        self._normalize_features()
        
        # Track episode statistics
        self.current_idx = 0
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_outcomes = []
        
    def _normalize_features(self):
        """Normalize features to [0, 1] range for state representation."""
        # Amount normalization (log scale due to wide range)
        if 'amount' in self.eval_df.columns:
            self.eval_df['amount_log'] = np.log1p(self.eval_df['amount'])
            max_log_amount = self.eval_df['amount_log'].max()
            self.eval_df['amount_normalized'] = (
                self.eval_df['amount_log'] / max_log_amount
                if max_log_amount > 0 else 0
            )
        else:
            self.eval_df['amount_normalized'] = 0.5
            
        # Ensure prob_fraud is normalized
        if 'prob_fraud' not in self.eval_df.columns and 'prob_benign' in self.eval_df.columns:
            self.eval_df['prob_fraud'] = 1.0 - self.eval_df['prob_benign']
            
        # Clamp probabilities
        self.eval_df['prob_fraud'] = self.eval_df['prob_fraud'].clip(0, 1)
        
        # Rule count normalization
        if 'rule_count' in self.eval_df.columns:
            max_rules = self.eval_df['rule_count'].max()
            self.eval_df['rule_count_normalized'] = (
                self.eval_df['rule_count'] / max_rules if max_rules > 0 else 0
            )
        else:
            self.eval_df['rule_count_normalized'] = 0.5
    
    def reset(self) -> np.ndarray:
        """Reset environment to start of episode with optional subsampling."""
        self.current_idx = 0
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_outcomes = []
        
        # Subsample and shuffle for each episode if max_steps is set
        if self.max_steps_per_episode and self.max_steps_per_episode < len(self.eval_df):
            # BALANCED SAMPLING FOR RL TRAINING (50% Fraud / 50% Benign)
            fraud_pool = self.eval_df[self.eval_df['isFraud'] == 1]
            benign_pool = self.eval_df[self.eval_df['isFraud'] == 0]
            
            target_fraud = min(len(fraud_pool), self.max_steps_per_episode // 2)
            target_benign = self.max_steps_per_episode - target_fraud
            
            # Use replace=True for fraud if we don't have enough absolute fraud cases
            sampled_fraud = fraud_pool.sample(n=target_fraud, replace=True, random_state=None)
            sampled_benign = benign_pool.sample(n=target_benign, random_state=None)
            
            self._episode_df = pd.concat([sampled_fraud, sampled_benign]).sample(
                frac=1, random_state=None
            ).reset_index(drop=True)
        else:
            self._episode_df = self.eval_df
        
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """
        Get current state representation.
        
        Returns:
            State vector [prob_fraud, rule_count_norm, amount_norm, fraud_rate, workload]
        """
        df = self._episode_df if hasattr(self, '_episode_df') else self.eval_df
        
        if self.current_idx >= len(df):
            # Return terminal state
            return np.zeros(5)
        
        row = df.iloc[self.current_idx]
        
        # Calculate workload: % of total alerts to be processed (higher = more workload)
        remaining_alerts = len(df) - self.current_idx
        workload_level = remaining_alerts / len(df)
        workload_level = workload_level * self.workload_scaling  # Apply scaling
        workload_level = np.clip(workload_level, 0, 1)
        
        state = np.array([
            row['prob_fraud'],                      # Fraud probability
            row['rule_count_normalized'],           # Rule count (0-1)
            row['amount_normalized'],               # Amount (normalized log scale)
            self.fraud_rate,                        # System fraud rate
            workload_level,                         # Workload pressure
        ], dtype=np.float32)
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute action and receive reward.
        
        Args:
            action: 0 = SUPPRESS, 1 = ESCALATE
            
        Returns:
            state, reward, done, info
        """
        df = self._episode_df if hasattr(self, '_episode_df') else self.eval_df
        
        if self.current_idx >= len(df):
            return np.zeros(5), 0.0, True, {}
        
        row = df.iloc[self.current_idx]
        is_fraud = bool(row['isFraud'])
        current_state = self.get_state()
        
        # Compute reward based on action and ground truth
        reward = self._compute_reward(action, is_fraud)
        
        # Track episode information
        self.episode_rewards.append(reward)
        self.episode_actions.append(action)
        self.episode_outcomes.append({
            'action': action,
            'is_fraud': is_fraud,
            'reward': reward,
            'prob_fraud': row['prob_fraud'],
            'idx': self.current_idx
        })
        
        # Move to next state
        self.current_idx += 1
        done = self.current_idx >= len(df)
        next_state = self.get_state()
        
        info = {
            'is_fraud': is_fraud,
            'action_name': 'SUPPRESS' if action == 0 else 'ESCALATE',
            'prob_fraud': row['prob_fraud'],
            'rule_count': row['rule_count'] if 'rule_count' in row else 0
        }
        
        return next_state, reward, done, info
    
    def _compute_reward(self, action: int, is_fraud: bool) -> float:
        """
        Compute asymmetric reward.
        
        Reward Structure (Asymmetric):
          - Suppress (0) + Fraud (1): -50  (CATASTROPHIC - missed fraud)
          - Suppress (0) + Benign (0): +1  (CORRECT - saved analyst time)
          - Escalate (1) + Fraud (1): +1   (CORRECT - routed fraud)
          - Escalate (1) + Benign (0): -0.1 (ACCEPTABLE - analyst time cost)
        """
        if action == 0:  # SUPPRESS
            if is_fraud:
                return RL_REWARD_SUPPRESS_FRAUD   # Catastrophic penalty
            else:
                return RL_REWARD_SUPPRESS_BENIGN  # Reward
        else:  # ESCALATE (action == 1)
            if is_fraud:
                return RL_REWARD_ESCALATE_FRAUD   # Reward
            else:
                return RL_REWARD_ESCALATE_BENIGN  # Small penalty
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary statistics of completed episode."""
        if not self.episode_outcomes:
            return {}
        
        outcomes = pd.DataFrame(self.episode_outcomes)
        
        suppress_count = (outcomes['action'] == 0).sum()
        escalate_count = (outcomes['action'] == 1).sum()
        
        fraud_cases = outcomes[outcomes['is_fraud'] == True]
        benign_cases = outcomes[outcomes['is_fraud'] == False]
        
        # Fraud handling
        fraud_suppressed = len(fraud_cases[fraud_cases['action'] == 0])
        fraud_escalated = len(fraud_cases[fraud_cases['action'] == 1])
        fraud_caught_rate = fraud_escalated / len(fraud_cases) if len(fraud_cases) > 0 else 0
        
        # Benign handling
        benign_suppressed = len(benign_cases[benign_cases['action'] == 0])
        benign_escalated = len(benign_cases[benign_cases['action'] == 1])
        fp_reduction_rate = benign_suppressed / len(benign_cases) if len(benign_cases) > 0 else 0
        
        return {
            'total_alerts': len(outcomes),
            'suppress_count': suppress_count,
            'escalate_count': escalate_count,
            'total_reward': sum(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards),
            'fraud_caught': fraud_escalated,
            'fraud_missed': fraud_suppressed,
            'fraud_catch_rate': fraud_caught_rate,
            'fp_suppressed': benign_suppressed,
            'fp_escalated': benign_escalated,
            'fp_reduction_rate': fp_reduction_rate,
            'workload_savings': suppress_count / len(outcomes) if len(outcomes) > 0 else 0,
        }


class AdaptiveThresholdPolicy:
    """
    Baseline policy: static threshold-based suppression (for comparison).
    
    This represents the current system and serves as a baseline for RL comparison.
    """
    
    def __init__(self, base_threshold: float = 0.90):
        """
        Initialize static threshold policy.
        
        Args:
            base_threshold: Base suppression threshold for prob_benign
        """
        self.base_threshold = base_threshold
        self.current_workload_level = 0.0
        
    def adjust_threshold(self, workload_level: float) -> float:
        """
        Adaptively adjust threshold based on workload.
        
        When workload is high, increase threshold to suppress more alerts.
        When workload is low, decrease threshold for more thorough review.
        """
        # Workload-based adjustment (max ±0.05)
        adjustment = workload_level * 0.05
        adjusted_threshold = self.base_threshold + adjustment
        adjusted_threshold = np.clip(adjusted_threshold, 0.0, 1.0)
        self.current_workload_level = workload_level
        return adjusted_threshold
    
    def decide(self, state: np.ndarray) -> int:
        """
        Make suppression decision based on state.
        
        Args:
            state: [prob_fraud, rule_count_norm, amount_norm, fraud_rate, workload_level]
            
        Returns:
            0 = SUPPRESS, 1 = ESCALATE
        """
        prob_fraud = state[0]
        workload_level = state[4]
        
        # Adjust threshold dynamically
        threshold = self.adjust_threshold(workload_level)
        
        # Suppress if fraud probability is below threshold
        prob_benign = 1.0 - prob_fraud
        if prob_benign >= threshold:
            return 0  # SUPPRESS
        else:
            return 1  # ESCALATE
