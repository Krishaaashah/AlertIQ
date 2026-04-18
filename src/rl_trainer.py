"""
RL Trainer and Policy Comparison Module.

This module:
1. Trains RL agents (Q-learning or contextual bandit)
2. Compares RL policies against static baseline thresholds
3. Evaluates and logs policy performance
4. Generates evaluation reports
"""

import sys
import numpy as np
import pandas as pd
import pickle
import json
import os
from typing import Dict, Tuple, List, Any
import matplotlib.pyplot as plt
from datetime import datetime

from .rl_environment import AlertSuppressionEnvironment, AdaptiveThresholdPolicy
from .rl_agent import QLearningAgent, ContextualBanditAgent
from .config import RL_MAX_STEPS_PER_EPISODE


class RLTrainer:
    """
    Trainer for RL agents in alert suppression environment.
    """
    
    def __init__(self, eval_df: pd.DataFrame, agent_type: str = 'dqn',
                 device: str = 'cpu', seed: int = 42,
                 max_steps_per_episode: int = RL_MAX_STEPS_PER_EPISODE):
        """
        Initialize trainer.
        
        Args:
            eval_df: Evaluation dataset with columns [prob_fraud, isFraud, rule_count, amount, ...]
            agent_type: 'dqn' (Deep Q-Network) or 'bandit' (contextual bandit)
            device: 'cpu' or 'cuda'
            seed: Random seed
            max_steps_per_episode: Max samples per episode for training efficiency
        """
        np.random.seed(seed)
        
        self.eval_df = eval_df
        self.agent_type = agent_type
        self.device = device
        self.seed = seed
        self.max_steps = max_steps_per_episode
        
        # Create environment and agent
        self.env = AlertSuppressionEnvironment(
            eval_df, max_steps_per_episode=max_steps_per_episode
        )
        
        if agent_type == 'dqn':
            self.agent = QLearningAgent(state_dim=5, action_dim=2, device=device)
        else:  # bandit
            self.agent = ContextualBanditAgent(state_dim=5, action_dim=2)
        
        # Training logs
        self.episode_history = []
        self.batch_losses = []
    
    def train(self, num_episodes: int = 50, batch_size: int = 32,
              update_frequency: int = 4, verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Train the RL agent.
        
        Args:
            num_episodes: Number of training episodes
            batch_size: Batch size for training
            update_frequency: Update target network every N episodes (for DQN)
            verbose: Print progress
            
        Returns:
            Episode history with performance metrics
        """
        steps_per_ep = self.max_steps or len(self.eval_df)
        print(f"\n{'='*60}")
        print(f"Training {self.agent_type.upper()} Agent")
        print(f"  Episodes: {num_episodes} | Steps/episode: {steps_per_ep:,}")
        print(f"{'='*60}")
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0.0
            episode_losses = []
            
            # Collect experiences for one episode
            for step in range(steps_per_ep):
                if isinstance(self.agent, QLearningAgent):
                    # DQN agent
                    action = self.agent.select_action(state, training=True)
                    next_state, reward, done, info = self.env.step(action)
                    self.agent.remember(state, action, reward, next_state, done)
                    
                    # Training step
                    loss = self.agent.train_step(batch_size)
                    episode_losses.append(loss)
                    
                    episode_reward += reward
                    state = next_state
                    
                    if done:
                        break
                else:
                    # Contextual bandit agent
                    action = self.agent.select_action(state, training=True)
                    next_state, reward, done, info = self.env.step(action)
                    episode_reward += reward
                    state = next_state
                    
                    if done:
                        break
            
            # Post-episode training (for bandit agents)
            if isinstance(self.agent, ContextualBanditAgent) and self.env.episode_outcomes:
                batch = [
                    (o['state'] if 'state' in o else np.zeros(5), 
                     o['action'], o['reward'], np.zeros(5), True)
                    for o in self.env.episode_outcomes[-32:]
                ]
                loss = self.agent.train_step(batch)
                episode_losses.append(loss)
            
            # Periodic updates
            if isinstance(self.agent, QLearningAgent) and (episode + 1) % update_frequency == 0:
                self.agent.update_target_network()
            
            if isinstance(self.agent, QLearningAgent):
                self.agent.decay_epsilon()
            
            # Log episode
            summary = self.env.get_episode_summary()
            summary['episode'] = episode
            summary['avg_loss'] = np.mean(episode_losses) if episode_losses else 0.0
            summary['total_reward'] = episode_reward
            self.episode_history.append(summary)
            
            if verbose and (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1:3d}: "
                      f"Reward={episode_reward:7.2f}, "
                      f"Fraud Catch={summary['fraud_catch_rate']:.3f}, "
                      f"FP Reduction={summary['fp_reduction_rate']:.3f}, "
                      f"Workload Saved={summary['workload_savings']:.3f}")
        
        print(f"{'='*60}")
        print("Training Complete!")
        return self.episode_history
    
    def evaluate_policy(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate trained policy on clean episodes (no training).
        Uses full dataset (no subsampling) for fair comparison with static baselines.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation metrics
        """
        print(f"\nEvaluating {self.agent_type.upper()} Agent Policy (full dataset)...")
        sys.stdout.flush()
        
        # Use full dataset for evaluation (no subsampling)
        eval_env = AlertSuppressionEnvironment(self.eval_df, max_steps_per_episode=None)
        eval_episode_history = []
        
        for episode in range(num_episodes):
            state = eval_env.reset()
            
            while True:
                action = self.agent.select_action(state, training=False)
                next_state, reward, done, info = eval_env.step(action)
                state = next_state
                if done:
                    break
            
            summary = eval_env.get_episode_summary()
            eval_episode_history.append(summary)
        
        # Aggregate metrics
        metrics = {
            'agent_type': self.agent_type,
            'avg_fraud_catch_rate': np.mean([e['fraud_catch_rate'] for e in eval_episode_history]),
            'avg_fp_reduction_rate': np.mean([e['fp_reduction_rate'] for e in eval_episode_history]),
            'avg_workload_savings': np.mean([e['workload_savings'] for e in eval_episode_history]),
            'avg_total_reward': np.mean([e['total_reward'] for e in eval_episode_history]),
            'std_reward': np.std([e['total_reward'] for e in eval_episode_history]),
        }
        
        print(f"  Fraud Catch Rate: {metrics['avg_fraud_catch_rate']:.4f}")
        print(f"  FP Reduction: {metrics['avg_fp_reduction_rate']:.4f}")
        print(f"  Workload Savings: {metrics['avg_workload_savings']:.4f}")
        print(f"  Avg Reward: {metrics['avg_total_reward']:.2f} (±{metrics['std_reward']:.2f})")
        sys.stdout.flush()
        
        return metrics


class PolicyComparator:
    """
    Compare RL policy against static baseline policies.
    """
    
    def __init__(self, eval_df: pd.DataFrame):
        """
        Initialize comparator.
        
        Args:
            eval_df: Evaluation dataset
        """
        self.eval_df = eval_df
        self.results = {}
    
    def evaluate_static_policy(self, policy_name: str, threshold: float) -> Dict[str, Any]:
        """
        Evaluate static threshold policy.
        
        Args:
            policy_name: Name of policy (e.g., "Threshold_0.90")
            threshold: Suppression threshold
            
        Returns:
            Evaluation metrics
        """
        print(f"\nEvaluating {policy_name}...")
        
        env = AlertSuppressionEnvironment(self.eval_df)
        policy = AdaptiveThresholdPolicy(base_threshold=threshold)
        
        state = env.reset()
        while True:
            action = policy.decide(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            if done:
                break
        
        summary = env.get_episode_summary()
        summary['policy_name'] = policy_name
        summary['threshold'] = threshold
        
        print(f"  Fraud Catch Rate: {summary['fraud_catch_rate']:.4f}")
        print(f"  FP Reduction: {summary['fp_reduction_rate']:.4f}")
        print(f"  Workload Savings: {summary['workload_savings']:.4f}")
        
        return summary
    
    def compare_policies(self, rl_metrics: Dict[str, Any],
                        baseline_thresholds: List[float] = None) -> pd.DataFrame:
        """
        Compare RL agent against multiple static policies.
        
        Args:
            rl_metrics: Metrics from RL agent
            baseline_thresholds: Static thresholds to evaluate
            
        Returns:
            Comparison DataFrame
        """
        if baseline_thresholds is None:
            baseline_thresholds = [0.80, 0.85, 0.90, 0.95, 0.99]
        
        print("\n" + "="*70)
        print("POLICY COMPARISON: RL vs Static Thresholds")
        print("="*70)
        
        comparison_results = []
        
        # RL agent result
        comparison_results.append({
            'Policy': 'RL Agent',
            'Parameter': rl_metrics['agent_type'],
            'Fraud Catch Rate': rl_metrics['avg_fraud_catch_rate'],
            'FP Reduction Rate': rl_metrics['avg_fp_reduction_rate'],
            'Workload Savings': rl_metrics['avg_workload_savings'],
            'Avg Reward': rl_metrics['avg_total_reward'],
        })
        
        # Static policies
        for threshold in baseline_thresholds:
            policy_name = f"Static_{threshold:.2f}"
            baseline_metrics = self.evaluate_static_policy(policy_name, threshold)
            
            comparison_results.append({
                'Policy': f'Static Threshold',
                'Parameter': f'{threshold:.2f}',
                'Fraud Catch Rate': baseline_metrics['fraud_catch_rate'],
                'FP Reduction Rate': baseline_metrics['fp_reduction_rate'],
                'Workload Savings': baseline_metrics['workload_savings'],
                'Avg Reward': baseline_metrics['total_reward'],
            })
        
        comparison_df = pd.DataFrame(comparison_results)
        
        print("\n" + "="*70)
        print("FINAL COMPARISON:")
        print("="*70)
        print(comparison_df.to_string(index=False, float_format="%.4f"))
        print("="*70)
        
        return comparison_df


def save_policy(agent: Any, params: Dict, filepath: str):
    """Save trained policy to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    state = {
        'agent_type': type(agent).__name__,
        'params': params,
        'timestamp': datetime.now().isoformat(),
    }
    
    if hasattr(agent, 'q_network'):
        # DQN agent - save PyTorch model
        state['model_state'] = agent.q_network.state_dict()
    elif hasattr(agent, 'weights'):
        # Bandit agent - save numpy weights
        state['weights'] = agent.weights
    
    with open(filepath, 'wb') as f:
        pickle.dump(state, f)
    
    print(f"Policy saved to {filepath}")


def load_policy(filepath: str, agent_type: str = 'dqn', device: str = 'cpu') -> Any:
    """Load trained policy from disk."""
    with open(filepath, 'rb') as f:
        state = pickle.load(f)
    
    # Reconstruct agent
    if agent_type == 'dqn':
        agent = QLearningAgent(device=device)
        if 'model_state' in state:
            agent.q_network.load_state_dict(state['model_state'])
    else:
        agent = ContextualBanditAgent()
        if 'weights' in state:
            agent.weights = state['weights']
    
    print(f"Policy loaded from {filepath}")
    return agent
