"""
Main RL Pipeline Execution Script.

This module orchestrates:
1. Loading evaluation data
2. Training RL agents (DQN and Contextual Bandit)
3. Comparing RL vs static policies
4. Generating performance visualizations
5. Saving results and trained models
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Any

from .config import (
    OUTPUT_DIR, RL_OUTPUT_DIR, TRAIN_DATA_PATH, TEST_DATA_PATH,
    RL_TRAINING_EPISODES, RL_EVALUATION_EPISODES, RL_BASELINE_THRESHOLDS,
    RL_BATCH_SIZE, RANDOM_SEED, RL_MAX_STEPS_PER_EPISODE
)
from .suppression_model import train_suppression_model, load_and_prepare_data
from .rl_environment import AlertSuppressionEnvironment, AdaptiveThresholdPolicy
from .rl_agent import QLearningAgent, ContextualBanditAgent
from .rl_trainer import RLTrainer, PolicyComparator, save_policy, load_policy


class RLDecisionOptimizer:
    """
    End-to-end RL decision optimization pipeline.
    """
    
    def __init__(self, data_dir: str = None, outputs_dir: str = None,
                 seed: int = RANDOM_SEED):
        """
        Initialize optimizer.
        
        Args:
            data_dir: Directory containing train_feedback.csv and test_feedback.csv
            outputs_dir: Output directory for results and models
            seed: Random seed
        """
        self.data_dir = data_dir or OUTPUT_DIR
        self.outputs_dir = outputs_dir or RL_OUTPUT_DIR
        self.seed = seed
        
        # Create output directory
        os.makedirs(self.outputs_dir, exist_ok=True)
        
        self.eval_df = None
        self.agents = {}
        self.comparison_results = None
    
    def load_evaluation_data(self) -> pd.DataFrame:
        """
        Load and prepare evaluation data.
        Uses the pre-trained suppression model if available (much faster),
        otherwise retrains from scratch.
        
        Returns:
            Evaluation DataFrame with fraud probabilities
        """
        import joblib
        from .config import SUPPRESSION_MODEL_PATH
        
        print("\n" + "="*70)
        print("LOADING EVALUATION DATA")
        print("="*70)
        sys.stdout.flush()
        
        train_path = os.path.join(self.data_dir, 'train_feedback.csv')
        test_path = os.path.join(self.data_dir, 'test_feedback.csv')
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"Error: Data files not found. Expected:")
            print(f"  {train_path}")
            print(f"  {test_path}")
            return None
        
        print(f"Loading test data from {test_path}...")
        sys.stdout.flush()
        test_df = pd.read_csv(test_path)
        
        # Try to load existing model (fast path)
        if os.path.exists(SUPPRESSION_MODEL_PATH):
            print(f"Loading pre-trained model from {SUPPRESSION_MODEL_PATH}...")
            sys.stdout.flush()
            model = joblib.load(SUPPRESSION_MODEL_PATH)
            
            # Generate predictions using the loaded model
            leakage_cols = ['analyst_decision', 'analyst_id', 'analyst_weight', 'analyst_type', 'isFraud']
            X_test = test_df.drop(columns=[c for c in leakage_cols if c in test_df.columns])
            probs = model.predict_proba(X_test)
            
            eval_df = test_df.copy()
            eval_df['prob_benign'] = probs[:, 0]
            eval_df['prob_fraud'] = probs[:, 1]
        else:
            # Slow path: retrain model
            print("No saved model found — retraining suppression model...")
            sys.stdout.flush()
            train_df, _ = load_and_prepare_data(train_path, test_path)
            model, eval_df = train_suppression_model(train_df, test_df)
        
        self.eval_df = eval_df
        
        print(f"Loaded {len(eval_df):,} test samples")
        print(f"Fraud rate: {eval_df['isFraud'].mean():.4f}")
        print(f"Fraud probability range: [{eval_df['prob_fraud'].min():.4f}, "
              f"{eval_df['prob_fraud'].max():.4f}]")
        sys.stdout.flush()
        
        return eval_df
    
    def train_dqn_agent(self, num_episodes: int = RL_TRAINING_EPISODES,
                       batch_size: int = RL_BATCH_SIZE) -> Dict[str, Any]:
        """
        Train Deep Q-Network agent.
        
        Args:
            num_episodes: Training episodes
            batch_size: Batch size
            
        Returns:
            Evaluation metrics
        """
        print("\n" + "="*70)
        print("TRAINING DEEP Q-NETWORK (DQN) AGENT")
        print("="*70)
        
        trainer = RLTrainer(
            self.eval_df,
            agent_type='dqn',
            device='cpu',
            seed=self.seed,
            max_steps_per_episode=RL_MAX_STEPS_PER_EPISODE
        )
        
        # Train
        history = trainer.train(num_episodes=num_episodes, batch_size=batch_size, verbose=True)
        
        # Evaluate
        metrics = trainer.evaluate_policy(num_episodes=RL_EVALUATION_EPISODES)
        
        # Save model
        model_path = os.path.join(self.outputs_dir, 'dqn_agent.pkl')
        save_policy(trainer.agent, {'episodes': num_episodes, 'batch_size': batch_size},
                   model_path)
        
        self.agents['dqn'] = {
            'trainer': trainer,
            'agent': trainer.agent,
            'metrics': metrics,
            'history': history
        }
        
        return metrics
    
    def train_bandit_agent(self, num_episodes: int = RL_TRAINING_EPISODES) -> Dict[str, Any]:
        """
        Train Contextual Bandit agent.
        
        Args:
            num_episodes: Training episodes
            
        Returns:
            Evaluation metrics
        """
        print("\n" + "="*70)
        print("TRAINING CONTEXTUAL BANDIT AGENT")
        print("="*70)
        
        trainer = RLTrainer(
            self.eval_df,
            agent_type='bandit',
            seed=self.seed,
            max_steps_per_episode=RL_MAX_STEPS_PER_EPISODE
        )
        
        # Train
        history = trainer.train(num_episodes=num_episodes, verbose=True)
        
        # Evaluate
        metrics = trainer.evaluate_policy(num_episodes=RL_EVALUATION_EPISODES)
        
        # Save model
        model_path = os.path.join(self.outputs_dir, 'bandit_agent.pkl')
        save_policy(trainer.agent, {'episodes': num_episodes},
                   model_path)
        
        self.agents['bandit'] = {
            'trainer': trainer,
            'agent': trainer.agent,
            'metrics': metrics,
            'history': history
        }
        
        return metrics
    
    def compare_policies(self, baseline_thresholds: list = None) -> pd.DataFrame:
        """
        Compare all RL agents against static baseline policies.
        
        Args:
            baseline_thresholds: List of static thresholds to evaluate
            
        Returns:
            Comparison DataFrame
        """
        if baseline_thresholds is None:
            baseline_thresholds = RL_BASELINE_THRESHOLDS
        
        print("\n" + "="*70)
        print("POLICY COMPARISON")
        print("="*70)
        
        comparator = PolicyComparator(self.eval_df)
        
        # Get best RL metrics (prefer DQN if available, else bandit)
        if 'dqn' in self.agents:
            rl_metrics = self.agents['dqn']['metrics']
        else:
            rl_metrics = self.agents['bandit']['metrics']
        
        # Compare
        comparison_df = comparator.compare_policies(rl_metrics, baseline_thresholds)
        self.comparison_results = comparison_df
        
        # Save comparison
        comparison_path = os.path.join(self.outputs_dir, 'policy_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nComparison results saved to {comparison_path}")
        
        return comparison_df
    
    def generate_training_curves(self):
        """Generate training curves for all agents."""
        print("\nGenerating training curves...")
        
        num_agents = len(self.agents)
        if num_agents == 0:
            print("No agents trained yet.")
            return
        
        fig, axes = plt.subplots(1, num_agents, figsize=(6 * num_agents, 5))
        if num_agents == 1:
            axes = [axes]
        
        for idx, (agent_name, agent_data) in enumerate(self.agents.items()):
            history = agent_data['history']
            episodes = [h['episode'] for h in history]
            rewards = [h['total_reward'] for h in history]
            fraud_catch = [h['fraud_catch_rate'] for h in history]
            fp_reduction = [h['fp_reduction_rate'] for h in history]
            
            ax = axes[idx]
            ax2 = ax.twinx()
            
            # Rewards on left axis
            line1 = ax.plot(episodes, rewards, 'b-', label='Total Reward', linewidth=2)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Total Reward', color='b')
            ax.tick_params(axis='y', labelcolor='b')
            ax.grid(True, alpha=0.3)
            
            # Metrics on right axis
            line2 = ax2.plot(episodes, fraud_catch, 'r--', label='Fraud Catch Rate', linewidth=2)
            line3 = ax2.plot(episodes, fp_reduction, 'g--', label='FP Reduction Rate', linewidth=2)
            ax2.set_ylabel('Metric Rate')
            ax2.set_ylim(0, 1)
            
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
            
            ax.set_title(f'{agent_name.upper()} Training Progress')
        
        plt.tight_layout()
        chart_path = os.path.join(self.outputs_dir, 'training_curves.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to {chart_path}")
    
    def generate_policy_comparison_chart(self):
        """Generate comparison charts."""
        if self.comparison_results is None:
            print("No comparison results available.")
            return
        
        print("\nGenerating policy comparison charts...")
        
        df = self.comparison_results
        
        # Create comparison figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract data
        rl_row = df[df['Policy'] == 'RL Agent'].iloc[0]
        static_rows = df[df['Policy'] == 'Static Threshold']
        
        metrics = ['Fraud Catch Rate', 'FP Reduction Rate', 'Workload Savings', 'Avg Reward']
        ax_list = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = ax_list[idx]
            
            # Plot static policies
            x_vals = range(len(static_rows))
            static_vals = static_rows[metric].values
            
            ax.bar(x_vals, static_vals, alpha=0.7, label='Static Policy', color='steelblue')
            
            # Plot RL agent
            rl_val = rl_row[metric]
            ax.axhline(y=rl_val, color='red', linestyle='--', linewidth=2, label='RL Agent')
            
            ax.set_xlabel('Policy')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.set_xticks(x_vals)
            ax.set_xticklabels([f"T={t}" for t in static_rows['Parameter'].values], 
                              rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend()
        
        plt.tight_layout()
        chart_path = os.path.join(self.outputs_dir, 'policy_comparison.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Comparison chart saved to {chart_path}")
    
    def generate_adaptive_threshold_chart(self):
        """Generate adaptive threshold adjustment visualization."""
        print("\nGenerating adaptive threshold visualization...")
        
        workload_levels = np.linspace(0, 1, 100)
        base_thresholds = [0.80, 0.85, 0.90, 0.95]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for base_thresh in base_thresholds:
            policy = AdaptiveThresholdPolicy(base_threshold=base_thresh)
            
            adjusted = []
            for wl in workload_levels:
                adj_thresh = policy.adjust_threshold(wl)
                adjusted.append(adj_thresh)
            
            ax.plot(workload_levels, adjusted, marker='o', label=f'Base={base_thresh:.2f}',
                   linewidth=2, markersize=4)
        
        ax.set_xlabel('Workload Level', fontsize=12)
        ax.set_ylabel('Suppression Threshold (prob_benign >= X)', fontsize=12)
        ax.set_title('Adaptive Threshold Adjustment with Workload', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        chart_path = os.path.join(self.outputs_dir, 'adaptive_threshold.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Adaptive threshold chart saved to {chart_path}")
    
    def generate_report(self):
        """Generate comprehensive evaluation report."""
        print("\nGenerating comprehensive report...")
        
        report = []
        report.append("="*70)
        report.append("RL DECISION OPTIMIZATION EVALUATION REPORT")
        report.append("="*70)
        report.append("")
        
        # Data summary
        report.append("DATA SUMMARY:")
        report.append(f"  Test Set Size: {len(self.eval_df)}")
        report.append(f"  Fraud Rate: {self.eval_df['isFraud'].mean():.4f}")
        report.append(f"  Fraud Count: {self.eval_df['isFraud'].sum()}")
        report.append("")
        
        # Agent results
        report.append("TRAINED AGENTS:")
        for agent_name, agent_data in self.agents.items():
            metrics = agent_data['metrics']
            report.append(f"\n  {agent_name.upper()}:")
            report.append(f"    Fraud Catch Rate: {metrics['avg_fraud_catch_rate']:.4f}")
            report.append(f"    FP Reduction Rate: {metrics['avg_fp_reduction_rate']:.4f}")
            report.append(f"    Workload Savings: {metrics['avg_workload_savings']:.4f}")
            report.append(f"    Avg Episode Reward: {metrics['avg_total_reward']:.2f} "
                         f"(+/-{metrics['std_reward']:.2f})")
        report.append("")
        
        # Comparison
        if self.comparison_results is not None:
            report.append("POLICY COMPARISON:")
            report.append(self.comparison_results.to_string(index=False))
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("  1. RL agents provide adaptive decision-making based on system context")
        report.append("  2. Use DQN for more complex state-action relationships")
        report.append("  3. Use Contextual Bandit for interpretability and faster training")
        report.append("  4. Monitor fraud catch rate to ensure > 99% recall")
        report.append("  5. Periodically retrain to adapt to changing fraud patterns")
        report.append("")
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = os.path.join(self.outputs_dir, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nReport saved to {report_path}")
    
    def run_full_pipeline(self, num_episodes_dqn: int = RL_TRAINING_EPISODES,
                         num_episodes_bandit: int = RL_TRAINING_EPISODES,
                         train_dqn: bool = True,
                         train_bandit: bool = True) -> Dict[str, Any]:
        """
        Run complete RL optimization pipeline.
        
        Args:
            num_episodes_dqn: Episodes for DQN training
            num_episodes_bandit: Episodes for Bandit training
            train_dqn: Whether to train DQN agent
            train_bandit: Whether to train Bandit agent
            
        Returns:
            Results dictionary
        """
        results = {}
        
        # Load data
        self.load_evaluation_data()
        if self.eval_df is None:
            return results
        
        # Train agents
        if train_dqn:
            results['dqn'] = self.train_dqn_agent(num_episodes=num_episodes_dqn)
        
        if train_bandit:
            results['bandit'] = self.train_bandit_agent(num_episodes=num_episodes_bandit)
        
        # Compare policies
        if self.agents:
            results['comparison'] = self.compare_policies()
        
        # Generate visualizations
        self.generate_training_curves()
        self.generate_policy_comparison_chart()
        self.generate_adaptive_threshold_chart()
        
        # Generate report
        self.generate_report()
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETED")
        print(f"Results saved to: {os.path.abspath(self.outputs_dir)}")
        print("="*70)
        
        return results
