"""
RL Agent for Alert Suppression Optimization.

This module implements a Q-learning agent with neural network function approximation
to learn optimal suppression decisions. The agent learns a policy that balances
fraud detection (recall) with analyst workload reduction.

Algorithm: Deep Q-Learning (DQN) with experience replay and target networks.
State: [prob_fraud, rule_count_norm, amount_norm, fraud_rate, workload_level]
Actions: [SUPPRESS=0, ESCALATE=1]
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Dict, Any
from collections import deque
import random


class QNetwork(nn.Module):
    """
    Neural network for Q-value approximation (DQN).
    
    Input: 5D state vector
    Output: Q-values for 2 actions
    """
    
    def __init__(self, state_dim: int = 5, action_dim: int = 2, hidden_dim: int = 64):
        """
        Initialize Q-network.
        
        Args:
            state_dim: Dimension of state space (default 5)
            action_dim: Dimension of action space (default 2: suppress/escalate)
            hidden_dim: Hidden layer dimension
        """
        super(QNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(state)


class ReplayBuffer:
    """Experience replay buffer for DQN training stability."""
    
    def __init__(self, capacity: int = 10000):
        """Initialize replay buffer."""
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Store experience in buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                np.ndarray, np.ndarray]:
        """Sample batch of experiences."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self) -> int:
        return len(self.buffer)


class QLearningAgent:
    """
    Deep Q-Learning agent for alert suppression optimization.
    
    Learns a policy that decides whether to suppress or escalate each alert
    based on context and system state.
    """
    
    def __init__(self, state_dim: int = 5, action_dim: int = 2, 
                 learning_rate: float = 0.001, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995, device: str = None):
        """
        Initialize Q-learning agent.
        
        Args:
            state_dim: State space dimension (default 5)
            action_dim: Action space dimension (default 2)
            learning_rate: Optimizer learning rate
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay factor for epsilon-greedy strategy
            device: 'cpu' or 'cuda' (auto-detect if None)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        
        # Device selection
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Networks
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for robustness
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # Training statistics
        self.training_losses = []
        self.episode_rewards = []
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy strategy.
        
        Args:
            state: Current state vector
            training: Whether in training mode (affects epsilon)
            
        Returns:
            Action (0=SUPPRESS, 1=ESCALATE)
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.action_dim)
        else:
            # Exploit: greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax(dim=1).item()
    
    def train_step(self, batch_size: int = 32) -> float:
        """
        Perform one training step using experience replay.
        
        Args:
            batch_size: Batch size for training
            
        Returns:
            Loss value for monitoring
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values (policy network)
        current_q_values = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        # Next Q-values (target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_states_t).max(dim=1)[0]
            target_q_values = rewards_t + self.gamma * next_q_values * (1 - dones_t)
        
        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network weights (hard update)."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get training statistics."""
        return {
            'epsilon': self.epsilon,
            'avg_loss': np.mean(self.training_losses[-100:]) if self.training_losses else 0.0,
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0,
        }


class ContextualBanditAgent:
    """
    Alternative simplified agent using contextual bandit approach.
    
    Uses linear regression to learn context-dependent action values instead of
    deep Q-learning. Faster training, interpretable weights.
    """
    
    def __init__(self, state_dim: int = 5, action_dim: int = 2,
                 learning_rate: float = 0.01):
        """
        Initialize contextual bandit agent.
        
        Args:
            state_dim: State dimension
            action_dim: Number of actions
            learning_rate: Learning rate for gradient descent
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # Linear weights for each action
        self.weights = np.random.normal(0, 0.01, size=(action_dim, state_dim))
        self.action_counts = np.zeros(action_dim)
        self.epsilon = 0.1
        
        self.training_losses = []
        self.episode_rewards = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy strategy with linear Q-values.
        
        Args:
            state: Current state vector
            training: Whether in training mode
            
        Returns:
            Action (0=SUPPRESS, 1=ESCALATE)
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        # Compute Q-values for each action (linear)
        q_values = self.weights @ state
        return np.argmax(q_values)
    
    def train_step(self, batch_of_transitions: List[Tuple]) -> float:
        """
        Update weights using batch of transitions.
        
        Args:
            batch_of_transitions: List of (state, action, reward, next_state, done) tuples
            
        Returns:
            Loss value
        """
        if not batch_of_transitions:
            return 0.0
        
        total_loss = 0.0
        for state, action, reward, next_state, done in batch_of_transitions:
            # Current Q-value
            current_q = self.weights[action] @ state
            
            # Target Q-value
            if done:
                target_q = reward
            else:
                next_q_values = self.weights @ next_state
                target_q = reward + 0.99 * np.max(next_q_values)
            
            # Temporal difference error
            td_error = target_q - current_q
            loss = td_error ** 2
            
            # Gradient descent update
            gradient = -2 * td_error * state
            self.weights[action] -= self.learning_rate * gradient
            
            total_loss += loss
        
        avg_loss = total_loss / len(batch_of_transitions) if batch_of_transitions else 0.0
        self.training_losses.append(avg_loss)
        return avg_loss
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get training statistics."""
        return {
            'avg_loss': np.mean(self.training_losses[-100:]) if self.training_losses else 0.0,
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0,
            'epsilon': self.epsilon,
        }
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """
        Get feature importance for each action.
        
        Returns:
            Dictionary mapping action names to weight vectors
        """
        feature_names = ['prob_fraud', 'rule_count', 'amount', 'fraud_rate', 'workload']
        return {
            'SUPPRESS': self.weights[0],
            'ESCALATE': self.weights[1],
            'feature_names': feature_names
        }
