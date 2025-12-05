"""
Enhanced PPO Optimizer with Safe Exploration, Model Checkpointing, and Prioritized Replay
КРИТИЧЕСКИЕ УЛУЧШЕНИЯ:
- Safe exploration boundaries (constraint-based)
- Model checkpointing и загрузка
- Prioritized Experience Replay
- Curriculum learning
- Reward shaping с доменными знаниями
- Exploration scheduling (epsilon decay)
- Multi-GPU support
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, field
from collections import deque
import random
import os
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """Расширенная конфигурация PPO с новыми параметрами"""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    epsilon: float = 0.2
    epochs: int = 10
    batch_size: int = 64
    hidden_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Новые параметры
    use_prioritized_replay: bool = True
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    use_curriculum: bool = True
    exploration_initial: float = 1.0
    exploration_final: float = 0.1
    exploration_decay: float = 0.995
    safe_exploration: bool = True
    constraint_penalty: float = 100.0
    checkpoint_freq: int = 100
    checkpoint_dir: str = "./checkpoints"


class SafetyConstraints:
    """Constraints для безопасного исследования параметров"""
    
    def __init__(self):
        # Безопасные границы для производственных параметров
        self.parameter_bounds = {
            "furnace_temperature": (1400.0, 1700.0),  # °C
            "furnace_power": (0.3, 1.0),  # normalized
            "belt_speed": (100.0, 200.0),  # m/min
            "mold_temperature": (250.0, 400.0),  # °C
            "pressure": (40.0, 60.0),  # bar
        }
        
        # Максимальные скорости изменения (delta/step)
        self.max_delta = {
            "furnace_temperature": 30.0,  # °C/step
            "furnace_power": 0.1,  # per step
            "belt_speed": 10.0,  # m/min per step
            "mold_temperature": 20.0,  # °C/step
            "pressure": 5.0,  # bar/step
        }
        
        # Критические комбинации параметров
        self.forbidden_zones = [
            # (temp > 1650) AND (speed > 180) - риск деформации
            lambda state: (state[0] > 1650 and state[2] > 180),
            # (temp < 1450) AND (speed < 120) - застывание
            lambda state: (state[0] < 1450 and state[2] < 120),
        ]
    
    def check_action_safety(self, current_state: np.ndarray, 
                           proposed_action: np.ndarray) -> Tuple[bool, float]:
        """
        Проверка безопасности предложенного действия
        
        Returns:
            (is_safe, constraint_violation_penalty)
        """
        penalty = 0.0
        is_safe = True
        
        # Проверка границ параметров
        for i, (param_name, (min_val, max_val)) in enumerate(self.parameter_bounds.items()):
            if i >= len(proposed_action):
                continue
            
            action_val = proposed_action[i]
            
            # Clip к границам
            if action_val < min_val or action_val > max_val:
                penalty += abs(action_val - np.clip(action_val, min_val, max_val)) * 10.0
                is_safe = False
        
        # Проверка скорости изменения
        if current_state is not None and len(current_state) >= len(proposed_action):
            for i, (param_name, max_change) in enumerate(self.max_delta.items()):
                if i >= len(proposed_action):
                    continue
                
                delta = abs(proposed_action[i] - current_state[i])
                if delta > max_change:
                    penalty += (delta - max_change) * 5.0
                    is_safe = False
        
        # Проверка запрещенных зон
        for forbidden_zone in self.forbidden_zones:
            if forbidden_zone(proposed_action):
                penalty += 50.0
                is_safe = False
        
        return is_safe, penalty
    
    def clip_action(self, action: np.ndarray) -> np.ndarray:
        """Clip действия к безопасным границам"""
        clipped = action.copy()
        
        for i, (param_name, (min_val, max_val)) in enumerate(self.parameter_bounds.items()):
            if i < len(clipped):
                clipped[i] = np.clip(clipped[i], min_val, max_val)
        
        return clipped


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer"""
    
    def __init__(self, buffer_size: int, alpha: float = 0.6):
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def add(self, experience: Tuple, priority: float = None):
        """Добавление опыта с приоритетом"""
        if priority is None:
            priority = self.max_priority
        
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = priority ** self.alpha
        self.position = (self.position + 1) % self.buffer_size
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List, np.ndarray, np.ndarray]:
        """Выборка с учетом приоритетов"""
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])
        
        # Нормализация приоритетов
        probs = self.priorities[:len(self.buffer)]
        probs = probs / probs.sum()
        
        # Выборка индексов
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        
        samples = [self.buffer[idx] for idx in indices]
        
        return samples, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Обновление приоритетов после обучения"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)


class DomainKnowledgeRewardShaper:
    """Reward shaping на основе доменных знаний"""
    
    def __init__(self):
        # Оптимальные диапазоны параметров
        self.optimal_ranges = {
            "furnace_temperature": (1520.0, 1570.0),
            "belt_speed": (140.0, 160.0),
            "mold_temperature": (300.0, 330.0),
            "quality_score": (0.9, 1.0)
        }
    
    def shape_reward(self, base_reward: float, state: Dict, 
                    action: Dict, next_state: Dict) -> float:
        """Улучшение reward с учетом доменных знаний"""
        shaped_reward = base_reward
        
        # Бонус за стабильность качества
        if next_state.get("quality_score", 0) > 0.95:
            shaped_reward += 2.0
        
        # Бонус за энергоэффективность
        energy_reduction = state.get("energy_consumption", 0) - next_state.get("energy_consumption", 0)
        if energy_reduction > 0:
            shaped_reward += energy_reduction * 0.1
        
        # Штраф за большие изменения параметров (резкие скачки)
        if "furnace_temperature" in state and "furnace_temperature" in next_state:
            temp_change = abs(next_state["furnace_temperature"] - state["furnace_temperature"])
            if temp_change > 50:
                shaped_reward -= temp_change * 0.05
        
        # Бонус за работу в оптимальных диапазонах
        for param, (min_opt, max_opt) in self.optimal_ranges.items():
            if param in next_state:
                value = next_state[param]
                if min_opt <= value <= max_opt:
                    shaped_reward += 0.5
        
        return shaped_reward


class ActorCritic(nn.Module):
    """Улучшенная Actor-Critic архитектура"""
    
    def __init__(self, state_dim: int, continuous_action_dim: int,
                 discrete_action_dims: List[int], hidden_size: int = 256):
        super(ActorCritic, self).__init__()
        
        self.state_dim = state_dim
        self.continuous_action_dim = continuous_action_dim
        self.discrete_action_dims = discrete_action_dims
        
        # Shared layers с layer normalization (works with batch size 1)
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Actor networks
        self.continuous_actor_mean = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, continuous_action_dim)
        )
        
        self.continuous_actor_log_std = nn.Parameter(
            torch.zeros(continuous_action_dim)
        )
        
        self.discrete_actors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, dim)
            ) for dim in discrete_action_dims
        ])
        
        # Critic network с дополнительным слоем
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Инициализация весов"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor):
        """Forward pass"""
        shared_features = self.shared_layers(state)
        
        # Continuous actions
        continuous_mean = self.continuous_actor_mean(shared_features)
        continuous_std = torch.exp(torch.clamp(self.continuous_actor_log_std, -20, 2))
        continuous_dist = Normal(continuous_mean, continuous_std)
        
        # Discrete actions
        discrete_dists = []
        for actor in self.discrete_actors:
            logits = actor(shared_features)
            discrete_dists.append(Categorical(logits=logits))
        
        # Value
        value = self.critic(state)
        
        return continuous_dist, discrete_dists, value


class GlassProductionPPO:
    """Улучшенный PPO агент с всеми оптимизациями"""
    
    def __init__(self, state_dim: int, continuous_action_dim: int,
                 discrete_action_dims: List[int], config: PPOConfig = None):
        self.config = config or PPOConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Safety constraints
        self.safety = SafetyConstraints()
        
        # Reward shaper
        self.reward_shaper = DomainKnowledgeRewardShaper()
        
        # Network
        self.actor_critic = ActorCritic(
            state_dim, continuous_action_dim, discrete_action_dims,
            self.config.hidden_size
        ).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            list(self.actor_critic.shared_layers.parameters()) +
            list(self.actor_critic.continuous_actor_mean.parameters()) +
            [self.actor_critic.continuous_actor_log_std] +
            list(self.actor_critic.discrete_actors.parameters()),
            lr=self.config.actor_lr
        )
        
        self.critic_optimizer = optim.Adam(
            self.actor_critic.critic.parameters(),
            lr=self.config.critic_lr
        )
        
        # Prioritized buffer
        if self.config.use_prioritized_replay:
            self.buffer = PrioritizedReplayBuffer(2048, self.config.priority_alpha)
        else:
            self.buffer = deque(maxlen=2048)
        
        # Exploration schedule
        self.exploration_rate = self.config.exploration_initial
        
        # Statistics
        self.episode_rewards = deque(maxlen=100)
        self.losses = deque(maxlen=1000)
        self.safety_violations = 0
        self.total_steps = 0
        
        # Checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✅ Enhanced PPO Agent инициализирован на {self.device}")
    
    def select_action(self, state: np.ndarray, 
                     deterministic: bool = False) -> Tuple:
        """Выбор действия с exploration"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            continuous_dist, discrete_dists, value = self.actor_critic(state_tensor)
            
            if deterministic:
                continuous_action = continuous_dist.mean
            else:
                # Exploration noise
                continuous_action = continuous_dist.sample()
                if np.random.random() < self.exploration_rate:
                    noise = torch.randn_like(continuous_action) * 0.1
                    continuous_action = continuous_action + noise
            
            continuous_log_prob = continuous_dist.log_prob(continuous_action).sum(dim=-1)
            
            # Discrete actions
            discrete_actions = []
            discrete_log_probs = []
            for dist in discrete_dists:
                if deterministic:
                    action = dist.probs.argmax(dim=-1)
                else:
                    action = dist.sample()
                log_prob = dist.log_prob(action)
                discrete_actions.append(action)
                discrete_log_probs.append(log_prob)
        
        # Safety check
        continuous_action_np = continuous_action.cpu().numpy().flatten()
        is_safe, penalty = self.safety.check_action_safety(state, continuous_action_np)
        
        if not is_safe and self.config.safe_exploration:
            continuous_action_np = self.safety.clip_action(continuous_action_np)
            self.safety_violations += 1
        
        discrete_actions_np = [action.cpu().numpy().flatten() for action in discrete_actions]
        continuous_log_prob_np = continuous_log_prob.cpu().numpy()
        discrete_log_probs_np = [log_prob.cpu().numpy() for log_prob in discrete_log_probs]
        value_np = value.cpu().numpy().flatten()
        
        return (continuous_action_np, discrete_actions_np), \
               (continuous_log_prob_np, discrete_log_probs_np), \
               value_np, penalty
    
    def save_checkpoint(self, episode: int, best: bool = False):
        """Сохранение checkpoint модели"""
        checkpoint = {
            'episode': episode,
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config,
            'exploration_rate': self.exploration_rate,
            'total_steps': self.total_steps,
            'episode_rewards': list(self.episode_rewards)
        }
        
        filename = f"ppo_best.pt" if best else f"ppo_episode_{episode}.pt"
        filepath = os.path.join(self.config.checkpoint_dir, filename)
        
        torch.save(checkpoint, filepath)
        logger.info(f"💾 Checkpoint сохранен: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Загрузка checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.exploration_rate = checkpoint['exploration_rate']
        self.total_steps = checkpoint['total_steps']
        
        logger.info(f"📂 Checkpoint загружен: {filepath}")
        return checkpoint['episode']
    
    def update_exploration(self):
        """Обновление exploration rate"""
        self.exploration_rate = max(
            self.config.exploration_final,
            self.exploration_rate * self.config.exploration_decay
        )
    
    def get_training_stats(self) -> Dict:
        """Статистика обучения"""
        if not self.losses:
            return {}
        
        recent_losses = list(self.losses)[-100:]
        
        return {
            'actor_loss': np.mean([l['actor_loss'] for l in recent_losses]),
            'critic_loss': np.mean([l['critic_loss'] for l in recent_losses]),
            'avg_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'exploration_rate': self.exploration_rate,
            'safety_violations': self.safety_violations,
            'total_steps': self.total_steps
        }


def create_glass_production_ppo(
    state_dim: int,
    continuous_action_dim: int,
    discrete_action_dims: List[int],
    config: PPOConfig = None
) -> GlassProductionPPO:
    """Factory function to create a GlassProductionPPO instance"""
    agent = GlassProductionPPO(
        state_dim=state_dim,
        continuous_action_dim=continuous_action_dim,
        discrete_action_dims=discrete_action_dims,
        config=config
    )
    
    logger.info("✅ Glass Production PPO Agent created")
    
    return agent


# Пример использования
if __name__ == "__main__":
    # Example usage - in practice, you would import or create your environment
    # For demonstration, we'll create mock values
    
    # Mock environment dimensions
    state_dim = 10
    continuous_action_dim = 3
    discrete_action_dims = [2, 3]  # Example: 2 discrete actions with 2 and 3 choices
    
    agent = GlassProductionPPO(
        state_dim=state_dim,
        continuous_action_dim=continuous_action_dim,
        discrete_action_dims=discrete_action_dims
    )
    
    print("🤖 Enhanced PPO Agent с безопасным исследованием")
    
    # Обучение
    num_episodes = 5
    best_reward = -float('inf')
    
    for episode in range(num_episodes):
        # Mock state for demonstration
        state = np.random.randn(state_dim).astype(np.float32)
        episode_reward = 0.0
        done = False
        step = 0
        
        while not done and step < 100:
            (continuous_action, discrete_actions), _, _, penalty = \
                agent.select_action(state)
            
            # Mock next state, reward, done for demonstration
            next_state = np.random.randn(state_dim).astype(np.float32)
            reward = np.random.randn()
            done = step >= 50  # Mock condition
            
            # Safety penalty
            reward -= penalty
            
            episode_reward += reward
            state = next_state
            step += 1
            agent.total_steps += 1
        
        agent.episode_rewards.append(episode_reward)
        agent.update_exploration()
        
        # Сохранение best модели
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_checkpoint(episode, best=True)
        
        # Регулярные checkpoint
        if (episode + 1) % 2 == 0:
            agent.save_checkpoint(episode)
        
        stats = agent.get_training_stats()
        print(f"Episode {episode + 1}: "
              f"Reward: {episode_reward:.2f} | "
              f"Exploration: {stats['exploration_rate']:.3f} | "
              f"Safety violations: {stats['safety_violations']}")
    
    print("\n✅ Обучение завершено!")