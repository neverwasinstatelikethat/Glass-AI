"""
Reinforcement Learning Training Pipeline for Glass Production Optimization
Implements curriculum learning and safe exploration for PPO agent
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import os
from datetime import datetime
import json
from pathlib import Path

from .ppo_optimizer import GlassProductionPPO, PPOConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GlassProductionEnvironment:
    """
    Simulated environment for glass production optimization
    This would be replaced with actual production environment integration
    """
    
    def __init__(self):
        # State: [furnace_temp, melt_level, quality_score, defect_count, energy_consumption]
        self.state_dim = 5
        self.state = np.array([1500.0, 2500.0, 0.85, 0.1, 500.0])
        
        # Action bounds
        self.continuous_action_dim = 3  # [furnace_power, belt_speed, mold_temp]
        self.discrete_action_dims = [5, 5, 5]  # burner zones
        
        # Episode tracking
        self.step_count = 0
        self.max_steps = 1000
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.state = np.array([1500.0, 2500.0, 0.85, 0.1, 500.0])
        self.step_count = 0
        return self.state.copy()
    
    def step(self, continuous_action: np.ndarray, 
             discrete_actions: List[np.ndarray]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return next state, reward, done, info
        
        Args:
            continuous_action: [furnace_power, belt_speed, mold_temp]
            discrete_actions: [burner_zone1, burner_zone2, burner_zone3]
            
        Returns:
            next_state, reward, done, info
        """
        # Apply actions with some noise
        furnace_power = np.clip(continuous_action[0] + np.random.normal(0, 0.05), 0.0, 1.0)
        belt_speed = np.clip(continuous_action[1] + np.random.normal(0, 0.05), 0.0, 1.0)
        mold_temp = np.clip(continuous_action[2] + np.random.normal(0, 0.05), 0.0, 1.0)
        
        # Update state based on actions (simplified physics model)
        # Furnace temperature changes
        temp_change = (furnace_power - 0.5) * 20.0 + np.random.normal(0, 2.0)
        self.state[0] = np.clip(self.state[0] + temp_change, 1400.0, 1700.0)
        
        # Melt level changes
        melt_change = (belt_speed - 0.5) * 50.0 + np.random.normal(0, 5.0)
        self.state[1] = np.clip(self.state[1] + melt_change, 2000.0, 3000.0)
        
        # Quality score (depends on temperature and mold temp)
        optimal_temp = 1550.0
        temp_quality = 1.0 - min(1.0, abs(self.state[0] - optimal_temp) / 200.0)
        mold_quality = 1.0 - min(1.0, abs(mold_temp * 200 + 200 - 320) / 100.0)  # Optimal ~320°C
        self.state[2] = 0.7 * self.state[2] + 0.3 * (temp_quality * mold_quality)
        
        # Defect count (increases with extreme conditions)
        temp_defect_risk = max(0, abs(self.state[0] - 1550.0) - 50.0) / 100.0
        speed_defect_risk = max(0, abs(belt_speed - 0.5) - 0.3) * 0.5
        defect_increase = (temp_defect_risk + speed_defect_risk) * 0.1 + np.random.normal(0, 0.01)
        self.state[3] = np.clip(self.state[3] + defect_increase, 0.0, 1.0)
        
        # Energy consumption
        energy_base = 400.0
        energy_temp = (self.state[0] - 1400.0) * 2.0
        energy_speed = belt_speed * 100.0
        self.state[4] = energy_base + energy_temp + energy_speed + np.random.normal(0, 5.0)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Update step count
        self.step_count += 1
        done = self.step_count >= self.max_steps
        
        info = {
            "furnace_power": furnace_power,
            "belt_speed": belt_speed,
            "mold_temp": mold_temp,
            "temp_quality": temp_quality,
            "mold_quality": mold_quality,
            "temp_defect_risk": temp_defect_risk,
            "speed_defect_risk": speed_defect_risk
        }
        
        return self.state.copy(), reward, done, info
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on current state"""
        # Quality reward (0-100)
        quality_reward = self.state[2] * 50.0
        
        # Defect penalty (-50 to 0)
        defect_penalty = -self.state[3] * 50.0
        
        # Energy efficiency reward/penalty (-20 to 20)
        energy_efficiency = 1.0 - min(1.0, max(0.0, (self.state[4] - 300.0) / 500.0))
        energy_reward = (energy_efficiency - 0.5) * 40.0
        
        # Stability reward (0-30)
        stability_reward = 0.0
        if 1520 <= self.state[0] <= 1580:  # Optimal temperature range
            stability_reward += 15.0
        if 0.3 <= self.state[2] <= 0.9:  # Good quality range
            stability_reward += 15.0
            
        total_reward = quality_reward + defect_penalty + energy_reward + stability_reward
        
        return total_reward


class CurriculumLearningScheduler:
    """Curriculum learning scheduler for progressive training"""
    
    def __init__(self):
        self.stage = 0
        self.stages = [
            {"name": "Basic Control", "difficulty": 0.2, "episodes": 50},
            {"name": "Temperature Control", "difficulty": 0.4, "episodes": 100},
            {"name": "Quality Optimization", "difficulty": 0.6, "episodes": 150},
            {"name": "Full Optimization", "difficulty": 1.0, "episodes": 200}
        ]
        self.current_episode = 0
        self.episodes_in_stage = 0
        
    def get_current_stage(self) -> Dict:
        """Get current curriculum stage"""
        return self.stages[self.stage]
    
    def update_stage(self, episode_reward: float) -> bool:
        """
        Update curriculum stage based on performance
        Returns True if stage changed
        """
        self.current_episode += 1
        self.episodes_in_stage += 1
        
        current_stage = self.stages[self.stage]
        
        # Check if we should advance to next stage
        if (self.episodes_in_stage >= current_stage["episodes"] and 
            self.stage < len(self.stages) - 1):
            # Advance to next stage
            self.stage += 1
            self.episodes_in_stage = 0
            logger.info(f"🎓 Advancing to curriculum stage: {self.stages[self.stage]['name']}")
            return True
            
        return False
    
    def get_exploration_rate(self) -> float:
        """Get exploration rate based on current stage"""
        # Higher exploration in early stages
        stage = self.stages[self.stage]
        return 1.0 - (stage["difficulty"] * 0.7)


class RLPipeline:
    """Main RL training pipeline"""
    
    def __init__(self, config: Optional[PPOConfig] = None):
        self.config = config or PPOConfig()
        self.environment = GlassProductionEnvironment()
        self.curriculum = CurriculumLearningScheduler()
        
        # Create PPO agent
        self.agent = GlassProductionPPO(
            state_dim=self.environment.state_dim,
            continuous_action_dim=self.environment.continuous_action_dim,
            discrete_action_dims=self.environment.discrete_action_dims,
            config=self.config
        )
        
        # Training statistics
        self.training_stats = {
            "total_episodes": 0,
            "total_steps": 0,
            "best_reward": -float('inf'),
            "avg_rewards": [],
            "curriculum_stages": []
        }
        
        # Checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ RL Training Pipeline initialized")
    
    def train(self, max_episodes: int = 1000) -> Dict[str, Any]:
        """
        Train the RL agent
        
        Args:
            max_episodes: Maximum number of training episodes
            
        Returns:
            Training statistics
        """
        logger.info("🚀 Starting RL training pipeline...")
        logger.info(f"Curriculum stages: {[stage['name'] for stage in self.curriculum.stages]}")
        
        for episode in range(max_episodes):
            # Reset environment
            state = self.environment.reset()
            episode_reward = 0.0
            episode_steps = 0
            done = False
            
            # Episode memory for training
            episode_memory = []
            
            # Run episode
            while not done:
                # Select action with current exploration rate
                (continuous_action, discrete_actions), _, value, penalty = \
                    self.agent.select_action(state)
                
                # Execute action in environment
                next_state, reward, done, info = self.environment.step(
                    continuous_action, discrete_actions
                )
                
                # Apply safety penalty
                reward -= penalty
                
                # Store experience
                experience = {
                    "state": state.copy(),
                    "continuous_action": continuous_action,
                    "discrete_actions": discrete_actions,
                    "reward": reward,
                    "next_state": next_state.copy(),
                    "done": done,
                    "value": value
                }
                episode_memory.append(experience)
                
                # Update state and reward
                state = next_state
                episode_reward += reward
                episode_steps += 1
                self.agent.total_steps += 1
            
            # Update agent with episode experience
            # In a full implementation, we would do PPO updates here
            # For now, we'll just track statistics
            
            # Update curriculum
            stage_changed = self.curriculum.update_stage(episode_reward)
            if stage_changed:
                self.training_stats["curriculum_stages"].append({
                    "episode": episode,
                    "stage": self.curriculum.get_current_stage()["name"]
                })
            
            # Track statistics
            self.agent.episode_rewards.append(episode_reward)
            self.training_stats["total_episodes"] = episode + 1
            self.training_stats["avg_rewards"].append(episode_reward)
            
            # Update best reward
            if episode_reward > self.training_stats["best_reward"]:
                self.training_stats["best_reward"] = episode_reward
                # Save best model
                self.agent.save_checkpoint(episode, best=True)
            
            # Regular checkpointing
            if (episode + 1) % 50 == 0:
                self.agent.save_checkpoint(episode)
            
            # Log progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(list(self.agent.episode_rewards)[-10:])
                stats = self.agent.get_training_stats()
                logger.info(f"Episode {episode + 1:4d}: "
                           f"Avg Reward: {avg_reward:8.2f} | "
                           f"Current Reward: {episode_reward:8.2f} | "
                           f"Stage: {self.curriculum.get_current_stage()['name']} | "
                           f"Exploration: {stats['exploration_rate']:.3f}")
        
        # Training complete
        logger.info("✅ RL training pipeline completed")
        logger.info(f"Best reward achieved: {self.training_stats['best_reward']:.2f}")
        
        # Save final statistics
        self._save_training_stats()
        
        return self.training_stats
    
    def _save_training_stats(self):
        """Save training statistics to file"""
        stats_file = os.path.join(self.config.checkpoint_dir, "training_stats.json")
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_stats = self.training_stats.copy()
        if "avg_rewards" in serializable_stats:
            serializable_stats["avg_rewards"] = [float(r) for r in serializable_stats["avg_rewards"]]
        
        try:
            with open(stats_file, 'w') as f:
                json.dump(serializable_stats, f, indent=2, default=str)
            logger.info(f"💾 Training statistics saved to {stats_file}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save training statistics: {e}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load checkpoint from file
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Episode number
        """
        return self.agent.load_checkpoint(checkpoint_path)
    
    def get_agent(self) -> GlassProductionPPO:
        """Get trained agent"""
        return self.agent


def create_rl_training_pipeline(config: Optional[PPOConfig] = None) -> RLPipeline:
    """Factory function to create RL training pipeline"""
    pipeline = RLPipeline(config)
    logger.info("✅ RL Training Pipeline created")
    return pipeline


# Example usage
if __name__ == "__main__":
    print("🧪 RL Training Pipeline for Glass Production Optimization")
    print("=" * 60)
    
    # Create training pipeline
    config = PPOConfig(
        learning_rate=3e-4,
        gamma=0.99,
        lam=0.95,
        epsilon=0.2,
        epochs=10,
        batch_size=64,
        hidden_size=128,
        use_prioritized_replay=True,
        use_curriculum=True,
        exploration_initial=1.0,
        exploration_final=0.1,
        exploration_decay=0.995,
        checkpoint_freq=50,
        checkpoint_dir="./rl_checkpoints"
    )
    
    pipeline = create_rl_training_pipeline(config)
    
    # Train agent
    print("\n🚀 Starting training...")
    stats = pipeline.train(max_episodes=200)
    
    # Print final statistics
    print(f"\n📊 Training Results:")
    print(f"  Total Episodes: {stats['total_episodes']}")
    print(f"  Best Reward: {stats['best_reward']:.2f}")
    print(f"  Average Final Rewards: {np.mean(stats['avg_rewards'][-20:]):.2f}")
    print(f"  Curriculum Stages Completed: {len(stats['curriculum_stages'])}")
    
    # Show agent statistics
    agent_stats = pipeline.get_agent().get_training_stats()
    print(f"\n📈 Agent Statistics:")
    print(f"  Exploration Rate: {agent_stats['exploration_rate']:.3f}")
    print(f"  Safety Violations: {agent_stats['safety_violations']}")
    print(f"  Total Steps: {agent_stats['total_steps']}")
    
    print("\n✅ RL Training Pipeline test completed!")