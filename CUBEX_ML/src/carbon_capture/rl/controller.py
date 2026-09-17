"""Process Control Coordinator interfacing RL recommendations with E-PINN validation."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from carbon_capture.rl.environment import ProcessState, ScrubberEnvironment
from carbon_capture.rl.q_learning import QLearningAgent
from carbon_capture.utils.logging import get_logger

logger = get_logger(__name__)

class RLProcessController:
    """Coordinates RL fan optimization and formats resulting process states."""

    def __init__(self, agent: QLearningAgent, env: ScrubberEnvironment):
        self.agent = agent
        self.env = env

    def train(self, episodes: int = 200, steps_per_episode: int = 50) -> List[float]:
        """Train Q-learning agent on the scrubber simulation."""
        logger.info(f"Training Q-learning controller for {episodes} episodes...")
        rewards_history = []

        for ep in range(episodes):
            state = self.env.reset()
            ep_reward = 0.0
            for _ in range(steps_per_episode):
                action = self.agent.select_action(state, training=True)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.update(state, action, reward, next_state, done)
                state = next_state
                ep_reward += reward
            rewards_history.append(ep_reward)

        logger.info(f"Q-learning training complete. Final episode reward: {rewards_history[-1]:.2f}")
        return rewards_history

    def get_control_recommendation(self, current_state: ProcessState) -> Dict[str, float]:
        """Recommend optimal fan adjustment."""
        action_idx = self.agent.select_action(current_state, training=False)
        delta_rpm = [-100.0, 0.0, 100.0][action_idx]
        target_rpm = float(np.clip(current_state.fan_speed_rpm + delta_rpm, self.env.min_rpm, self.env.max_rpm))

        return {
            "action_idx": action_idx,
            "delta_rpm": delta_rpm,
            "recommended_fan_speed_rpm": target_rpm,
        }
