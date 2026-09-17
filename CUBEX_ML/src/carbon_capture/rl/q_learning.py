"""Tabular Q-Learning Algorithm with State Discretization."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from carbon_capture.rl.environment import ProcessState
from carbon_capture.utils.logging import get_logger

logger = get_logger(__name__)

class QLearningAgent:
    """Q-learning agent for fan speed process control."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int = 42,
    ):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        # 3 discrete actions: -100 RPM, 0 RPM, +100 RPM
        self.num_actions = 3
        self.q_table: Dict[Tuple[int, int, int, int], np.ndarray] = {}

    def discretize_state(self, state: ProcessState) -> Tuple[int, int, int, int]:
        """Discretize continuous sensor states into finite grid bins."""
        co2_bin = int(np.clip((state.co2_concentration_pct - 6.0) / 1.5, 0, 9))
        temp_bin = int(np.clip((state.temperature_c - 30.0) / 5.0, 0, 9))
        rpm_bin = int(np.clip((state.fan_speed_rpm - 1000.0) / 100.0, 0, 10))
        flow_bin = int(np.clip((state.gas_flow_m3_s - 15.0) / 4.0, 0, 9))
        return (co2_bin, temp_bin, rpm_bin, flow_bin)

    def get_q_values(self, discrete_state: Tuple[int, int, int, int]) -> np.ndarray:
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.num_actions, dtype=np.float32)
        return self.q_table[discrete_state]

    def select_action(self, state: ProcessState, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        discrete_s = self.discretize_state(state)
        q_vals = self.get_q_values(discrete_s)

        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_actions))
        return int(np.argmax(q_vals))

    def update(
        self,
        state: ProcessState,
        action: int,
        reward: float,
        next_state: ProcessState,
        done: bool,
    ) -> float:
        """Bellman optimality update on Q-table."""
        s = self.discretize_state(state)
        s_next = self.discretize_state(next_state)

        q_current = self.get_q_values(s)[action]
        q_next_max = 0.0 if done else np.max(self.get_q_values(s_next))

        target = reward + self.gamma * q_next_max
        td_error = target - q_current
        self.q_table[s][action] += self.lr * td_error

        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return float(td_error)

    def save(self, file_path: Union[str, Path]) -> None:
        """Save Q-table to disk."""
        import pickle
        with open(file_path, "wb") as f:
            pickle.dump({
                "q_table": self.q_table,
                "epsilon": self.epsilon,
                "lr": self.lr,
                "gamma": self.gamma,
            }, f)

    def load(self, file_path: Union[str, Path]) -> None:
        """Load Q-table from disk."""
        import pickle
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            self.q_table = data["q_table"]
            self.epsilon = data["epsilon"]
            self.lr = data["lr"]
            self.gamma = data["gamma"]
