"""Carbon Capture Scrubber Process Environment for Q-Learning Control."""

from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class ProcessState:
    co2_concentration_pct: float  # e.g., 8.0 - 15.0 %
    temperature_c: float          # e.g., 35.0 - 65.0 C
    fan_speed_rpm: float          # 1000 - 2000 RPM
    gas_flow_m3_s: float          # e.g., 20.0 - 50.0 m3/s

class ScrubberEnvironment:
    """Simulates packed-bed scrubber dynamics for auxiliary fan optimization."""

    def __init__(
        self,
        min_rpm: float = 1000.0,
        max_rpm: float = 2000.0,
        target_co2_pct: float = 10.0,
        seed: int = 42,
    ):
        self.min_rpm = min_rpm
        self.max_rpm = max_rpm
        self.target_co2_pct = target_co2_pct
        self.rng = np.random.default_rng(seed)
        self.state = self.reset()

    def reset(self) -> ProcessState:
        """Reset environment to standard starting conditions."""
        self.state = ProcessState(
            co2_concentration_pct=float(self.rng.uniform(10.0, 14.0)),
            temperature_c=float(self.rng.uniform(40.0, 55.0)),
            fan_speed_rpm=1500.0,
            gas_flow_m3_s=35.0,
        )
        return self.state

    def step(self, action_idx: int) -> Tuple[ProcessState, float, bool, dict]:
        """Apply fan speed adjustment action (-100, 0, +100 RPM)."""
        # Actions: 0 -> -100 RPM, 1 -> 0 RPM, 2 -> +100 RPM
        delta_rpm = [-100.0, 0.0, 100.0][action_idx]

        new_rpm = np.clip(self.state.fan_speed_rpm + delta_rpm, self.min_rpm, self.max_rpm)

        # Physics response: higher fan speed increases gas flow, reduces CO2 concentration, increases power
        gas_flow = 15.0 + (new_rpm / self.max_rpm) * 35.0 + float(self.rng.normal(0, 0.5))
        new_co2 = max(2.0, self.state.co2_concentration_pct - (delta_rpm / 500.0) + float(self.rng.normal(0, 0.1)))
        new_temp = np.clip(self.state.temperature_c + float(self.rng.normal(0, 0.2)), 30.0, 70.0)

        self.state = ProcessState(
            co2_concentration_pct=float(new_co2),
            temperature_c=float(new_temp),
            fan_speed_rpm=float(new_rpm),
            gas_flow_m3_s=float(gas_flow),
        )

        # Reward: penalize deviation from target CO2 and excessive fan power
        co2_error = abs(self.state.co2_concentration_pct - self.target_co2_pct)
        power_penalty = (new_rpm / self.max_rpm) ** 2
        reward = -1.0 * co2_error - 0.5 * power_penalty

        done = False
        info = {
            "fan_speed_rpm": new_rpm,
            "power_penalty": power_penalty,
            "co2_error": co2_error,
        }

        return self.state, reward, done, info
