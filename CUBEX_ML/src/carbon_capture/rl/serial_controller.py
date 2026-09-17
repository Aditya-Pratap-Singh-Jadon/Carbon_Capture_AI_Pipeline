"""Phase 11: RL Control Loop — Source-Agnostic Process Controller Interface.

Architecture:
    Telemetry (Digital Twin / Arduino / CSV)
        ↓
    ProcessState (derived from observed telemetry)
        ↓
    QLearningAgent.select_action()
        ↓
    FanCommand (delta_rpm, target_rpm)
        ↓
    Digital Twin / future hardware (external)
        ↓
    New telemetry observation

CRITICAL RULES (enforced here):
- Q-learning determines FAN COMMANDS only.
- Q-learning does NOT calculate F_valid.
- Q-learning does NOT calculate carbon credits.
- Q-learning does NOT override E-PINN validation.
- Q-learning does NOT modify physics residuals.
- Commanded RPM ≠ Observed RPM (the Digital Twin is responsible for applying commands).

The interface is source-agnostic: it accepts a telemetry record and returns a command.
Whether the telemetry came from a Virtual COM or a real Arduino is irrelevant to the controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from carbon_capture.input.source.base import DigitalTwinTelemetryRecord
from carbon_capture.rl.environment import ProcessState, ScrubberEnvironment
from carbon_capture.rl.q_learning import QLearningAgent
from carbon_capture.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FanCommand:
    """The output of the RL controller — a single fan speed command.

    This is a COMMAND. The actual resulting fan speed is observed in the
    NEXT telemetry record, not here. Commanded ≠ observed.
    """
    delta_rpm: float          # Adjustment: -100, 0, or +100 RPM
    recommended_fan_rpm: float  # Clipped to [min_rpm, max_rpm]
    action_idx: int           # Internal Q-table action index
    current_fan_rpm_observed: float   # The observed Fan_Speed_RPM from telemetry
    is_bounded: bool          # True if recommended_fan_rpm was clipped


def telemetry_to_process_state(record: DigitalTwinTelemetryRecord) -> Optional[ProcessState]:
    """
    Derive a ProcessState from an observed DigitalTwinTelemetryRecord.

    Only uses the fields that are DIRECTLY OBSERVED from telemetry.
    Returns None if any required field is a dropout (None).

    State vector (matches QLearningAgent.discretize_state):
        co2_concentration_pct  ← derived from CO2_ppm: ppm / 10000 = %
        temperature_c          ← Temperature_C directly
        fan_speed_rpm          ← Fan_Speed_RPM directly
        gas_flow_m3_s          ← derived from Gas_Flow_L_min: L/min / 1000 * 60 = m³/s
                                  (1 L/min = 1/1000 m³/min = 1/60000 m³/s)
                                  so m³/s = L/min / 60000

    Derivation notes:
    - CO2_ppm → co2_concentration_pct: 1 ppm = 0.0001%, so divide by 10000.
    - Gas_Flow_L_min → m³/s: 1 L/min = 1.6667e-5 m³/s.
    These are exact unit conversions, not approximations or empirical estimates.
    """
    required = {
        "CO2_ppm": record.CO2_ppm,
        "Temperature_C": record.Temperature_C,
        "Fan_Speed_RPM": record.Fan_Speed_RPM,
        "Gas_Flow_L_min": record.Gas_Flow_L_min,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        logger.warning(
            f"[RL] Cannot derive ProcessState: dropout fields = {missing}. "
            "No fan command will be issued."
        )
        return None

    co2_pct = record.CO2_ppm / 10000.0         # ppm → %
    gas_flow_m3_s = record.Gas_Flow_L_min / 60000.0  # L/min → m³/s

    return ProcessState(
        co2_concentration_pct=co2_pct,
        temperature_c=record.Temperature_C,
        fan_speed_rpm=record.Fan_Speed_RPM,
        gas_flow_m3_s=gas_flow_m3_s,
    )


class SerialRLController:
    """
    Source-agnostic RL process controller.

    Accepts telemetry records (from any source: Digital Twin, Arduino, CSV replay).
    Produces fan speed commands.
    Does NOT touch E-PINN, F_valid, or carbon credits.
    """

    def __init__(
        self,
        agent: QLearningAgent,
        env: ScrubberEnvironment,
        training: bool = False,
    ):
        self.agent = agent
        self.env = env
        self.training = training

    def step(self, record: DigitalTwinTelemetryRecord) -> Optional[FanCommand]:
        """
        Derive a fan command from one telemetry record.

        Returns None if the telemetry has dropout fields required for state derivation.
        The caller must handle None gracefully (e.g. hold last command or skip).
        """
        state = telemetry_to_process_state(record)
        if state is None:
            return None

        action_idx = self.agent.select_action(state, training=self.training)
        deltas = [-100.0, 0.0, 100.0]
        delta = deltas[action_idx]

        raw_target = state.fan_speed_rpm + delta
        clipped = float(
            max(self.env.min_rpm, min(self.env.max_rpm, raw_target))
        )
        bounded = abs(clipped - raw_target) > 1e-3

        cmd = FanCommand(
            delta_rpm=delta,
            recommended_fan_rpm=clipped,
            action_idx=action_idx,
            current_fan_rpm_observed=state.fan_speed_rpm,
            is_bounded=bounded,
        )

        logger.debug(
            f"[RL] Observed RPM={state.fan_speed_rpm:.1f} | "
            f"Action={delta:+.0f} RPM | Command={clipped:.1f} RPM | "
            f"Bounded={bounded}"
        )
        return cmd

    def update(
        self,
        prev_record: DigitalTwinTelemetryRecord,
        action_idx: int,
        reward: float,
        next_record: DigitalTwinTelemetryRecord,
        done: bool = False,
    ) -> Optional[float]:
        """Update Q-table from a (state, action, reward, next_state) transition."""
        s0 = telemetry_to_process_state(prev_record)
        s1 = telemetry_to_process_state(next_record)
        if s0 is None or s1 is None:
            return None
        td = self.agent.update(s0, action_idx, reward, s1, done)
        return td
