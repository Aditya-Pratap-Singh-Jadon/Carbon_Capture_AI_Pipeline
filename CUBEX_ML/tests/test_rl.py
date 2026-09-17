"""Unit tests for Q-Learning Process Controller and Environment."""

import pytest
from carbon_capture.rl.environment import ScrubberEnvironment, ProcessState
from carbon_capture.rl.q_learning import QLearningAgent
from carbon_capture.rl.controller import RLProcessController

def test_environment_fan_limits():
    env = ScrubberEnvironment(min_rpm=1000.0, max_rpm=2000.0)
    state = env.reset()

    # Step down repeatedly
    for _ in range(20):
        state, reward, done, info = env.step(0)  # -100 RPM
    assert state.fan_speed_rpm >= 1000.0

    # Step up repeatedly
    for _ in range(20):
        state, reward, done, info = env.step(2)  # +100 RPM
    assert state.fan_speed_rpm <= 2000.0

def test_q_learning_agent_update():
    agent = QLearningAgent(epsilon=0.0)  # Pure exploitation
    s1 = ProcessState(12.0, 45.0, 1500.0, 35.0)
    s2 = ProcessState(10.0, 45.0, 1600.0, 38.0)

    action = agent.select_action(s1, training=False)
    td_error = agent.update(s1, action, reward=5.0, next_state=s2, done=False)

    disc_s1 = agent.discretize_state(s1)
    assert agent.q_table[disc_s1][action] != 0.0
