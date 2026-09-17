"""CLI script to train Q-learning process control agent."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from carbon_capture.rl.environment import ScrubberEnvironment
from carbon_capture.rl.q_learning import QLearningAgent
from carbon_capture.rl.controller import RLProcessController
from carbon_capture.utils.logging import setup_logging, get_logger

setup_logging("INFO")
logger = get_logger("train_rl")

def main():
    env = ScrubberEnvironment()
    agent = QLearningAgent()
    controller = RLProcessController(agent, env)

    save_dir = Path("models/rl")
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training Q-learning process controller...")
    controller.train(episodes=150, steps_per_episode=40)

    model_path = save_dir / "q_table.pkl"
    agent.save(model_path)
    logger.info(f"Saved Q-learning agent model to {model_path}")

    # Test recommendation
    test_state = env.reset()
    rec = controller.get_control_recommendation(test_state)
    logger.info(f"Sample state recommendation: {rec}")

if __name__ == "__main__":
    main()
