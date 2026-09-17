"""CLI script to train E-PINN verification model."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml
from carbon_capture.pipeline.training_pipeline import TrainingPipeline
from carbon_capture.utils.logging import setup_logging, get_logger

setup_logging("INFO")
logger = get_logger("train_epinn")

def main():
    config_path = Path("configs/config.yaml")
    model_config_path = Path("configs/model.yaml")
    physics_config_path = Path("configs/physics.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(model_config_path, "r", encoding="utf-8") as f:
        cfg["model"] = yaml.safe_load(f)["model"]
    with open(physics_config_path, "r", encoding="utf-8") as f:
        cfg["physics"] = yaml.safe_load(f)["physics"]

    train_csv = Path("data/synthetic/train.csv")
    val_csv = Path("data/synthetic/val.csv")

    if not train_csv.exists() or not val_csv.exists():
        logger.error("Synthetic datasets not found. Run python scripts/generate_data.py first.")
        sys.exit(1)

    pipeline = TrainingPipeline(cfg)
    logger.info("Executing E-PINN training pipeline...")
    pipeline.run(
        train_path=train_csv,
        val_path=val_csv,
        model_save_dir=Path("models/epinn"),
        scaler_save_dir=Path("models/scalers"),
    )
    logger.info("E-PINN training finished successfully.")

if __name__ == "__main__":
    main()
