"""E-PINN Training Pipeline with Leakage-Free Normalization and Checkpointing."""

from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import torch
from carbon_capture.input.loader import DataLoader
from carbon_capture.preprocessing.cleaning import DataCleaner
from carbon_capture.preprocessing.scaling import CanonicalScaler
from carbon_capture.epinn.model import EPINNModel
from carbon_capture.epinn.trainer import EPINNTrainer
from carbon_capture.utils.logging import get_logger
from carbon_capture.utils.serialization import save_artifact
from carbon_capture.utils.reproducibility import set_seed

logger = get_logger(__name__)

class TrainingPipeline:
    """Orchestrates end-to-end training of the E-PINN verification model."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        set_seed(self.config.get("random_seed", 42))

    def run(
        self,
        train_path: Path,
        val_path: Path,
        model_save_dir: Path,
        scaler_save_dir: Path,
    ) -> Dict[str, Any]:
        """Execute complete training workflow."""
        loader = DataLoader()
        cleaner = DataCleaner()

        logger.info(f"Loading training data from {train_path}...")
        train_df, train_val_result = loader.load_from_csv(train_path)
        if not train_val_result.valid:
            raise ValueError(f"Training data validation failed: {train_val_result.errors}")

        logger.info(f"Loading validation data from {val_path}...")
        val_df, val_val_result = loader.load_from_csv(val_path)
        if not val_val_result.valid:
            raise ValueError(f"Validation data validation failed: {val_val_result.errors}")

        # Data hygiene
        cleaner.fit(train_df)
        train_clean, _ = cleaner.clean(train_df)
        val_clean, _ = cleaner.clean(val_df)

        # Leakage-free feature scaling (fitted STRICTLY on training data)
        scaler = CanonicalScaler()
        scaler.fit(train_clean)
        char_scales = scaler.get_characteristic_scales()

        # Build E-PINN model
        pe_aux_mode = self.config.get("physics", {}).get("pe_aux_mode", "verbatim")
        arch_cfg = self.config.get("model", {}).get("architecture", {})
        train_cfg = self.config.get("model", {}).get("training", {})

        input_dim = len(scaler.feature_names)
        model = EPINNModel(
            input_dim=input_dim,
            hidden_dims=arch_cfg.get("hidden_dims", [128, 128, 64]),
            characteristic_scales=char_scales,
            pe_aux_mode=pe_aux_mode,
            activation=arch_cfg.get("activation", "silu"),
            dropout=arch_cfg.get("dropout", 0.05),
        )

        # Trainer
        trainer = EPINNTrainer(
            model=model,
            lr=train_cfg.get("learning_rate", 0.001),
            weight_decay=train_cfg.get("weight_decay", 1e-5),
            grad_clip=train_cfg.get("grad_clip_norm", 1.0),
        )

        # Train targets (if available)
        from carbon_capture.input.synthetic import SyntheticDataGenerator
        gen = SyntheticDataGenerator()
        train_targets = gen.compute_ground_truth_targets(train_clean, pe_aux_mode=pe_aux_mode)
        val_targets = gen.compute_ground_truth_targets(val_clean, pe_aux_mode=pe_aux_mode)

        checkpoint_dir = Path(model_save_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        history = trainer.fit(
            train_df=train_clean,
            val_df=val_clean,
            scaler=scaler,
            train_targets=train_targets,
            val_targets=val_targets,
            epochs=train_cfg.get("epochs", 50),
            batch_size=train_cfg.get("batch_size", 32),
            early_stopping_patience=train_cfg.get("early_stopping_patience", 15),
            checkpoint_dir=checkpoint_dir,
        )

        # Persist model and scaler artifacts
        scaler_dir = Path(scaler_save_dir)
        scaler_dir.mkdir(parents=True, exist_ok=True)
        save_artifact(scaler.to_dict(), scaler_dir / "canonical_scaler.json", artifact_type="json")
        
        model_metadata = {
            "input_dim": input_dim,
            "hidden_dims": arch_cfg.get("hidden_dims", [128, 128, 64]),
            "activation": arch_cfg.get("activation", "silu"),
            "dropout": arch_cfg.get("dropout", 0.05),
            "pe_aux_mode": pe_aux_mode,
        }
        save_artifact(model_metadata, checkpoint_dir / "model_config.json", artifact_type="json")
        save_artifact(model.state_dict(), checkpoint_dir / "epinn_final.pt", artifact_type="torch")

        logger.info(f"Saved trained E-PINN and scaler to {checkpoint_dir} and {scaler_dir}")
        return {
            "model": model,
            "scaler": scaler,
            "history": history,
            "characteristic_scales": char_scales,
        }
