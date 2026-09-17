"""E-PINN Trainer with Dual-Ascent Augmented Lagrangian Updates."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from carbon_capture.input.canonical_order import CANONICAL_INPUT_ORDER
from carbon_capture.preprocessing.scaling import CanonicalScaler
from carbon_capture.epinn.model import EPINNModel
from carbon_capture.utils.logging import get_logger

logger = get_logger(__name__)

class EPINNTrainer:
    """Trains E-PINN with physics loss and augmented Lagrangian constraint updates."""

    def __init__(
        self,
        model: EPINNModel,
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.grad_clip = grad_clip
        self.history: List[Dict[str, float]] = []

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        scaler: CanonicalScaler,
        train_targets: Optional[pd.DataFrame] = None,
        val_targets: Optional[pd.DataFrame] = None,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 15,
        checkpoint_dir: Optional[Path] = None,
    ) -> Dict[str, list]:
        """Execute complete training loop with dual ascent updates."""
        # Convert inputs to scaled tensors
        x_train_scaled = scaler.transform(train_df)
        x_val_scaled = scaler.transform(val_df)

        # Retain raw unscaled inputs for physics equations
        x_train_raw = train_df[CANONICAL_INPUT_ORDER].to_numpy(dtype=np.float32)
        x_val_raw = val_df[CANONICAL_INPUT_ORDER].to_numpy(dtype=np.float32)

        # Prepare target tensors if available
        train_tensors = [torch.tensor(x_train_scaled), torch.tensor(x_train_raw)]
        val_tensors = [torch.tensor(x_val_scaled), torch.tensor(x_val_raw)]

        if train_targets is not None:
            y_train_arr = train_targets.to_numpy(dtype=np.float32)
            train_tensors.append(torch.tensor(y_train_arr))

        if val_targets is not None:
            y_val_arr = val_targets.to_numpy(dtype=np.float32)
            val_tensors.append(torch.tensor(y_val_arr))

        train_dataset = TensorDataset(*train_tensors)
        val_dataset = TensorDataset(*val_tensors)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(f"Starting E-PINN training on {self.device} for {epochs} epochs...")

        history_dict: Dict[str, list] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "l_physics": [],
            "l_constraints": [],
            "l_cc": [],
        }

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            epoch_breakdown = {}
            total_batches = len(train_loader)

            for batch in train_loader:
                x_scaled = batch[0].to(self.device)
                x_raw = batch[1].to(self.device)

                # Map raw input variables to dictionary
                raw_inputs_dict = {
                    name: x_raw[:, idx] for idx, name in enumerate(CANONICAL_INPUT_ORDER)
                }

                targets_dict = None
                if len(batch) > 2 and train_targets is not None:
                    y_batch = batch[2].to(self.device)
                    targets_dict = {
                        col: y_batch[:, idx] for idx, col in enumerate(train_targets.columns)
                    }

                self.optimizer.zero_grad()
                preds = self.model(x_scaled)

                loss, breakdown = self.model.loss_fn(
                    preds=preds,
                    inputs=raw_inputs_dict,
                    targets=targets_dict,
                    lagrangian_manager=self.model.lagrangian_manager,
                )

                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                epoch_loss += loss.item()
                for k, v in breakdown.items():
                    val = v.item() if isinstance(v, torch.Tensor) else float(v)
                    epoch_breakdown[k] = epoch_breakdown.get(k, 0.0) + val

            avg_train_loss = epoch_loss / total_batches

            # Validation pass
            self.model.eval()
            val_loss = 0.0
            val_batches = len(val_loader)
            all_violations: Dict[str, List[float]] = {}

            with torch.no_grad():
                for batch in val_loader:
                    x_s = batch[0].to(self.device)
                    x_r = batch[1].to(self.device)
                    raw_in = {name: x_r[:, idx] for idx, name in enumerate(CANONICAL_INPUT_ORDER)}

                    targets_d = None
                    if len(batch) > 2 and val_targets is not None:
                        y_b = batch[2].to(self.device)
                        targets_d = {col: y_b[:, idx] for idx, col in enumerate(val_targets.columns)}

                    preds = self.model(x_s)
                    v_loss, v_breakdown = self.model.loss_fn(
                        preds=preds,
                        inputs=raw_in,
                        targets=targets_d,
                        lagrangian_manager=self.model.lagrangian_manager,
                    )
                    val_loss += v_loss.item()

                    # Collect constraint violations
                    g_dict = self.model.lagrangian_manager.compute_constraint_functions(preds, raw_in)
                    for c_name, g_val in g_dict.items():
                        v = float(torch.relu(g_val).mean().item())
                        if c_name not in all_violations:
                            all_violations[c_name] = []
                        all_violations[c_name].append(v)

            avg_val_loss = val_loss / max(val_batches, 1)

            # Dual ascent update at the end of each epoch (Phase 7.4)
            mean_violations = {k: float(np.mean(v)) for k, v in all_violations.items()}
            constraint_statuses = self.model.lagrangian_manager.update_dual_variables(mean_violations)

            history_dict["epoch"].append(epoch)
            history_dict["train_loss"].append(avg_train_loss)
            history_dict["val_loss"].append(avg_val_loss)
            history_dict["l_physics"].append(epoch_breakdown.get("l_physics", 0.0) / total_batches)
            history_dict["l_constraints"].append(epoch_breakdown.get("l_constraints", 0.0) / total_batches)
            history_dict["l_cc"].append(epoch_breakdown.get("l_cc", 0.0) / total_batches)

            if epoch % 5 == 0 or epoch == 1:
                max_viol = max([s.violation for s in constraint_statuses]) if constraint_statuses else 0.0
                logger.info(
                    f"Epoch {epoch:3d}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} | Max Constraint Viol: {max_viol:.4e}"
                )

            # Checkpointing and Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                if checkpoint_dir:
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), checkpoint_dir / "best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break

        return history_dict
