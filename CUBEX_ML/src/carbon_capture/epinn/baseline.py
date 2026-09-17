"""Phase 5: Standalone Non-Physics Baseline Model.

A pure MLP trained with data loss only — NO physics residuals, NO Augmented Lagrangian.
Purpose: scientific comparison vs E-PINN to demonstrate the contribution of physics constraints.
This model is NOT a verification authority. Its sole role is comparative benchmarking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from carbon_capture.epinn.architecture import PHYSICAL_OUTPUT_NAMES as EPINN_OUTPUT_NAMES
from carbon_capture.preprocessing.scaling import CanonicalScaler
from carbon_capture.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BaselineMLP(nn.Module):
    """Pure data-driven MLP with no physics constraints or physics residuals."""

    def __init__(self, input_dim: int = 26, hidden_dims: List[int] = None):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64, 32]
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, len(EPINN_OUTPUT_NAMES)))
        self.net = nn.Sequential(*layers)
        self.output_names = EPINN_OUTPUT_NAMES

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.net(x)
        return {name: raw[:, i] for i, name in enumerate(self.output_names)}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@dataclass
class BaselineTrainingResult:
    train_losses: List[float]
    val_losses: List[float]
    best_val_loss: float
    epochs_trained: int
    training_time_s: float


class BaselineTrainer:
    """Trains the baseline MLP with MSE data loss only — no physics."""

    def __init__(
        self,
        model: BaselineMLP,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def train(
        self,
        X_train: np.ndarray,
        Y_train: Dict[str, np.ndarray],
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[Dict[str, np.ndarray]] = None,
        epochs: int = 100,
        batch_size: int = 64,
    ) -> BaselineTrainingResult:
        x_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        # Stack all targets
        y_keys = list(Y_train.keys())
        y_stack = np.stack([Y_train[k] for k in y_keys], axis=1)
        y_t = torch.tensor(y_stack, dtype=torch.float32, device=self.device)

        ds = TensorDataset(x_t, y_t)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        x_val_t = y_val_t = None
        if X_val is not None and Y_val is not None:
            x_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
            y_val_stack = np.stack([Y_val[k] for k in y_keys], axis=1)
            y_val_t = torch.tensor(y_val_stack, dtype=torch.float32, device=self.device)

        train_losses, val_losses = [], []
        best_val = float("inf")
        t0 = time.time()

        for epoch in range(epochs):
            self.model.train()
            ep_loss = 0.0
            for xb, yb in dl:
                self.optimizer.zero_grad()
                out = self.model(xb)
                preds_stack = torch.stack([out[k] for k in y_keys], dim=1)
                loss = nn.functional.mse_loss(preds_stack, yb)
                loss.backward()
                self.optimizer.step()
                ep_loss += loss.item()
            avg_ep = ep_loss / len(dl)
            train_losses.append(avg_ep)

            if x_val_t is not None:
                self.model.eval()
                with torch.no_grad():
                    out_v = self.model(x_val_t)
                    ps_v = torch.stack([out_v[k] for k in y_keys], dim=1)
                    vl = nn.functional.mse_loss(ps_v, y_val_t).item()
                val_losses.append(vl)
                if vl < best_val:
                    best_val = vl

            if epoch % 20 == 0:
                logger.info(f"[Baseline] Epoch {epoch}/{epochs} | train_loss={avg_ep:.6f}")

        return BaselineTrainingResult(
            train_losses=train_losses,
            val_losses=val_losses,
            best_val_loss=best_val if val_losses else float("nan"),
            epochs_trained=epochs,
            training_time_s=time.time() - t0,
        )


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

@dataclass
class BaselineComparisonReport:
    baseline_test_mse: Dict[str, float]
    epinn_test_mse: Dict[str, float]
    baseline_better_count: int
    epinn_better_count: int
    summary: str

    def print_table(self) -> None:
        print("\n=== BASELINE vs E-PINN COMPARISON ===")
        print(f"{'Output':<25} {'Baseline MSE':>14} {'E-PINN MSE':>14} {'Winner':>10}")
        print("-" * 68)
        for k in self.baseline_test_mse:
            bm = self.baseline_test_mse[k]
            em = self.epinn_test_mse.get(k, float("nan"))
            winner = "E-PINN" if em < bm else "Baseline"
            print(f"{k:<25} {bm:>14.6f} {em:>14.6f} {winner:>10}")
        print("-" * 68)
        print(self.summary)


def compare_baseline_vs_epinn(
    baseline_model: BaselineMLP,
    epinn_predict_fn,   # callable: df -> dict[str, np.ndarray]
    X_test: np.ndarray,
    Y_test: Dict[str, np.ndarray],
    scaler: CanonicalScaler,
    device: torch.device = torch.device("cpu"),
) -> BaselineComparisonReport:
    """Compute per-output MSE for both models on the held-out test set."""
    baseline_model.eval()
    x_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        baseline_out = baseline_model(x_t)
    bm_preds = {k: v.cpu().numpy() for k, v in baseline_out.items()}

    epinn_preds = epinn_predict_fn(X_test)

    baseline_mse, epinn_mse = {}, {}
    for k in Y_test:
        if k in bm_preds:
            baseline_mse[k] = float(np.mean((bm_preds[k] - Y_test[k]) ** 2))
        if k in epinn_preds:
            epinn_mse[k] = float(np.mean((epinn_preds[k] - Y_test[k]) ** 2))

    bl_better = sum(1 for k in baseline_mse if baseline_mse[k] < epinn_mse.get(k, float("inf")))
    ep_better = sum(1 for k in baseline_mse if epinn_mse.get(k, float("inf")) < baseline_mse[k])

    summary = (
        f"E-PINN wins on {ep_better}/{len(baseline_mse)} outputs. "
        f"Baseline wins on {bl_better}/{len(baseline_mse)} outputs. "
        "Note: E-PINN advantage includes physics constraint enforcement; "
        "MSE alone does not capture physics consistency."
    )

    return BaselineComparisonReport(
        baseline_test_mse=baseline_mse,
        epinn_test_mse=epinn_mse,
        baseline_better_count=bl_better,
        epinn_better_count=ep_better,
        summary=summary,
    )
