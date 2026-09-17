"""E-PINN High-Level Model Container and Orchestrator."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from carbon_capture.input.canonical_order import CANONICAL_INPUT_ORDER
from carbon_capture.preprocessing.scaling import CanonicalScaler
from carbon_capture.epinn.architecture import EPINNMultiHead, PHYSICAL_OUTPUT_NAMES
from carbon_capture.epinn.losses import EPINNLoss
from carbon_capture.epinn.constraints import AugmentedLagrangianManager
from carbon_capture.utils.logging import get_logger

logger = get_logger(__name__)

class EPINNModel(nn.Module):
    """Unified Extended Physics-Informed Neural Network Model."""

    def __init__(
        self,
        input_dim: int = 26,
        hidden_dims: Optional[List[int]] = None,
        characteristic_scales: Optional[Dict[str, float]] = None,
        pe_aux_mode: str = "verbatim",
        activation: str = "silu",
        dropout: float = 0.05,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.network = EPINNMultiHead(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
        )
        self.loss_fn = EPINNLoss(
            characteristic_scales=characteristic_scales,
            pe_aux_mode=pe_aux_mode,
        )
        self.lagrangian_manager = AugmentedLagrangianManager()
        self.pe_aux_mode = pe_aux_mode

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.network(x)

    def predict_from_dataframe(
        self,
        df: pd.DataFrame,
        scaler: CanonicalScaler,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, np.ndarray]:
        """Run forward inference on preprocessed canonical DataFrame."""
        self.eval()
        x_scaled = scaler.transform(df)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=device)

        with torch.no_grad():
            preds_tensor = self.forward(x_tensor)

        preds_numpy = {
            name: tensor.cpu().numpy() for name, tensor in preds_tensor.items()
        }
        return preds_numpy
