"""E-PINN Inference and Verification Evaluator."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch
from carbon_capture.input.canonical_order import CANONICAL_INPUT_ORDER
from carbon_capture.preprocessing.scaling import CanonicalScaler
from carbon_capture.epinn.model import EPINNModel
from carbon_capture.physics.residuals import compute_raw_residuals, normalize_residuals
from carbon_capture.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class EPINNInferenceOutput:
    predictions: Dict[str, np.ndarray]
    raw_residuals: Dict[str, np.ndarray]
    normalized_residuals: Dict[str, np.ndarray]
    residual_scales: Dict[str, float]
    constraint_violations: Dict[str, float]
    is_physics_consistent: bool
    mean_normalized_residual: float
    max_normalized_residual: float

class EPINNInferenceEngine:
    """Orchestrates E-PINN inference, physics checks, and residual evaluation."""

    def __init__(
        self,
        model: EPINNModel,
        scaler: CanonicalScaler,
        device: Optional[torch.device] = None,
        max_residual_tol: float = 3.0,
        mean_residual_tol: float = 1.5,
    ):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.scaler = scaler
        self.max_residual_tol = max_residual_tol
        self.mean_residual_tol = mean_residual_tol

    def evaluate(self, df: pd.DataFrame) -> EPINNInferenceOutput:
        """Run complete E-PINN inference and residual evaluation on preprocessed canonical DataFrame."""
        # 1. Forward model pass
        preds = self.model.predict_from_dataframe(df, self.scaler, device=self.device)
        return self.evaluate_from_preds(preds, df)

    def evaluate_from_preds(self, preds: Dict[str, np.ndarray], df: pd.DataFrame) -> EPINNInferenceOutput:
        """Compute residuals and consistency metrics from pre-calculated predictions."""
        # 2. Raw inputs dictionary
        raw_inputs = {col: df[col].to_numpy(dtype=np.float32) for col in CANONICAL_INPUT_ORDER}

        # 3. Compute raw residuals
        raw_residuals = compute_raw_residuals(preds, raw_inputs, pe_aux_mode=self.model.pe_aux_mode)

        # 4. Normalize residuals
        char_scales = self.scaler.get_characteristic_scales()
        norm_residuals, scales_used = normalize_residuals(raw_residuals, preds, char_scales)

        # 5. Evaluate constraints
        with torch.no_grad():
            preds_t = {k: torch.tensor(v, device=self.device) for k, v in preds.items()}
            inputs_t = {k: torch.tensor(v, device=self.device) for k, v in raw_inputs.items()}
            g_dict = self.model.lagrangian_manager.compute_constraint_functions(preds_t, inputs_t)
            violations = {
                name: float(torch.relu(g).mean().item()) for name, g in g_dict.items()
            }

        # 6. Aggregated residual metrics
        mean_abs_norm = [float(np.mean(np.abs(r))) for r in norm_residuals.values()]
        mean_norm_res = float(np.mean(mean_abs_norm))
        max_norm_res = float(np.max(mean_abs_norm))

        max_constraint_viol = max(violations.values()) if violations else 0.0

        is_consistent = (
            max_norm_res <= self.max_residual_tol
            and mean_norm_res <= self.mean_residual_tol
            and max_constraint_viol <= 1e-3
        )

        return EPINNInferenceOutput(
            predictions=preds,
            raw_residuals=raw_residuals,
            normalized_residuals=norm_residuals,
            residual_scales=scales_used,
            constraint_violations=violations,
            is_physics_consistent=is_consistent,
            mean_normalized_residual=mean_norm_res,
            max_normalized_residual=max_norm_res,
        )
