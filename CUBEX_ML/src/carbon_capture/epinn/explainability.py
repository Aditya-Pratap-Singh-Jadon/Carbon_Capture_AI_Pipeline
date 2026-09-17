"""Phase 10: Model Explainability & Feature Attribution.

Method: Gradient-based input feature importance (Vanilla Gradients / Saliency).
This approach is model-native (no external library required) and scientifically
appropriate for a physics-constrained neural network where the input–output mapping
is not a black box — the physics residuals already provide interpretable intermediate
quantities.

Approach justification:
- SHAP (KernelSHAP) requires repeated model evaluations and is not guaranteed to
  respect physics constraints; it could attribute importance to correlations that
  violate the governing equations.
- LIME makes local linear approximations that can contradict the physics structure.
- Vanilla Gradients (∂output/∂input) measure the local sensitivity of each output
  to each input variable. This is consistent with the physics Jacobian interpretation.

Explainability is OBSERVATIONAL ONLY:
- It does not modify predictions.
- It does not affect F_valid.
- It does not alter residuals.
- It is purely diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from carbon_capture.epinn.architecture import PHYSICAL_OUTPUT_NAMES
from carbon_capture.epinn.model import EPINNModel
from carbon_capture.input.canonical_order import CORE_PHYSICAL_INPUT_ORDER
from carbon_capture.preprocessing.scaling import CanonicalScaler
from carbon_capture.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureImportanceResult:
    """Per-output gradient-based feature importance."""
    output_name: str
    feature_names: List[str]
    mean_abs_gradient: np.ndarray        # shape: (n_features,)
    top_k_features: List[str]            # top-k by mean absolute gradient
    top_k_importances: List[float]
    method: str = "vanilla_gradients"
    diagnostic_note: str = ""


@dataclass
class ExplainabilityReport:
    """Aggregated explainability for all outputs, for a batch of inputs."""
    importances: Dict[str, FeatureImportanceResult] = field(default_factory=dict)
    global_feature_rank: List[str] = field(default_factory=list)  # averaged across outputs
    method: str = "vanilla_gradients"
    n_samples_analysed: int = 0
    note: str = (
        "Observational/diagnostic only. Does not modify predictions, F_valid, "
        "residuals, or constraints."
    )

    def print_summary(self, top_k: int = 5) -> None:
        print(f"\n=== EXPLAINABILITY REPORT [{self.method}] ===")
        print(f"Samples analysed: {self.n_samples_analysed}")
        print(f"Global top-{top_k} features (averaged across all outputs):")
        for i, fn in enumerate(self.global_feature_rank[:top_k], 1):
            print(f"  {i}. {fn}")
        print()
        for out_name, imp in self.importances.items():
            print(f"  {out_name}: top features = {imp.top_k_features[:top_k]}")
        print(f"\nNote: {self.note}\n")


class GradientExplainer:
    """
    Computes vanilla gradient (saliency) feature importance for the E-PINN.
    Pure gradient computation; no model modifications.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def explain(
        self,
        model: EPINNModel,
        df: pd.DataFrame,
        scaler: CanonicalScaler,
        device: torch.device = torch.device("cpu"),
        output_names: Optional[List[str]] = None,
    ) -> ExplainabilityReport:
        """
        Compute ∂output/∂input gradients for each output across the batch.
        Gradients are taken w.r.t. the raw (un-scaled) input representation,
        then averaged over the batch using mean absolute value.
        """
        if output_names is None:
            output_names = PHYSICAL_OUTPUT_NAMES

        feature_names = CORE_PHYSICAL_INPUT_ORDER  # 26 physical features

        x_scaled = scaler.transform(df)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=device, requires_grad=True)

        model.eval()
        out_dict = model(x_tensor)

        report = ExplainabilityReport(
            method="vanilla_gradients",
            n_samples_analysed=len(df),
        )

        all_mean_grads = []

        for out_name in output_names:
            if out_name not in out_dict:
                continue

            # Zero accumulated gradients
            if x_tensor.grad is not None:
                x_tensor.grad.zero_()

            # Sum output to get a scalar for backward
            scalar = out_dict[out_name].sum()
            scalar.backward(retain_graph=True)

            if x_tensor.grad is None:
                continue

            # Mean absolute gradient across batch, shape: (n_features,)
            mag = x_tensor.grad.abs().mean(dim=0).cpu().detach().numpy()
            # Only report on the first 26 features (physical input layer)
            mag_phys = mag[:len(feature_names)]

            top_idx = np.argsort(mag_phys)[::-1][: self.top_k]
            top_feats = [feature_names[i] for i in top_idx]
            top_vals = [float(mag_phys[i]) for i in top_idx]

            result = FeatureImportanceResult(
                output_name=out_name,
                feature_names=feature_names,
                mean_abs_gradient=mag_phys,
                top_k_features=top_feats,
                top_k_importances=top_vals,
                diagnostic_note=(
                    "Vanilla gradient saliency. Observational only. "
                    "Does not affect model predictions or F_valid."
                ),
            )
            report.importances[out_name] = result
            all_mean_grads.append(mag_phys)

        # Global feature rank: average importance across all outputs
        if all_mean_grads:
            global_mean = np.mean(np.stack(all_mean_grads, axis=0), axis=0)
            global_rank_idx = np.argsort(global_mean)[::-1]
            report.global_feature_rank = [feature_names[i] for i in global_rank_idx]

        logger.info(
            f"[Explainability] Gradient attribution complete. "
            f"Top global feature: {report.global_feature_rank[0] if report.global_feature_rank else 'N/A'}"
        )
        return report
