"""Phase 9: Uncertainty Integration into the Validation Architecture.

Method: Monte-Carlo Dropout (already implemented in uncertainty.py).
MC-Dropout provides epistemic uncertainty by treating dropout as a Bayesian approximation.

Scientific defensibility assessment:
- MC-Dropout is an established approximation to Bayesian deep learning (Gal & Ghahramani 2016).
- It quantifies epistemic (model) uncertainty, NOT aleatoric (data) uncertainty.
- The normalized dispersion metric (1 / (1 + mean_std/mean_mag)) produces a
  confidence score in [0, 1] where 1 = maximally confident prediction.

Integration into F_valid:
- F_valid CANNOT be automatically modified by uncertainty without a scientifically
  justified threshold. The project specification does not define one.
- Therefore: uncertainty is integrated as a DIAGNOSTIC output alongside F_valid,
  NOT as a multiplier on F_valid.
- The uncertainty_flag field is set to 'HIGH_UNCERTAINTY' when any output's
  confidence score falls below 0.5 (a conservative but interpretable threshold
  derived from the normalized dispersion exceeding 1.0 — i.e. std > mean).
- This flag is logged and included in the audit report but does NOT override F_valid.

This is explicitly documented as a known limitation in KNOWN_LIMITATIONS.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from carbon_capture.epinn.model import EPINNModel
from carbon_capture.epinn.uncertainty import UncertaintyEstimator, UncertaintyResult
from carbon_capture.epinn.inference import EPINNInferenceOutput
from carbon_capture.preprocessing.scaling import CanonicalScaler
from carbon_capture.utils.logging import get_logger

logger = get_logger(__name__)

# Threshold for flagging HIGH_UNCERTAINTY.
# Scientific basis: normalized dispersion > 1.0 means std exceeds mean magnitude.
# Mapped to confidence_score < 0.5 via: score = 1 / (1 + dispersion).
_UNCERTAINTY_FLAG_THRESHOLD = 0.5


@dataclass
class UncertaintyAugmentedResult:
    """Wraps an EPINNInferenceOutput with uncertainty diagnostics.

    F_valid semantics are NOT changed.
    uncertainty_flag is a DIAGNOSTIC only.
    """
    inference_output: EPINNInferenceOutput
    uncertainty: Optional[UncertaintyResult]
    uncertainty_flag: str            # 'HIGH_UNCERTAINTY', 'LOW_UNCERTAINTY', or 'NOT_COMPUTED'
    low_confidence_outputs: list     # output names with confidence < threshold
    diagnostic_note: str


class UncertaintyIntegrator:
    """Integrates MC-Dropout uncertainty estimates into the validation pipeline as diagnostics."""

    def __init__(self, n_samples: int = 20):
        self.estimator = UncertaintyEstimator(n_samples=n_samples)

    def augment(
        self,
        inference_output: EPINNInferenceOutput,
        model: EPINNModel,
        df: pd.DataFrame,
        scaler: CanonicalScaler,
        device: torch.device = torch.device("cpu"),
        run_uncertainty: bool = True,
    ) -> UncertaintyAugmentedResult:
        """
        Compute MC-Dropout uncertainty and augment the inference result diagnostically.
        F_valid is preserved from inference_output unchanged.
        """
        if not run_uncertainty:
            return UncertaintyAugmentedResult(
                inference_output=inference_output,
                uncertainty=None,
                uncertainty_flag="NOT_COMPUTED",
                low_confidence_outputs=[],
                diagnostic_note="Uncertainty computation skipped (run_uncertainty=False).",
            )

        try:
            unc = self.estimator.estimate_mc_dropout(model, df, scaler, device)
        except Exception as e:
            logger.warning(f"[Uncertainty] MC-Dropout failed: {e}")
            return UncertaintyAugmentedResult(
                inference_output=inference_output,
                uncertainty=None,
                uncertainty_flag="NOT_COMPUTED",
                low_confidence_outputs=[],
                diagnostic_note=f"Uncertainty computation failed: {e}",
            )

        # Identify outputs with confidence below threshold
        low_conf = [
            name for name, score in (unc.confidence_scores or {}).items()
            if score < _UNCERTAINTY_FLAG_THRESHOLD
        ]
        flag = "HIGH_UNCERTAINTY" if low_conf else "LOW_UNCERTAINTY"

        if low_conf:
            logger.warning(
                f"[Uncertainty] HIGH_UNCERTAINTY flag on outputs: {low_conf}. "
                "F_valid is NOT modified — this is a diagnostic."
            )
        else:
            logger.info("[Uncertainty] LOW_UNCERTAINTY — prediction spread within acceptable range.")

        note = (
            f"MC-Dropout ({self.estimator.n_samples} samples). "
            f"F_valid is unchanged at {inference_output.is_physics_consistent}. "
            f"Flag: {flag}. "
            "Uncertainty does not override F_valid per scientific specification."
        )

        return UncertaintyAugmentedResult(
            inference_output=inference_output,
            uncertainty=unc,
            uncertainty_flag=flag,
            low_confidence_outputs=low_conf,
            diagnostic_note=note,
        )
