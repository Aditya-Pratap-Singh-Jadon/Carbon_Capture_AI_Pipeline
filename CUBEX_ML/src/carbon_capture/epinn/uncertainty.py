"""Uncertainty Estimation and Propagation (Phase 9 & Document 1 Section J).

Per project rules: If a scientifically meaningful uncertainty cannot be derived
without unverified assumptions, it is exposed as NOT AVAILABLE rather than fabricating
probabilities.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import torch
from carbon_capture.epinn.model import EPINNModel
from carbon_capture.preprocessing.scaling import CanonicalScaler
import pandas as pd

@dataclass
class UncertaintyResult:
    status: str  # "AVAILABLE" or "NOT AVAILABLE"
    method: str  # "empirical_ensemble", "mc_dropout", or "none"
    mean_predictions: Dict[str, np.ndarray]
    std_predictions: Optional[Dict[str, np.ndarray]]
    confidence_scores: Optional[Dict[str, float]]
    notes: str

class UncertaintyEstimator:
    """Estimates epistemic and aleatoric uncertainty via MC-Dropout or ensemble."""

    def __init__(self, n_samples: int = 20):
        self.n_samples = n_samples

    def estimate_mc_dropout(
        self,
        model: EPINNModel,
        df: pd.DataFrame,
        scaler: CanonicalScaler,
        device: torch.device = torch.device("cpu"),
    ) -> UncertaintyResult:
        """Run Monte-Carlo Dropout passes to quantify epistemic prediction spread."""
        model.train()  # Enable dropout during inference
        x_scaled = scaler.transform(df)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=device)

        sample_preds: Dict[str, List[np.ndarray]] = {}

        with torch.no_grad():
            for _ in range(self.n_samples):
                out = model(x_tensor)
                for name, t in out.items():
                    if name not in sample_preds:
                        sample_preds[name] = []
                    sample_preds[name].append(t.cpu().numpy())

        means = {}
        stds = {}
        conf_scores = {}

        for name, arrays in sample_preds.items():
            stacked = np.stack(arrays, axis=0)  # (n_samples, batch_size)
            m = np.mean(stacked, axis=0)
            s = np.std(stacked, axis=0)
            means[name] = m
            stds[name] = s
            # Derived normalized dispersion: 1 / (1 + mean_std / mean_mag)
            mean_rel_disp = float(np.mean(s) / (np.mean(np.abs(m)) + 1e-6))
            conf_scores[name] = float(np.clip(1.0 - mean_rel_disp, 0.0, 1.0))

        model.eval()

        return UncertaintyResult(
            status="AVAILABLE",
            method="mc_dropout",
            mean_predictions=means,
            std_predictions=stds,
            confidence_scores=conf_scores,
            notes="Monte Carlo dropout epistemic variance across stochastic passes."
        )
