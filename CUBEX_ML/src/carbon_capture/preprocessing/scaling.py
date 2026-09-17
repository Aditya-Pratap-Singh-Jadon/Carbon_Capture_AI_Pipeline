"""Strict Leakage-Free Canonical Feature Scaler for Core Physical Telemetry (R^26)."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from carbon_capture.input.canonical_order import CORE_PHYSICAL_INPUT_ORDER
from carbon_capture.utils.logging import get_logger

logger = get_logger(__name__)

class CanonicalScaler:
    """Standardizes features based exclusively on training data distributions (X_core in R^26)."""

    def __init__(self, feature_names: Optional[List[str]] = None, eps: float = 1e-8):
        self.feature_names = feature_names or list(CORE_PHYSICAL_INPUT_ORDER)
        self.eps = eps
        self.mean_: Dict[str, float] = {}
        self.scale_: Dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "CanonicalScaler":
        """Compute mean and standard deviation on training data only."""
        for col in self.feature_names:
            if col not in df.columns:
                raise ValueError(f"Feature '{col}' not found in training DataFrame during fit.")
            mean_val = float(df[col].mean())
            std_val = float(df[col].std())
            if std_val < self.eps or np.isnan(std_val):
                std_val = 1.0  # Guard against constant/zero-variance features
            self.mean_[col] = mean_val
            self.scale_[col] = std_val
        self.is_fitted = True
        logger.info(f"Fitted CanonicalScaler on {len(self.feature_names)} core physical features.")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform dataframe to scaled numpy array in canonical order."""
        if not self.is_fitted:
            raise RuntimeError("CanonicalScaler must be fitted before transforming data.")

        missing = [col for col in self.feature_names if col not in df.columns]
        if missing:
            raise ValueError(f"Missing features for transform: {missing}")

        data = np.zeros((len(df), len(self.feature_names)), dtype=np.float32)
        for i, col in enumerate(self.feature_names):
            vals = df[col].to_numpy(dtype=np.float32)
            data[:, i] = (vals - self.mean_[col]) / self.scale_[col]

        return data

    def inverse_transform(self, arr: np.ndarray) -> pd.DataFrame:
        """Invert scaled array back to original units DataFrame."""
        if not self.is_fitted:
            raise RuntimeError("CanonicalScaler must be fitted before inverse transforming.")

        data_dict = {}
        for i, col in enumerate(self.feature_names):
            data_dict[col] = arr[:, i] * self.scale_[col] + self.mean_[col]

        return pd.DataFrame(data_dict)

    def get_characteristic_scales(self) -> Dict[str, float]:
        """Return the characteristic scale (std) for each feature."""
        return dict(self.scale_)

    def to_dict(self) -> dict:
        """Export state for metadata persistence and audit."""
        return {
            "feature_names": self.feature_names,
            "mean": self.mean_,
            "scale": self.scale_,
            "eps": self.eps,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, state: dict) -> "CanonicalScaler":
        scaler = cls(feature_names=state["feature_names"], eps=state["eps"])
        scaler.mean_ = state["mean"]
        scaler.scale_ = state["scale"]
        scaler.is_fitted = state["is_fitted"]
        return scaler
