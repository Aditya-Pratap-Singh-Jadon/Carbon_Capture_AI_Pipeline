"""Optional Statistical Diagnostics (Isolation Forest & One-Class SVM).

IMPORTANT:
Per Section 17 of project specification: Isolation Forest and One-Class SVM
are OPTIONAL STATISTICAL DIAGNOSTICS ONLY. They are NOT the scientific validation authority.
The E-PINN physics residuals and deterministic credit mathematics are the authoritative engines.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from carbon_capture.input.canonical_order import CANONICAL_INPUT_ORDER
from carbon_capture.preprocessing.scaling import CanonicalScaler
from carbon_capture.utils.logging import get_logger

logger = get_logger(__name__)

class StatisticalAnomalyDiagnostics:
    """Optional statistical baseline for comparison and exploratory diagnostics."""

    def __init__(self, contamination: float = 0.05, seed: int = 42):
        self.iso_forest = IsolationForest(contamination=contamination, random_state=seed)
        self.oc_svm = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, scaler: CanonicalScaler) -> "StatisticalAnomalyDiagnostics":
        """Fit statistical models on scaled training data."""
        x = scaler.transform(df)
        self.iso_forest.fit(x)
        self.oc_svm.fit(x)
        self.is_fitted = True
        logger.info("Fitted optional Isolation Forest and One-Class SVM diagnostics.")
        return self

    def diagnose(self, df: pd.DataFrame, scaler: CanonicalScaler) -> Dict[str, np.ndarray]:
        """Compute statistical outlier scores."""
        if not self.is_fitted:
            raise RuntimeError("Statistical diagnostics must be fitted first.")

        x = scaler.transform(df)
        # Predictions: 1 for inlier, -1 for outlier
        if_preds = self.iso_forest.predict(x)
        svm_preds = self.oc_svm.predict(x)

        # Anomaly scores (lower score = more anomalous)
        if_scores = self.iso_forest.decision_function(x)

        return {
            "isolation_forest_outliers": np.where(if_preds == -1, 1, 0),
            "oc_svm_outliers": np.where(svm_preds == -1, 1, 0),
            "if_score": if_scores,
            "role": "OPTIONAL_STATISTICAL_DIAGNOSTIC",
        }
