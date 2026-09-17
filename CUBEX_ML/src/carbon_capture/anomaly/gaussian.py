"""Phase 13: Gaussian Matrix Measurement Model.

A multivariate Gaussian statistical model intended for clean process telemetry.
This models the 6 continuous fields from the 7-field Digital Twin telemetry contract:
(CO2_ppm, Temperature_C, Humidity_percent, Gas_Flow_L_min, Captured_CO2_g, Fan_Speed_RPM).
The timestamp is excluded from the statistical feature vector.

Features:
- Robust covariance estimation with configurable regularization.
- Mahalanobis distance and log-likelihood computation.
- Numerically stable linear algebra (Cholesky decomposition).
- Missing data handling via exact marginalization (no imputation).
- Parameter persistence.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import scipy.linalg
from carbon_capture.utils.logging import get_logger

logger = get_logger(__name__)

# The specific variables available at the Digital Twin boundary.
GAUSSIAN_FEATURES = [
    "CO2_ppm",
    "Temperature_C",
    "Humidity_percent",
    "Gas_Flow_L_min",
    "Captured_CO2_g",
    "Fan_Speed_RPM",
]


@dataclass
class GaussianResult:
    """Structured result of evaluating the Gaussian model on a single observation."""
    is_valid_input: bool
    mahalanobis_squared: float
    log_likelihood: float
    statistical_score: float  # e.g., Chi-square CDF or a calibrated confidence score
    covariance_regularized: bool
    observed_features: List[str]
    missing_features: List[str]
    diagnostics: str


class MultivariateGaussianModel:
    """
    Standalone statistical Gaussian model for process telemetry.
    Operates explicitly on the 6 continuous Digital Twin sensor fields.
    Does NOT fabricate or impute missing data; handles dropouts via marginalization.
    """

    def __init__(self, regularization: float = 1e-6):
        self.regularization = regularization
        self.mu: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None
        self.feature_names = GAUSSIAN_FEATURES
        self.dim = len(self.feature_names)
        self.is_calibrated = False
        self.was_regularized = False

    def calibrate(self, X: np.ndarray) -> None:
        """
        Calibrate the model (estimate mu and Sigma) from a clean reference dataset.
        X shape: (N, 6)
        """
        if X.shape[1] != self.dim:
            raise ValueError(f"Expected {self.dim} features, got {X.shape[1]}")
        
        # Check for NaN in calibration data
        if np.isnan(X).any():
            raise ValueError("Calibration dataset contains NaN. Clean reference data required.")

        N = X.shape[0]
        if N < self.dim + 1:
            raise ValueError(f"Insufficient calibration samples: {N}. Need > {self.dim}.")

        self.mu = np.mean(X, axis=0)
        # Unbiased covariance estimation
        self.sigma = np.cov(X, rowvar=False)

        # Check conditioning and regularize if needed
        try:
            # Try Cholesky to verify positive definiteness
            scipy.linalg.cholesky(self.sigma, lower=True)
            self.was_regularized = False
        except scipy.linalg.LinAlgError:
            logger.warning(f"Covariance matrix is singular/not PD. Applying regularization: +{self.regularization}*I")
            self.sigma += np.eye(self.dim) * self.regularization
            self.was_regularized = True
            
        self.is_calibrated = True
        logger.info(f"Gaussian model calibrated on {N} samples. Regularized: {self.was_regularized}")

    def evaluate_single(self, x_dict: Dict[str, Optional[float]]) -> GaussianResult:
        """
        Evaluate a single telemetry record given as a dictionary of features.
        Handles missing values (None or np.nan) via marginalization.
        """
        if not self.is_calibrated or self.mu is None or self.sigma is None:
            raise RuntimeError("Model must be calibrated before evaluation.")

        observed_indices = []
        missing_names = []
        observed_names = []
        x_vals = []

        for i, name in enumerate(self.feature_names):
            val = x_dict.get(name)
            if val is None or np.isnan(val):
                missing_names.append(name)
            else:
                observed_names.append(name)
                observed_indices.append(i)
                x_vals.append(val)

        if not observed_indices:
            return GaussianResult(
                is_valid_input=False,
                mahalanobis_squared=float('nan'),
                log_likelihood=float('nan'),
                statistical_score=float('nan'),
                covariance_regularized=self.was_regularized,
                observed_features=[],
                missing_features=missing_names,
                diagnostics="All features missing. Cannot evaluate.",
            )

        x_obs = np.array(x_vals, dtype=np.float64)
        idx = np.array(observed_indices, dtype=np.intp)
        
        # Marginalization: sub-vector mean and sub-matrix covariance
        mu_sub = self.mu[idx]
        # Advanced indexing for submatrix
        sigma_sub = self.sigma[np.ix_(idx, idx)]
        dim_sub = len(idx)

        # Add small regularization to submatrix if needed to prevent numerical instability during slicing
        if self.was_regularized:
            sigma_sub += np.eye(dim_sub) * 1e-8

        try:
            L = scipy.linalg.cholesky(sigma_sub, lower=True)
            diff = x_obs - mu_sub
            
            # Mahalanobis squared: (x - mu)^T Sigma^-1 (x - mu)
            # We solve L y = diff -> y = L^-1 diff, then y^T y = diff^T L^-T L^-1 diff = diff^T Sigma^-1 diff
            y = scipy.linalg.solve_triangular(L, diff, lower=True)
            mahalanobis_sq = np.dot(y, y)
            
            # Log determinant of Sigma_sub is 2 * sum(log(diag(L)))
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            
            # Log likelihood: -0.5 * (k * log(2pi) + log|Sigma| + D_M^2)
            log_lik = -0.5 * (dim_sub * np.log(2.0 * np.pi) + log_det + mahalanobis_sq)
            
            # Statistical score: rough proxy using Chi-square survival function (1 - CDF) could go here, 
            # but for now we'll just return a scaled version of distance or log-lik.
            # E.g., confidence decreases exponentially with Mahalanobis distance
            score = np.exp(-0.5 * mahalanobis_sq / dim_sub)
            
            diag_msg = f"Evaluated on {dim_sub} features."
            if missing_names:
                diag_msg += f" Marginalized over missing: {missing_names}"

            return GaussianResult(
                is_valid_input=True,
                mahalanobis_squared=float(mahalanobis_sq),
                log_likelihood=float(log_lik),
                statistical_score=float(score),
                covariance_regularized=self.was_regularized,
                observed_features=observed_names,
                missing_features=missing_names,
                diagnostics=diag_msg,
            )

        except scipy.linalg.LinAlgError:
            return GaussianResult(
                is_valid_input=False,
                mahalanobis_squared=float('nan'),
                log_likelihood=float('nan'),
                statistical_score=float('nan'),
                covariance_regularized=self.was_regularized,
                observed_features=observed_names,
                missing_features=missing_names,
                diagnostics="LinAlgError during evaluation (singular marginal covariance).",
            )

    def save(self, file_path: Union[str, Path]) -> None:
        """Persist model parameters."""
        if not self.is_calibrated:
            raise RuntimeError("Cannot save uncalibrated model.")
        with open(file_path, "wb") as f:
            pickle.dump({
                "mu": self.mu,
                "sigma": self.sigma,
                "feature_names": self.feature_names,
                "was_regularized": self.was_regularized,
                "regularization": self.regularization,
            }, f)
        logger.info(f"Model saved to {file_path}")

    def load(self, file_path: Union[str, Path]) -> None:
        """Load persisted model parameters."""
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.mu = data["mu"]
        self.sigma = data["sigma"]
        self.feature_names = data["feature_names"]
        self.was_regularized = data["was_regularized"]
        self.regularization = data["regularization"]
        self.dim = len(self.feature_names)
        self.is_calibrated = True
        logger.info(f"Model loaded from {file_path}")
