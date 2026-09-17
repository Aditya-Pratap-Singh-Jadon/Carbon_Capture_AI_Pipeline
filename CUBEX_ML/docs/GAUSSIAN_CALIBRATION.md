# Gaussian Model Calibration

**Phase 13 Component**

## Calibration Requirements

The Gaussian model must be calibrated on a clean, representative dataset of normal process behavior.

**Requirements:**
1. **No Anomalies**: The calibration dataset must contain exclusively normal, steady-state, or legitimate transitional telemetry.
2. **Completeness**: No missing values (`NaN` or `None`) are allowed in the calibration dataset.
3. **Sample Size**: Must have at least $N > 6$ (the dimensionality of the feature vector). In practice, $N \gg 6$ is required for stable covariance estimation.

## Process

1. **Load Data**: Ingest a clean dataset spanning the 6 continuous variables.
2. **Mean Estimation**:
   $$ \mu = \frac{1}{N} \sum_{i=1}^N x_i $$
3. **Covariance Estimation**:
   $$ \Sigma = \frac{1}{N-1} \sum_{i=1}^N (x_i - \mu)(x_i - \mu)^T $$
4. **Regularization Check**: The system attempts a Cholesky decomposition of $\Sigma$. If it fails (LinAlgError due to non-positive definite properties), Tikhonov regularization is applied ($\Sigma \leftarrow \Sigma + \lambda I$).
5. **Persistence**: The model parameters ($\mu$, $\Sigma$, feature names, regularization state) are serialized via Pickle and saved to disk.

## Determinism

Calibration is purely deterministic linear algebra. Providing the same clean dataset will yield exactly identical $\mu$ and $\Sigma$ parameters.
