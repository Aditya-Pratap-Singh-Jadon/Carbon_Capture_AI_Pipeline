# Gaussian Matrix Measurement Model

**Phase 13 Component**

## Mathematical Model

The Gaussian model represents the statistical distribution of clean process telemetry. It models the observation vector $x$ as a multivariate normal distribution:

$$ x \sim \mathcal{N}(\mu, \Sigma) $$

Where:
- $x \in \mathbb{R}^6$ is the observed telemetry feature vector.
- $\mu \in \mathbb{R}^6$ is the estimated mean vector.
- $\Sigma \in \mathbb{R}^{6 \times 6}$ is the covariance matrix.

### Statistical Evidence metrics:
1. **Mahalanobis Distance Squared ($D_M^2$)**: Measures the distance of an observation from the mean, scaled by the covariance.
   $$ D_M^2 = (x - \mu)^T \Sigma^{-1} (x - \mu) $$
2. **Log Likelihood**: Logarithm of the probability density function.
   $$ \ln \mathcal{L} = -0.5 \left( k \ln(2\pi) + \ln|\Sigma| + D_M^2 \right) $$

## Input Vector

The model is strictly bounded to the 6 continuous variables available at the Digital Twin output boundary:

1. `CO2_ppm` (ppm)
2. `Temperature_C` (°C)
3. `Humidity_percent` (%)
4. `Gas_Flow_L_min` (L/min)
5. `Captured_CO2_g` (g)
6. `Fan_Speed_RPM` (RPM)

*Note: The `timestamp` is excluded from the statistical feature vector.*

## Missing Data Handling (Marginalization)

If a sensor drops out (e.g., `-1.0` sent by Digital Twin, parsed as `None` or `NaN`), the model **does NOT impute or fabricate values**. Instead, it uses exact mathematical marginalization over the available dimensions.

If we partition $x$ into observed $x_o$ and missing $x_m$, the marginal distribution of $x_o$ is exactly:
$$ x_o \sim \mathcal{N}(\mu_o, \Sigma_{oo}) $$

The Mahalanobis distance and log-likelihood are computed using only the observed dimensions, maintaining rigorous mathematical validity.

## Covariance Robustness

If the empirical covariance matrix $\Sigma$ is singular or near-singular (e.g., due to highly correlated features or zero variance in calibration), the model employs Tikhonov regularization:
$$ \Sigma' = \Sigma + \lambda I $$
Where $\lambda$ is a small configurable regularization term (default `1e-6`). When regularization is applied, `covariance_regularized=True` is explicitly flagged in the `GaussianResult`.

Numerical operations avoid explicit matrix inversion $\Sigma^{-1}$, relying instead on numerically stable Cholesky decomposition ($L L^T = \Sigma$) and triangular solves.
