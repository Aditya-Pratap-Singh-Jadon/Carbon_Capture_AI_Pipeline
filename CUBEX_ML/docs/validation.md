# Scientific Verification Framework & Decision Engine

## 1. Decision Criteria
The verification engine issues one of three formal decisions:
- **`VERIFIED`**:
  - Input passes all schema, domain, and data hygiene checks.
  - Normalized physics residuals across all 15 governing equations satisfy:
    $$\max_r |\tilde{r}| \le 3.0 \quad \text{and} \quad \frac{1}{|R|}\sum_r |\tilde{r}| \le 1.5$$
  - Augmented Lagrangian physical and operational constraints satisfy $g_c \le 10^{-3}$.
- **`NOT VERIFIED`**:
  - Violation of schema, corrupted sensors (e.g. $\eta_{\text{purity}} > 1.0$), negative physical values, or residual violations exceeding tolerance thresholds.
  - In this state, **0.00 verified credits** are certified.
- **`INSUFFICIENT DATA`**:
  - Empty or missing record streams.

## 2. Statistical Diagnostics (Non-Authoritative)
Isolation Forest and One-Class SVM models are provided as optional baseline diagnostic tools. They do not have authority over physical verification decisions.
