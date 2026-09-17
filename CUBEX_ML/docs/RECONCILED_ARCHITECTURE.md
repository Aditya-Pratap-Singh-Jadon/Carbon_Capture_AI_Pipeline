# Reconciled System Architecture & Component Boundaries

**Document Version:** 1.0.0  
**Status:** Frozen Reconciled Architecture  
**Related Documents:** `docs/SPECIFICATION_RECONCILIATION.md`, `docs/RECONCILED_SCHEMA.md`, `docs/RECONCILED_EQUATIONS.md`

---

## 1. Architectural Reconciliation Decisions

### Modification 1: Authoritative Engine vs. Scientific Validation Authority
- **Original specification:** Stated that the E-PINN is the primary scientific engine, while also asking for deterministic carbon-credit calculation.
- **Issue:** Risk of conflating the neural network with the authoritative credit calculator.
- **Decision:** Establish a strict, unidirectional architectural boundary:
  $$\text{Sensor Input} \longrightarrow \text{E-PINN Reconstructs Physics} \longrightarrow \text{Validation Engine Evaluates Residuals} \longrightarrow \text{Deterministic Engine Computes Credits}$$
  The E-PINN *never* directly issues authoritative credits.
- **Reason:** Regulatory crediting frameworks require exact, closed-form algebraic traceability. A neural network is used to verify sensor integrity and physical plausibility, not to replace the accounting arithmetic.
- **Implementation Consequence:** Implemented in separate packages: `carbon_capture.epinn` and `carbon_capture.carbon_credit.calculator`.

### Modification 2: Separation of Process Control (Q-Learning) from Verification
- **Original specification:** Included both Q-learning controller requirements and E-PINN verification requirements.
- **Issue:** Temptation to combine RL and PINN into an end-to-end black-box agent.
- **Decision:** Q-learning operates in an independent control loop optimizing fan RPM against scrubber hydraulics and flue gas $\text{CO}_2$ concentration. The physical results of control actions (power, flows) subsequently flow through the E-PINN validation pipeline.
- **Reason:** Control is forward-looking dynamic optimization; verification is backward-looking scientific auditing.
- **Implementation Consequence:** `src/carbon_capture/rl/` is isolated from model training and verification decision code.

### Modification 3: Non-Authoritative Role of Statistical Diagnostics
- **Original specification:** Mentioned Isolation Forest and One-Class SVM.
- **Issue:** Statistical outlier detection often contradicts physical conservation laws (e.g. flagging a valid peak capture load as an anomaly).
- **Decision:** Isolation Forest and One-Class SVM are relegated strictly to optional diagnostics (`src/carbon_capture/anomaly/`). They have zero voting authority over the verification decision.
- **Reason:** Scientific physical verification must be grounded in conservation of mass and energy, not empirical multivariate distance.
- **Implementation Consequence:** Verification decisions depend exclusively on schema validation, physics residuals, and Lagrangian constraints.

---

## 2. Complete End-to-End Information Flow

```
+-----------------------------------------------------------------------------+
|                          1. INGESTION & HYGIENE                             |
|  Raw Telemetry (CSV / Dict / DataFrame)                                     |
|  - Canonical Order Enforcement: Maps aliases, reorders columns to Schema    |
|  - Data Hygiene: Rejects NaN/Inf, out-of-domain bounds, duplicate timestamps|
+-----------------------------------------------------------------------------+
                                       │
                                       ▼ (Clean Canonical Telemetry: X_core in R^26)
+-----------------------------------------------------------------------------+
|                          2. E-PINN SCIENTIFIC ENGINE                        |
|  Multi-Head Backbone (SiLU, LayerNorm, Dropout)                             |
|  - Reconstructs 15 physical quantities (E_red, E_rem, E_disp, PE_aux, ...)  |
|  - Evaluates 11 First-Principles Residuals (r_actual, r_removed, ...)       |
|  - Evaluates 3 Compositional Residuals (r_displaced, r_leakage, r_mult)     |
|  - Evaluates 1 Top-Level Credit Consistency Residual (r_CC)                 |
|  - Evaluates 23 Augmented Lagrangian Inequality Constraints (g_c <= 0)      |
+-----------------------------------------------------------------------------+
                                       │
                                       ▼ (Residuals, Preds, Constraint Violations)
+-----------------------------------------------------------------------------+
|                     3. SCIENTIFIC VALIDATION ENGINE                         |
|  - Evaluates mean normalized residual (threshold <= 1.5 sigma)              |
|  - Evaluates max normalized residual (threshold <= 3.0 sigma)               |
|  - Evaluates max constraint violation (threshold <= 1.0e-3)                 |
|  - Derives Validation Multiplier: F_valid in [0.0, 1.0]                     |
|  - Issues Verdict: VERIFIED | NOT VERIFIED | INSUFFICIENT DATA              |
+-----------------------------------------------------------------------------+
                                       │
                                       ▼ (F_valid, Validated Physical Quantities)
+-----------------------------------------------------------------------------+
|               4. DETERMINISTIC CARBON-CREDIT ENGINE                         |
|  Authoritative Closed-Form Accounting:                                      |
|  CC_T = [E_reduced + E_removed + E_displaced                                |
|          - PE_aux - PE_lifecycle - L_leakage]                               |
|         x (F_valid x F_SME x F_perm x alpha)                                |
+-----------------------------------------------------------------------------+
                                       │
                                       ▼ (Audit Data, CC_T, Diagnostics)
+-----------------------------------------------------------------------------+
|                      5. AUDIT TRAIL & REPORT PUBLISHER                      |
|  - Machine-Readable Audit Package: JSON                                     |
|  - Per-Record Ledger: CSV                                                   |
|  - Interactive Visual Dossier: HTML                                         |
|  - Executive / Third-Party Regulatory Report: PDF (ReportLab)               |
+-----------------------------------------------------------------------------+
```

---

## 3. Loss Architecture Formulation
$$L_{\text{total}} = L_{\text{data}} + L_{\text{physics}} + L_{\text{consistency}} + w_{CC} \cdot L_{CC} + L_{\text{constraints}}$$

1. **$L_{\text{data}}$**:
   $$L_{\text{data}} = \sum_{q \in \text{Observed}} \left[ \frac{(q_{\text{pred}} - q_{\text{obs}})^2}{2\sigma_q^2} + \ln \sigma_q \right] \quad \left( \text{or } \tilde{r}_q^2 \text{ where } \sigma_q \text{ is unknown} \right)$$
2. **$L_{\text{physics}}$**: Sum of squared normalized first-principles residuals:
   $$L_{\text{physics}} = \sum_{r \in \text{FirstPrinciples}} \left( \frac{r}{s_r} \right)^2$$
3. **$L_{\text{consistency}}$**: Sum of squared normalized compositional residuals (normalized by scale of larger operand):
   $$L_{\text{consistency}} = \left( \frac{r_{\text{displaced}}}{\max(|E_{\text{disp}}|, \epsilon)} \right)^2 + \left( \frac{r_{\text{leakage}}}{\max(|L_{\text{leak}}|, \epsilon)} \right)^2 + \left( \frac{r_{\text{mult}}}{\max(|F_{\text{mult}}|, 1.0)} \right)^2$$
4. **$L_{CC}$**: Top-level credit consistency normalized by $s_{CC}^2 = \text{Var}(CC_T)$:
   $$L_{CC} = w_{CC} \cdot \left( \frac{r_{\text{CC}}}{s_{CC}} \right)^2, \quad \text{with } w_{CC} = 1.0$$
5. **$L_{\text{constraints}}$**: Augmented Lagrangian inequality loss:
   $$L_{\text{constraints}} = \sum_{c=1}^{23} \left[ \mu_c \max(0, g_c)^2 + \lambda_c g_c \right]$$

---

## 4. Module Interface Contracts

| Module | Public Class / Interface | Inputs | Outputs |
|---|---|---|---|
| `carbon_capture.input` | `DataLoader` | File path or DataFrame | Canonical DataFrame, `ValidationResult` |
| `carbon_capture.preprocessing` | `CanonicalScaler` | Training DataFrame | Scaled tensor, Characteristic scales |
| `carbon_capture.epinn` | `EPINNInferenceEngine` | Clean Canonical DataFrame | `EPINNInferenceOutput` (preds, residuals, constraints) |
| `carbon_capture.validation` | `VerificationEngine` | `ValidationResult`, `EPINNInferenceOutput` | `VerificationDecision` (VERIFIED / NOT VERIFIED, $F_{\text{valid}}$) |
| `carbon_capture.carbon_credit` | `DeterministicCarbonCreditCalculator` | Canonical DataFrame, $F_{\text{valid}}$ | `CarbonCreditSummary`, Net $CC_T$ |
| `carbon_capture.reporting` | `ReportGenerator` | Audit payload, summaries | JSON, CSV, HTML, and PDF files |
| `carbon_capture.rl` | `RLProcessController` | `ProcessState` (CO₂, temp, fan, flow) | Recommended fan $\Delta\text{RPM}$ action |
