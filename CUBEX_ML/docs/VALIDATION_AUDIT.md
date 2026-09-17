# Scientific Validation & Failure-Mode Audit Report

This report documents a rigorous audit of the existing E-PINN carbon capture verification logic, specifically targeting the mathematical robustness of the system against adversarial failures and evaluating the boundaries of its current constraint mechanisms.

## 1. Overall Audit Result

**Status**: PROVISIONALLY VERIFIED, BUT CHECKS LACK GLOBAL BOUNDING  
**Summary**: The system behaves perfectly according to the frozen scientific specification (Document 1). However, a rigid adherence to this specification reveals significant "local" verification that lacks "global" physical constraint bounding. The pipeline successfully flags anomalies like purity `> 1.0`, negative temperatures/power, or mathematically corrupted deterministic credit calculations. However, without overarching global constraints linking input carbon to output byproduct quantities, it remains mathematically blind to structural absurdities (such as fabricating arbitrary masses of chemical byproduct).

## 2. Mass-Balance Failure Investigation

**Adversarial Vector**: Intentionally generating a test dataset where `M_byproduct = M_captured * 3.5`. This reflects an impossible physical scenario where the byproduct output mass significantly exceeds the captured carbon input mass.

**Outcome**: VERIFIED (High F_valid ~ 1.0)

**Explanation**: 
1. **Mathematical Consistency over Plausibility**: The E-PINN loss function minimizes the residual `r_disp_chem = E_displaced_chem - (M_byproduct * EF_virgin_displace)`. Because this is a multiplicative algebraic relationship and both `M_byproduct` and `EF_virgin_displace` are observed telemetry inputs provided to the network, the E-PINN trivially fits this mapping. It perfectly reconstructs an anomalous `E_displaced_chem` that mathematically aligns with the anomalous `M_byproduct`.
2. **Low Residuals**: Because the neural network's prediction correctly mirrors the algebraic equation, the residual `r_disp_chem` approaches zero. Normalization keeps this residual well under the 1.5/3.0 tolerance threshold.
3. **No Existing Constraint is Violated**: The only constraint evaluating `M_byproduct` is `c_m_byproduct_nonneg` (`-M_byproduct <= 0`). Since the anomalous mass is positive, the constraint is perfectly satisfied.
4. **Validation Policy Blind Spot**: The `VerificationEngine` awards `F_valid ≈ 1.0` because there are no residual anomalies and no physical bounds broken (among the frozen set of 23 constraints). While an unsupervised anomaly detector (Isolation Forest) would easily flag this point as a statistical outlier, statistical methods are strictly non-authoritative in this architecture.
5. **Conclusion (Option 2)**: The architecture behaves exactly as specified. We do not invent an arbitrary `max_ratio` constraint because the scientific document does not provide industrial-specific stoichiometric boundaries. This is marked as a **Known Limitation**.

## 3. `F_valid` Derivation Audit

**Objective**: Ensure no circularity or data leakage between the EPINN inputs and the derived Verification Multiplier (`F_valid`).
- **Result**: PASS.
- **Trace**: `F_valid` is evaluated completely post-inference in `VerificationEngine`. It depends strictly on `epinn_output.mean_normalized_residual` and `epinn_output.constraint_violations`. The network inputs include a pre-existing (or default) telemetry `F_valid` which acts solely as a feature for `CC_T` prediction, but the authoritative verification `F_valid` is derived entirely from the physical reconstruction confidence. There is no circular data leakage.

## 4. 15-Residual Audit

All 15 residuals evaluate exact equality between the EPINN's reconstructed quantities (e.g., `E_actual`) and the deterministic algebraic equations evaluated using input telemetry.
- **Equations & Sign Convention**: `preds[qty] - eq_qty(inputs)`. Standard difference.
- **Differentiability**: Fully differentiable via PyTorch.
- **Normalization**: Performed using characteristic scales (e.g., 10.0 for energy terms, 100.0 for total `CC_T`). Compositional variables use maximum absolute operand normalization as requested by the specification.
- **Loss Contribution**: Minimized symmetrically using MSE.
- **Verification Contribution**: `mean_normalized_residual` heavily penalizes `F_valid`.
- **Result**: PASS.

## 5. 23-Constraint Audit

The 23 constraints evaluate physical boundaries via the Augmented Lagrangian dual-ascent method.
- **Formulation**: `g_c <= 0`. E.g., `eta_purity - 1.0 <= 0`.
- **Implementation**: Strictly implemented via `torch.relu(g_c)`.
- **Effect on Verification**: `VerificationEngine` enforces a strict threshold `max_viol <= 1.0e-3`. Any violation above this zeroes `F_valid`.
- **Result**: PASS. (Note: Missing global constraints, e.g., mass-balance, are intentionally excluded as they were not in the frozen spec).

## 6. Carbon-Credit Engine Audit

**Objective**: Verify `CC_T` is calculated independently from the E-PINN prediction.
- **Result**: PASS.
- **Trace**: The `DeterministicCarbonCreditCalculator` re-evaluates all 15 equations purely from the observed input data, independently of the E-PINN outputs. It then applies the *derived* `F_valid` from the `VerificationEngine`. The E-PINN is strictly an observer/validator, not the final calculator.

## 7. Threshold Inventory

| Name | Value | Unit | Reason | Location |
|---|---|---|---|---|
| `max_residual_tol` | 3.0 | $\sigma$ (Norm) | Hard rejection for single extreme physical deviation | `verification.py` |
| `mean_residual_tol`| 1.5 | $\sigma$ (Norm) | Hard rejection for systematic average deviation | `verification.py` |
| `constraint_tol`   | 1e-3 | Physical unit | Precision boundary for physical inequality bounds | `verification.py` |
| `kappa_penalty`    | 0.5 | scalar | Exponential decay rate for `F_valid` | `verification.py` |

All thresholds are mathematically explicit and located centrally in `VerificationEngine`.

## 8. Adversarial Test Matrix

The system was tested against the following adversarial scenarios:
- **A. Clean Data**: PASS (VERIFIED, `F_valid > 0.95`)
- **B. Sensor Bias (Purity > 100%)**: REJECTED (Violates `c_eta_purity_upper`)
- **C. Negative Power**: REJECTED (Violates `c_p_fan_nonneg`)
- **D. Severe Mass-Balance Violation**: VERIFIED (Passes due to lack of specification constraint - **Known Limitation**)
- **E. Shuffled CSV Columns**: PASS (VERIFIED, `CANONICAL_INPUT_ORDER` dynamically resolves headers)
- **F. Missing Required Column**: REJECTED (`InputSchema` validation fails)

## 9. Training Leakage, Persistence, & Reproducibility Audit

- **Leakage**: Scalers are strictly fitted on training data. `ValidationResult` and `SyntheticDataGenerator` ensure test data sets do not bleed.
- **Persistence**: Models are correctly exported via `torch.save(model.state_dict(), ...)` and reloaded perfectly. Deterministic inference verifies state preservation.
- **Reproducibility**: Setting PyTorch and NumPy random seeds ensures identical outputs across runs for identical inputs.

## 10. Known Limitations & Recommended Scientific Improvements

1. **Mass Balance & Stoichiometry Constraints**: The system urgently requires rigorous global physical boundary constraints connecting carbon inputs (e.g., `FC_actual`, `M_captured`) to downstream physical products (`M_byproduct`). Without these, the algebraic reconstruction handles arbitrary masses without violating local equations. This would require industrial/process-specific stoichiometric evidence to define a scientifically valid mass-balance constraint.
2. **PE_aux Dimensional Inconsistency**: The verbatim implementation of `PE_aux = EC_hw - EF_grid` preserves a dimensional contradiction (kWh minus tCO2/kWh). A physical mode is configurable but defaults to verbatim for strict specification alignment.
3. **Overreliance on Statistical Detectors for Outliers**: Because the EPINN learns underlying mathematical mappings, numerical outliers that still obey the algebraic structure are completely invisible to physics verification. This mandates formally integrating anomaly detection into the verification decision boundary in future iterations.
