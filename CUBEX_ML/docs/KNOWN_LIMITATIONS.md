# Known Limitations

**CUBEX v1.0.0**

---

## L-01 — Digital Twin Cannot Drive E-PINN Inference (Hard Blocker)

**Description:** The Digital Twin provides exactly 7 telemetry fields. The E-PINN requires 26 physical variables. Of these 26, **11 are UNAVAILABLE** from the current Digital Twin contract and **0 are directly observed**.

**Blocked variables:** `FC_actual`, `FC_baseline`, `EC_actual`, `EC_baseline`, `eta_purity`, `P_fan`, `P_pump`, `M_byproduct`, `H_recovered`, `M_solvent_makeup`, `M_trans_k`

**Impact:** No valid E-PINN inference can be performed on Digital Twin telemetry without fabricating data. This is prohibited by the system architecture.

**Resolution path:** Extend the Digital Twin with at minimum: fuel flow meter, electricity meter, inline CO₂ purity analyser, pump power meter, product mass flow meter, solvent dosing meter, and heat flow meter. Alternatively, restrict E-PINN to a reduced model operating on a subset of residuals — but this requires specification revision.

---

## L-02 — Mass-Balance Adversarial Detection Limitation

**Description:** The E-PINN can reconstruct an algebraically consistent but physically implausible `M_byproduct` value (e.g. higher than `M_captured`). This occurs because the 15 residuals enforce local equation consistency, not global physical plausibility across variables.

**Impact:** A deliberately corrupted mass-balance dataset may receive a provisional VERIFIED result if all individual equation residuals are within tolerance.

**Why an arbitrary ratio constraint is not added:** No scientifically justified `max_ratio = M_byproduct / M_captured` is defined in the project specification. Inventing one would constitute fabrication.

**Resolution path:** Define and justify a process-specific material yield constraint from industrial chemistry literature, then add it as constraint c_24.

---

## L-03 — Uncertainty Threshold Not Scientifically Justified

**Description:** The `_UNCERTAINTY_FLAG_THRESHOLD = 0.5` used to flag HIGH_UNCERTAINTY is a conservative but not formally justified threshold. No industrial standard for MC-Dropout thresholding exists within the project specification.

**Impact:** The HIGH_UNCERTAINTY flag is diagnostic only. It does not change F_valid or carbon credits.

**Resolution path:** Calibrate threshold against held-out validation data using reliability diagrams or expected calibration error.

---

## L-04 — Fan Power Curve Required for P_fan Derivation

**Description:** `Fan_Speed_RPM` (RPM) cannot be converted to `P_fan` (kW) without a calibrated fan power curve. The power curve is hardware-specific and not part of the telemetry contract.

**Impact:** `P_fan` remains UNAVAILABLE from Digital Twin telemetry. `EC_hardware` and `PE_aux` residuals cannot be computed.

**Resolution path:** Add fan model and power curve to system configuration (`hardware.fan_power_curve`) to enable LEGITIMATELY_DERIVED classification.

---

## L-05 — Phase 5 Baseline Model Not Trained on Real Industrial Data

**Description:** The `BaselineMLP` comparison model uses the same synthetic dataset as the E-PINN. Real-world comparison requires real measurement data.

**Impact:** The E-PINN vs Baseline comparison is illustrative, not definitive.

---

## L-06 — Q-Learning Environment is Simplified Simulation

**Description:** `ScrubberEnvironment` uses a simplified physics model for RL training (linear fan-to-flow mapping, Gaussian noise). Real scrubber dynamics are nonlinear and require calibration against measurement data.

**Impact:** Q-table trained in simulation may not transfer directly to real hardware without retraining or fine-tuning.

---

## L-07 — t_op Derivation Assumes Continuous Operation

**Description:** `t_op` is derived as `timestamp[T] - timestamp[T-1]`. Any gap in operation (e.g. shutdown, restart) within a reporting period will be silently included as operating time.

**Resolution path:** Add a `process_active` sentinel or heartbeat field to the telemetry contract.

---

## L-08 — SHAP/LIME Not Implemented

**Description:** The explainability layer uses vanilla gradients (saliency maps). SHAP and LIME are not implemented because: (a) KernelSHAP can violate physics constraints in its perturbations; (b) LIME's local linearity assumption conflicts with the nonlinear physics structure.

**Impact:** Gradient attribution does not produce game-theoretically fair feature attributions (Shapley values).

**Resolution path:** Implement DeepLIFT or Integrated Gradients, which are gradient-based and compatible with neural network physics constraints.
