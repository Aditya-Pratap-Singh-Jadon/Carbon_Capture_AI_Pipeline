# Reconciled Data Schema: Carbon Capture AI Pipeline

**Document Version:** 1.0.0  
**Status:** Frozen Reconciled Contract  
**Related Documents:** `docs/SPECIFICATION_RECONCILIATION.md`, `docs/RECONCILED_EQUATIONS.md`

---

## 1. Schema Reconciliation Decisions

### Modification 1: Canonical Input Dimensionality
- **Original specification:** Stated as $\mathbf{X} \in \mathbb{R}^{28}$, but listed 30 variables.
- **Issue:** Dimensional inconsistency between textual dimension count (28) and enumerated fields (30).
- **Decision:** Split the schema into **Physical Process Inputs** ($\mathbf{X}_{\text{core}} \in \mathbb{R}^{26}$), **Policy Parameters** ($\mathbf{P} \in \mathbb{R}^3$), and **Validation Multiplier** ($F_{\text{valid}}$, output only). The full tabular input stream supports $D = 29$ (when policy parameters are included as contextual metadata) or $D = 26$ for the core physical E-PINN.
- **Reason:** $F_{\text{valid}}$ cannot be an input to the validation engine without circularity. Policy parameters ($F_{\text{SME}}, F_{\text{perm}}, \alpha$) are administrative scalars, not process sensor telemetry.
- **Implementation Consequence:** Eliminates model dimension mismatch. The neural network backbone is parameterized on the exact verified feature dimension.

### Modification 2: Exclusion of $F_{\text{valid}}$ from Neural Network Input
- **Original specification:** Listed $F_{\text{valid}}$ in the canonical input vector while naming it an "AI-Driven Validation Multiplier".
- **Issue:** Validation leakage and circular dependency ($F_{\text{valid}} \to \text{E-PINN} \to \text{Residuals} \to F_{\text{valid}}$).
- **Decision:** $F_{\text{valid}}$ is strictly excluded from the E-PINN model input vector. It is generated as an output of the post-E-PINN verification engine.
- **Reason:** A verification system cannot receive its own verdict as a feature.
- **Implementation Consequence:** $F_{\text{valid}}$ is computed in `VerificationEngine` and passed downstream to `DeterministicCarbonCreditCalculator`.

### Modification 3: Indexed Variables ($i$ Fuels and $k$ Transport Modes)
- **Original specification:** Equations state $\sum_i$ and $\sum_k$, but table lists scalar fields.
- **Issue:** Multi-fuel combustion and multi-modal transport cannot be expressed in a single scalar cell without aggregation rules.
- **Decision:** Establish a two-tier ingestion contract:
  1. Default single-fuel / single-leg: Direct scalar mapping.
  2. Multi-fuel / multi-leg: `DataLoader` aggregates energy-weighted emissions for combustion and total ton-km for logistics prior to model vector assembly.
- **Reason:** Maintains strict tensor dimension uniformity for batch processing while preserving exact summation arithmetic.
- **Implementation Consequence:** `DataLoader` handles both scalar and multi-leg aggregated inputs transparently.

---

## 2. Reconciled Canonical Input Table

### A. Core Physical Telemetry Vector ($\mathbf{X}_{\text{core}} \in \mathbb{R}^{26}$)
These 26 variables form the input vector to the E-PINN scientific validation engine:

| Index | Variable Name | Canonical Internal Unit | Physical Meaning | Admissible Domain |
|---|---|---|---|---|
| 0 | `FC_actual` | tonnes | Actual fuel consumption in reporting period | $[0, 10^7]$ |
| 1 | `FC_baseline` | tonnes | Counterfactual baseline fuel consumption | $[0, 10^7]$ |
| 2 | `NCV` | MJ/kg | Fuel Net Calorific Value | $(0, 200]$ |
| 3 | `EF_CO2` | $\text{tCO}_2/\text{t}$ | Fuel stoichiometric CO₂ emission factor | $(0, 10]$ |
| 4 | `OF` | dimensionless | Fuel Oxidation Factor (unburnt carbon fraction) | $[0, 1.0]$ |
| 5 | `EC_actual` | kWh | Actual facility electricity consumption | $[0, 10^8]$ |
| 6 | `EC_baseline` | kWh | Baseline facility electricity consumption | $[0, 10^8]$ |
| 7 | `EF_grid` | $\text{tCO}_2/\text{kWh}$ | Regional grid electricity emission intensity | $[0, 2.0]$ |
| 8 | `M_captured` | tonnes | Mass of captured / synthesized carbonate product | $[0, 10^6]$ |
| 9 | `P_CO2` | dimensionless | Stoichiometric mass fraction of CO₂ in product | $[0, 1.0]$ |
| 10 | `eta_purity` | dimensionless | Lab-assayed chemical purity of captured product | $[0, 1.0]$ |
| 11 | `M_byproduct` | tonnes | Displaced circular chemical byproduct delivered | $[0, 10^6]$ |
| 12 | `EF_virgin_displace`| $\text{tCO}_2\text{e}/\text{t}$ | Embodied carbon of virgin chemical displaced | $[0, 20.0]$ |
| 13 | `H_recovered` | MJ | Exothermic / waste heat reclaimed for process use | $[0, 10^9]$ |
| 14 | `eta_boiler` | dimensionless | Efficiency of on-site displaced steam/hot water boiler | $(0, 1.0]$ |
| 15 | `NCV_fuel` | MJ/kg | Net calorific value of fuel displaced for heat | $(0, 200]$ |
| 16 | `EF_CO2_fuel` | $\text{tCO}_2/\text{MJ}$ | Emission factor of fuel displaced for heat | $(0, 10]$ |
| 17 | `P_fan` | kW | Active electrical power drawn by scrubber fan | $[0, 10^4]$ |
| 18 | `P_pump` | kW | Active electrical power drawn by solvent pump | $[0, 10^4]$ |
| 19 | `t_op` | seconds | Total hardware active operating run time | $[0, 3.15 \times 10^7]$ |
| 20 | `M_solvent_makeup` | tonnes | Active chemical solvent replenishment mass | $[0, 10^5]$ |
| 21 | `EF_solvent_LCA` | $\text{tCO}_2\text{e}/\text{t}$ | Sorbent cradle-to-gate carbon intensity | $[0, 50.0]$ |
| 22 | `gamma_slip` | dimensionless | Fraction of captured CO₂ lost via slippage | $[0, 1.0]$ |
| 23 | `D_k` | km | Total transit distance for deliveries/byproduct | $[0, 50000]$ |
| 24 | `EF_vehicle_k` | $\text{tCO}_2\text{e}/\text{t-km}$| Logistics vehicle mode emission factor | $[0, 5.0]$ |
| 25 | `M_trans_k` | tonnes | Total cargo mass transported | $[0, 10^6]$ |

---

### B. Project Policy & Administrative Multipliers ($\mathbf{P} \in \mathbb{R}^3$)
These variables are project-level administrative constants used in deterministic credit calculation and policy consistency validation:

| Index | Variable Name | Unit | Role | Allowed Domain | Source |
|---|---|---|---|---|---|
| 26 | `F_SME` | dimensionless | MSME enterprise scale incentive multiplier | $[1.0, \infty)$ | Registry rule table |
| 27 | `F_perm` | dimensionless | 100-year containment permanence factor | $[0, 1.0]$ | Storage class protocol |
| 28 | `alpha` | dimensionless | Pre-set precision conservatism discount | $[0, 1.0]$ | Crediting standard policy |

---

### C. Validation & Reconstructed Quantities (Model & Engine Outputs)

| Output Name | Classification | Governing Formula / Equation | Unit |
|---|---|---|---|
| `E_actual` | Physical State | Equation 1 | $\text{tCO}_2\text{e}$ |
| `E_baseline` | Physical State | Equation 2 | $\text{tCO}_2\text{e}$ |
| `E_reduced` | Physical State | Equation 3 | $\text{tCO}_2\text{e}$ |
| `E_removed` | Physical State | Equation 4 | $\text{tCO}_2\text{e}$ |
| `E_displaced_chem` | Physical State | Equation 5 | $\text{tCO}_2\text{e}$ |
| `E_displaced_heat` | Physical State | Equation 6 | $\text{tCO}_2\text{e}$ |
| `E_displaced` | Physical State | Equation 7 | $\text{tCO}_2\text{e}$ |
| `EC_hardware` | Physical State | Equation 8 | $\text{kWh}$ |
| `PE_aux` | Physical State | Equation 9 | $\text{tCO}_2\text{e}$ (or kWh in verbatim) |
| `PE_lifecycle` | Physical State | Equation 10 | $\text{tCO}_2\text{e}$ |
| `L_slip` | Physical State | Equation 11 | $\text{tCO}_2\text{e}$ |
| `L_transport` | Physical State | Equation 12 | $\text{tCO}_2\text{e}$ |
| `L_leakage` | Physical State | Equation 13 | $\text{tCO}_2\text{e}$ |
| `F_multiplier` | Combined Factor | Equation 14 | dimensionless |
| `CC_T` | Authoritative Credit | Equation 15 | $\text{tCO}_2\text{e}$ |
| `F_valid` | Validation Metric | Post-PINN Residual Assessment | dimensionless $[0, 1]$ |
