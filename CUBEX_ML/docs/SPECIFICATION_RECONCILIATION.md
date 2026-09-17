# Specification Reconciliation Report: Carbon-Capture E-PINN Verification & Carbon-Credit AI Pipeline

**Document Version:** 1.0.0  
**Status:** Mandatory Pre-Implementation Reconciliation Pass  
**Author:** AI/ML Science & Software Architecture Team  
**Governing Documents:**  
1. *Information-Preserving E-PINN Objective Function for the Carbon-Credit Framework: Mathematical Derivation* (Doc 1)  
2. *Extended Physics-Informed Neural Network (E-PINN): Phase-Wise Development & Coding Plan* (Doc 2)  
3. *Factors to Consider in Our Model: Definitions & Parameters* (Doc 3)  

---

## Executive Summary
This document executes a formal **Specification Reconciliation Pass** across the mathematical specifications, data schemas, loss derivations, and component boundaries. Before freezing the neural network architecture and training pipelines, all internal ambiguities, dimensional discrepancies, variable dimensionalities, circularities, and residual taxonomies are isolated, analyzed, and explicitly resolved with clear scientific justifications.

---

## Issue 1: Canonical Input Count & Dimensionality Reconciled

### 1.1 Statement of Discrepancy
- Document 1 Section B and preliminary summaries cite: $\mathbf{X} \in \mathbb{R}^{28}$.
- However, direct enumeration of all named inputs across Document 1 and Document 3 yields **30 candidate variables**.

### 1.2 Explicit Variable Enumeration & Classification
Below is the independent count and audit of all 30 candidate variables:

| Index | Variable Name | Unit | Category | Mathematical Type | E-PINN Input? | Required in Schema? |
|---|---|---|---|---|---|---|
| 0 | `FC_actual` | tonnes | 1. Fuel / Direct Combustion | Continuous Scalar | **Yes** | Yes |
| 1 | `FC_baseline` | tonnes | 1. Fuel / Direct Combustion | Continuous Scalar | **Yes** | Yes |
| 2 | `NCV` | MJ/kg | 1. Fuel / Direct Combustion | Continuous Scalar | **Yes** | Yes |
| 3 | `EF_CO2` | $\text{tCO}_2/\text{t}$ | 1. Fuel / Direct Combustion | Continuous Scalar | **Yes** | Yes |
| 4 | `OF` | dimensionless | 1. Fuel / Direct Combustion | Bounded Fraction $[0,1]$ | **Yes** | Yes |
| 5 | `EC_actual` | kWh | 2. Scope 2 Electricity | Continuous Scalar | **Yes** | Yes |
| 6 | `EC_baseline` | kWh | 2. Scope 2 Electricity | Continuous Scalar | **Yes** | Yes |
| 7 | `EF_grid` | $\text{tCO}_2/\text{kWh}$ | 2. Scope 2 Electricity | Continuous Scalar | **Yes** | Yes |
| 8 | `M_captured` | tonnes | 3. Capture & Synthesis | Continuous Scalar | **Yes** | Yes |
| 9 | `P_CO2` | dimensionless | 3. Capture & Synthesis | Stoichiometric Fraction $[0,1]$ | **Yes** | Yes |
| 10 | `eta_purity` | dimensionless | 3. Capture & Synthesis | Assayed Fraction $[0,1]$ | **Yes** | Yes |
| 11 | `M_byproduct` | tonnes | 4. Displaced Products / Heat | Continuous Scalar | **Yes** | Yes |
| 12 | `EF_virgin_displace` | $\text{tCO}_2\text{e}/\text{t}$ | 4. Displaced Products / Heat | Reference Factor | **Yes** | Yes |
| 13 | `H_recovered` | MJ | 4. Displaced Products / Heat | Continuous Scalar | **Yes** | Yes |
| 14 | `eta_boiler` | dimensionless | 4. Displaced Products / Heat | Bounded Fraction $[0,1]$ | **Yes** | Yes |
| 15 | `NCV_fuel` | MJ/kg | 4. Displaced Products / Heat | Continuous Scalar | **Yes** | Yes |
| 16 | `EF_CO2_fuel` | $\text{tCO}_2/\text{MJ}$ | 4. Displaced Products / Heat | Continuous Scalar | **Yes** | Yes |
| 17 | `P_fan` | kW | 5. Auxiliary Hardware | Continuous Scalar | **Yes** | Yes |
| 18 | `P_pump` | kW | 5. Auxiliary Hardware | Continuous Scalar | **Yes** | Yes |
| 19 | `t_op` | seconds | 5. Auxiliary Hardware | Duration Scalar | **Yes** | Yes |
| 20 | `M_solvent_makeup` | tonnes | 6. Sorbent Degradation | Continuous Scalar | **Yes** | Yes |
| 21 | `EF_solvent_LCA` | $\text{tCO}_2\text{e}/\text{t}$ | 6. Sorbent Degradation | Reference Factor | **Yes** | Yes |
| 22 | `gamma_slip` | dimensionless | 7. Slippage & Distribution | Bounded Fraction $[0,1]$ | **Yes** | Yes |
| 23 | `D_k` | km | 7. Slippage & Distribution | Continuous Scalar | **Yes** | Yes |
| 24 | `EF_vehicle_k` | $\text{tCO}_2\text{e}/\text{t-km}$| 7. Slippage & Distribution | Reference Factor | **Yes** | Yes |
| 25 | `M_trans_k` | tonnes | 7. Slippage & Distribution | Continuous Scalar | **Yes** | Yes |
| 26 | `F_valid` | dimensionless | 8. Validation Multiplier | Metric in $[0,1]$ | **NO (Output)** | No (Excluded from Model Input) |
| 27 | `F_SME` | dimensionless | 9. Policy / Multipliers | Policy Scalar $\ge 1.0$ | **NO (Policy)** | Yes (Credit Engine Parameter) |
| 28 | `F_perm` | dimensionless | 9. Policy / Multipliers | Policy Fraction $[0,1]$ | **NO (Policy)** | Yes (Credit Engine Parameter) |
| 29 | `alpha` | dimensionless | 9. Policy / Multipliers | Policy Fraction $[0,1]$ | **NO (Policy)** | Yes (Credit Engine Parameter) |

### 1.3 Resolution: True Model Dimensionality vs. Full Pipeline Schema
- **The Core Physical Telemetry Vector (Model Input)** consists of indices **0 through 25**, which is exactly **26 physical process and reference inputs**:
  $$\mathbf{X}_{\text{core}} \in \mathbb{R}^{26}$$
- If policy parameters (`F_SME`, `F_perm`, `alpha`) are bundled into the model forward vector to allow the network to evaluate the multiplier residual $r_{\text{mult}}$, the input dimensionality becomes:
  $$\mathbf{X}_{\text{with\_policy}} \in \mathbb{R}^{29}$$
- **$F_{\text{valid}}$ is strictly excluded from the neural-network input vector** (see Issue 2 below) to eliminate circularity.
- **Decision:** The canonical input vector for the neural network is frozen at **$\mathbf{X} \in \mathbb{R}^{26}$ (Physical Only)** or **$\mathbf{X} \in \mathbb{R}^{29}$ (Physical + Policy Context)**. The document's "28" was an unverified draft count resulting from omitting $t_{\text{op}}$ and $P_{\text{pump}}$ or conflating policy multipliers. We freeze the exact dimensionality to **26 for the Pure Physical E-PINN** and **29 when including project policy scalars**.

---

## Issue 2: $F_{\text{valid}}$ Circularity & Architectural Role

### 2.1 Statement of the Problem
- Document 1 Section B classifies $F_{\text{valid}}$ as *"Indirectly observable / potentially predicted"* and Section R.2 flags:
  > *"Project.pdf describes it as an 'AI-Driven Validation Multiplier' — i.e., itself the output of a model, not a static input. It is unclear whether F_valid should be (a) an external input, (b) a co-predicted output, or (c) a value the E-PINN should learn to reproduce... Flagged for resolution before implementation."*
- **The Circularity Hazard:** If $F_{\text{valid}}$ is fed into the E-PINN as an input, and the E-PINN evaluates whether $F_{\text{multiplier}} = F_{\text{valid}} \cdot F_{\text{SME}} \cdot F_{\text{perm}} \cdot \alpha$, while the validation system adjusts $F_{\text{valid}}$ based on the E-PINN's own residuals, a closed causal loop is established:
  $$F_{\text{valid}} \longrightarrow \text{E-PINN Forward Pass} \longrightarrow \text{Residuals} \longrightarrow \text{Validation Assessment} \longrightarrow F_{\text{valid}}$$
  This creates **validation leakage**, allowing a corrupted input to self-justify by adjusting the validation multiplier upstream.

### 2.2 Evaluation of Candidate Roles
1. **Role A: External Input (REJECTED)**: Promotes circular reasoning and allows upstream telemetry to dictate its own verification score.
2. **Role B: Supervised Training Target (REJECTED)**: Requires external third-party labels for every training timestamp, which do not exist for novel facilities.
3. **Role C: Co-Predicted E-PINN Output (FLAGGED / RESTRICTED)**: A neural network cannot impartially judge the validity of its own reconstructions without an independent reference.
4. **Role D: Post-E-PINN Validation-Derived Multiplier (RECOMMENDED & ADOPTED)**:
   - The E-PINN takes raw physical telemetry ($X_{\text{core}} \in \mathbb{R}^{26}$).
   - The E-PINN outputs physical reconstructions $\hat{Y}$.
   - The **Scientific Validation Engine** independently inspects the 14 physics residuals and 23 constraint violations.
   - $F_{\text{valid}}$ is calculated deterministically as a continuous function of residual quality and data hygiene:
     $$F_{\text{valid}} = \text{DataHygieneScore} \times \exp\left( -\kappa \cdot \max(0, \bar{r}_{\text{norm}} - 1.0) \right) \times \prod_{c} \mathbf{1}_{g_c \le \text{tol}}$$
   - This calculated $F_{\text{valid}} \in [0, 1]$ is then passed to the deterministic credit engine.

### 2.3 Implementation Decision
- **Original specification:** Listed as an input while named an "AI-Driven Validation Multiplier".
- **Decision:** Remove $F_{\text{valid}}$ from the E-PINN input vector. Establish $F_{\text{valid}}$ as the **authoritative output of the Post-E-PINN Scientific Validation Engine**.
- **Implementation Consequence:** Eliminates all validation leakage. The neural network cannot manipulate its own validation multiplier.

---

## Issue 3: Representation of Indexed Variables ($\sum_i, \sum_k$)

### 3.1 Statement of the Problem
The governing equations explicitly use summations over multiple fuels ($i$) and transport legs ($k$):
$$E_{\text{actual}} = \sum_i \left( FC_{\text{actual},i} \times NCV_i \times EF_{\text{CO2},i} \times OF_i \right) + (EC_{\text{actual}} \times EF_{\text{grid}})$$
$$L_{\text{transport}} = \sum_k \left( D_k \times EF_{\text{vehicle},k} \times M_{\text{trans},k} \right)$$
In contrast, a flat CSV row provides scalar columns: `FC_actual`, `NCV`, `EF_CO2`, `OF`, `D_k`, `EF_vehicle_k`, `M_trans_k`.

### 3.2 Evaluation of Structural Options
1. **Variable-Length Lists / Nested DataFrames:** Strongly incompatible with standard batched PyTorch tensor architectures ($B \times D$).
2. **Fixed Supported Cardinality (Padded Slots):** e.g., `FC_actual_1`, `FC_actual_2`, `FC_actual_3`. Leads to sparse inputs and arbitrary hard limits on fuel sources.
3. **Pre-Aggregated Weighted Intensity (RECOMMENDED FOR TABULAR STREAM):**
   - For single-fuel facilities: $i=1$, matching the scalar fields directly.
   - For multi-fuel facilities: The ingestion layer computes the effective fuel energy and emission contribution prior to model entry, or sums sub-records in an iteration batch.
   - For transport legs: $L_{\text{transport}} = \sum_k (D_k \cdot EF_k \cdot M_k)$ is computed as the total ton-km transport emission across all legs in that reporting cycle.

### 3.3 Implementation Decision
- **Decision:** Support both primary scalar mode ($i=1, k=1$ default for standard continuous telemetry) and a multi-leg / multi-fuel aggregation interface in `DataLoader`. The model input vector receives the effective primary/aggregated parameters, preserving exact mathematical summation semantics without tensor dimension explosion.

---

## Issue 4: $PE_{\text{aux}}$ Dimensional Inconsistency

### 4.1 Statement of the Problem
Section C, Equation 9 states verbatim:
$$PE_{\text{aux}} = EC_{\text{hardware}} - EF_{\text{grid}}$$
- $EC_{\text{hardware}} = (P_{\text{fan}} + P_{\text{pump}}) \times (t_{\text{op}} / 3600)$ has units of **energy** ($\text{kWh}$).
- $EF_{\text{grid}}$ has units of **emission factor** ($\text{tCO}_2/\text{kWh}$).
- **Dimensional Failure:** Subtracting an emission rate per unit energy ($\text{tCO}_2/\text{kWh}$) from total electrical energy ($\text{kWh}$) is physically meaningless ($\text{kWh} - \text{tCO}_2/\text{kWh}$).

### 4.2 Reconciled Dual-Mode Formulation
Per project instructions, the supplied equation must NOT be silently overwritten or modified without explicit traceability.

1. **`SPECIFICATION_MODE` (Verbatim Contract)**:
   $$PE_{\text{aux}} = EC_{\text{hardware}} - EF_{\text{grid}}$$
   - Preserves exact formula for traceability and regression against the supplied specification document.
   - Emits an explicit audit warning in logs and reports: `DIMENSIONAL_INCONSISTENCY_WARNING: PE_aux evaluated using verbatim subtraction [kWh - tCO2/kWh].`
2. **`PHYSICAL_MODE` (Dimensionally Consistent Formulation)**:
   $$PE_{\text{aux}} = EC_{\text{hardware}} \times EF_{\text{grid}}$$
   - Multiplies auxiliary electricity consumption ($\text{kWh}$) by the grid emission factor ($\text{tCO}_2/\text{kWh}$), yielding actual auxiliary Scope 2 emissions in metric tonnes ($\text{tCO}_2$).
   - Validated by standard GHG Protocol and ISO 14064 methodology.

### 4.3 Implementation Decision
- Both modes are implemented in [equations.py](file:///c:/Users/lenovo/Desktop/CUBEX/src/carbon_capture/physics/equations.py#L90-L105).
- The active mode is explicitly selected via `configs/physics.yaml` (`pe_aux_mode: "verbatim"` vs `"physical"`).
- All generated reports explicitly state the active mode and its dimensional status.

---

## Issue 5: Exact Residual Count and Classification

### 5.1 Statement of the Problem
The text notes: *"L_physics contains all 14 Section D residuals"*, yet 15 residuals are named across the documentation:
1. $r_{\text{actual}}$
2. $r_{\text{baseline}}$
3. $r_{\text{reduced}}$
4. $r_{\text{removed}}$
5. $r_{\text{disp\_chem}}$
6. $r_{\text{disp\_heat}}$
7. $r_{\text{displaced}}$
8. $r_{\text{EChw}}$
9. $r_{\text{aux}}$
10. $r_{\text{lifecycle}}$
11. $r_{\text{slip}}$
12. $r_{\text{transport}}$
13. $r_{\text{leakage}}$
14. $r_{\text{mult}}$
15. $r_{\text{CC}}$

### 5.2 Mathematical Reconciliation
By analyzing Document 1 Sections D, E, F, G:
- **Section D specifies exactly 14 equality residuals**:
  - **11 First-Principles Residuals:**
    $r_{\text{actual}}, r_{\text{baseline}}, r_{\text{reduced}}, r_{\text{removed}}, r_{\text{disp\_chem}}, r_{\text{disp\_heat}}, r_{\text{EChw}}, r_{\text{aux}}, r_{\text{lifecycle}}, r_{\text{slip}}, r_{\text{transport}}$.
  - **3 Compositional Consistency Residuals:**
    $r_{\text{displaced}}$ ($E_{\text{displaced}} - [E_{\text{disp,chem}} + E_{\text{disp,heat}}]$),
    $r_{\text{leakage}}$ ($L_{\text{leakage}} - [L_{\text{slip}} + L_{\text{transport}}]$),
    $r_{\text{mult}}$ ($F_{\text{multiplier}} - [F_{\text{valid}} \cdot F_{\text{SME}} \cdot F_{\text{perm}} \cdot \alpha]$).
  - Total in Section D = **14 residuals**.
- **Section G specifies the 15th residual ($r_{\text{CC}}$)**:
  - $r_{\text{CC}} = \hat{CC}_T - [(\hat{E}_{\text{red}} + \hat{E}_{\text{rem}} + \hat{E}_{\text{disp}} - \hat{PE}_{\text{aux}} - \hat{PE}_{\text{life}} - \hat{L}_{\text{leak}}) \cdot \hat{F}_{\text{mult}}]$.
  - Section G and Section O explicitly isolate $r_{\text{CC}}$ into its own independently weighted loss term:
    $$L_{\text{total}} = L_{\text{data}} + L_{\text{physics}} + L_{\text{consistency}} + w_{CC} \cdot L_{CC} + L_{\text{constraints}}$$
- **Decision:** The residual count is **15 total residuals**:
  - 11 first-principles residuals $\rightarrow L_{\text{physics}}$
  - 3 compositional residuals $\rightarrow L_{\text{consistency}}$
  - 1 top-level accounting consistency residual $\rightarrow L_{CC}$

---

## Issue 6: Principled Formulation of $w_{CC}$

### 6.1 Statement of the Problem
In the loss objective:
$$L_{\text{total}} = L_{\text{data}} + L_{\text{physics}} + L_{\text{consistency}} + w_{CC} \cdot \tilde{r}_{\text{CC}}^2 + L_{\text{constraints}}$$
What determines $w_{CC}$?

### 6.2 Mathematical Derivation
Per Document 1 Section G and O:
- $r_{\text{CC}}$ is an algebraic consequence of the intermediate residuals at the global optimum ($r \to 0$).
- If $w_{CC}$ is too large, the network minimizes $CC_T$ by allowing internal errors to cancel out inside the bracket (5-dimensional degenerate null space, Section K).
- If $w_{CC} = 0$, there is no top-level supervision anchoring the multiplier product to the bracket.
- **Normalization Strategy:**
  $$\tilde{r}_{\text{CC}} = \frac{r_{\text{CC}}}{s_{\text{CC}}}$$
  where $s_{\text{CC}} = \text{Std}(CC_T)$ computed over the training set.
- Under characteristic-scale normalization, $\tilde{r}_{\text{CC}}$ is dimensionless with expected magnitude $\sim 1.0$, exactly matching the normalized physics residuals $\tilde{r}_q$.
- **Principled Weight:** Setting **$w_{CC} = 1.0$** preserves equal scale with the physics loss without dominating internal identifiability.

---

## Issue 7: Taxonomy of Sensor & Process Telemetry Variables

### 7.1 Separation into Functional Categories
Document 3 lists process engineering variables from scrubber operations. These must NOT be blindly combined into the carbon-credit E-PINN input vector.

| Variable | Category | Functional Role | Destination Component |
|---|---|---|---|
| `Q_gas` (actual gas flow rate, $\text{m}^3/\text{s}$) | Process / Hydraulic Telemetry | Governs scrubber bed fluid dynamics | Q-Learning State / Thermodynamic Model |
| `delta_P_scrub` (bed pressure drop, Pa) | Process / Hydraulic Telemetry | Determines fan hydraulic resistance | Thermodynamic Power Model |
| `Q_liq` (liquid circulation flow, $\text{m}^3/\text{s}$) | Process / Hydraulic Telemetry | Solvent wetting flow rate | Thermodynamic Power Model |
| `rho_liq` (solvent density, $\text{kg}/\text{m}^3$) | Thermodynamic Fluid Property | Used in pump head calculations | Thermodynamic Property Engine |
| `H_pump` (dynamic head, m) | Hardware Spec / Telemetry | Required for solvent recirculation | Thermodynamic Power Model |
| `eta_fan` (fan motor efficiency) | Hardware Equipment Spec | Fan electrical conversion efficiency | Physics Calculation of $P_{\text{fan}}$ |
| `eta_pump` (pump motor efficiency) | Hardware Equipment Spec | Pump electrical conversion efficiency | Physics Calculation of $P_{\text{pump}}$ |
| `T_flue_gas` (gas temperature, °C/K) | Process State | Affects reaction kinetics and Peng-Robinson $Z$ | Q-Learning State & EOS |
| `CO2_inlet_conc` (%) | Process State | Capture bed inlet concentration | Q-Learning State |
| `CO2_outlet_conc` (%) | Process State | Scrubber outlet slip concentration | Q-Learning State & Slippage Check |
| `fan_speed_rpm` (RPM) | Actuator State | Controllable operational speed (1000–2000 RPM) | Q-Learning State & Action |

- **Architectural Boundary:**
  - $P_{\text{fan}}$ and $P_{\text{pump}}$ are the **direct electrical power inputs** for the E-PINN and carbon accounting.
  - The hydraulic/sensor variables ($Q_{\text{gas}}, \Delta P_{\text{scrub}}, Q_{\text{liq}}, \rho_{\text{liq}}, H_{\text{pump}}, \eta_{\text{fan}}, \eta_{\text{pump}}$) live in `src/carbon_capture/physics/thermodynamics.py` to calculate or validate $P_{\text{fan}}$ and $P_{\text{pump}}$ when power meters are unmetered or suspect.
  - Process variables ($T_{\text{flue\_gas}}, \text{CO}_{2,\text{inlet}}, \text{CO}_{2,\text{outlet}}, \text{fan\_speed\_rpm}$) belong exclusively to `src/carbon_capture/rl/` for closed-loop control.

---

## Issue 8: Definitive Unit Harmonization System

### 8.1 Explicit Unit Mapping
To prevent silent conversion errors, the pipeline establishes a **single internal canonical unit system**:

| Measurement Domain | Input / Display Unit | Internal Canonical Unit | Output / Reporting Unit | Conversion Factor to Canonical |
|---|---|---|---|---|
| Fuel Consumption | $\text{m}^3$ or kg | **metric tonnes (t)** | metric tonnes (t) | $\times 10^{-3}$ (from kg) or $\rho_{\text{fuel}} \times 10^{-3}$ |
| Energy (Net Calorific) | $\text{J}/\text{kg}$ or $\text{kJ}/\text{kg}$ | **$\text{MJ}/\text{kg}$** | $\text{MJ}/\text{kg}$ | $\times 10^{-6}$ (from J/kg) |
| Electricity Consumption | MWh or Wh | **$\text{kWh}$** | $\text{kWh}$ | $\times 10^3$ (from MWh) |
| Grid Emission Factor | $\text{gCO}_2/\text{kWh}$ or $\text{kgCO}_2/\text{MWh}$ | **$\text{tCO}_2/\text{kWh}$** | $\text{tCO}_2/\text{kWh}$ | $\times 10^{-6}$ (from $\text{gCO}_2/\text{kWh}$) |
| Captured Mass | kg | **metric tonnes (t)** | metric tonnes (t) | $\times 10^{-3}$ |
| Byproduct Mass | kg | **metric tonnes (t)** | metric tonnes (t) | $\times 10^{-3}$ |
| Sorbent Makeup Mass | kg | **metric tonnes (t)** | metric tonnes (t) | $\times 10^{-3}$ |
| Thermal Heat Recovered | GJ or kcal | **MJ** | MJ | $\times 10^3$ (from GJ) |
| Auxiliary Motor Power | W or MW | **kW** | kW | $\times 10^{-3}$ (from W) |
| Operating Duration | hours or minutes | **seconds (s)** | seconds (s) | $\times 3600$ (from hours) |
| Transport Distance | miles | **km** | km | $\times 1.60934$ |
| Transport Factor | $\text{gCO}_2/\text{t-km}$ | **$\text{tCO}_2\text{e}/\text{t-km}$** | $\text{tCO}_2\text{e}/\text{t-km}$ | $\times 10^{-6}$ |
| Carbon Credits | kg or lbs | **$\text{tCO}_2\text{e}$** | metric tonnes $\text{CO}_2\text{e}$ | 1.0 |

All conversions occur strictly within `src/carbon_capture/physics/units.py`.

---

## Issue 9: Observed vs. Derived vs. Predicted Classification (Target Leakage Prevention)

To prevent circularity or trivial learning, every variable is explicitly classified:

| Variable | Classification | Description | May be Model Input? |
|---|---|---|---|
| `FC_actual`, `FC_baseline`, `NCV`, `EF_CO2`, `OF` | **OBSERVED** | Metered telemetry & fuel fuel specs | **YES** |
| `EC_actual`, `EC_baseline`, `EF_grid` | **OBSERVED** | Utility power meter & grid factors | **YES** |
| `M_captured`, `P_CO2`, `eta_purity` | **OBSERVED** | Weight load cells & lab assays | **YES** |
| `M_byproduct`, `EF_virgin_displace` | **OBSERVED** | Scale tickets & LCA database | **YES** |
| `H_recovered`, `eta_boiler`, `NCV_fuel`, `EF_CO2_fuel` | **OBSERVED** | Heat meters & boiler specs | **YES** |
| `P_fan`, `P_pump`, `t_op` | **OBSERVED** | Hardware power sensors & run clocks | **YES** |
| `M_solvent_makeup`, `EF_solvent_LCA` | **OBSERVED** | Chemical warehouse logs & LCA factors | **YES** |
| `gamma_slip`, `D_k`, `EF_vehicle_k`, `M_trans_k` | **OBSERVED** | Flow diff meters & logistics logs | **YES** |
| `E_actual`, `E_baseline`, `E_reduced` | **DERIVED / PREDICTED** | Intermediate emissions | **NO** (Predicted/Target) |
| `E_removed`, `E_displaced_chem`, `E_displaced_heat`, `E_displaced` | **DERIVED / PREDICTED** | Intermediate capture/displacement | **NO** (Predicted/Target) |
| `EC_hardware`, `PE_aux`, `PE_lifecycle` | **DERIVED / PREDICTED** | Intermediate auxiliary penalties | **NO** (Predicted/Target) |
| `L_slip`, `L_transport`, `L_leakage` | **DERIVED / PREDICTED** | Intermediate leakage penalties | **NO** (Predicted/Target) |
| `F_multiplier` | **DERIVED / PREDICTED** | Multiplier product | **NO** (Predicted/Target) |
| `CC_T` | **DERIVED / PREDICTED** | Net carbon credit volume | **NO** (Authoritative Target) |
| `F_valid` | **VALIDATION OUTPUT** | Physics/data integrity metric | **NO** (Validation System Output) |
| `F_SME`, `F_perm`, `alpha` | **POLICY CONFIG** | Legal/administrative discounts | Optional context only |

---

## Issue 10: Independence of Deterministic Carbon-Credit Engine

- **Strict Boundary:** The deterministic carbon-credit engine is completely independent of PyTorch, NumPy gradient tracking, or neural network weights.
- **Reference Calculation Path:** The deterministic calculator can be run directly on raw or validated telemetry without instantiating or loading any neural network.
- **Authority:** If the E-PINN predicts $\hat{CC}_T = 12,500$ but the deterministic calculator evaluates $CC_T = 12,480$, the authoritative certified credit is **$12,480\text{ tCO}_2\text{e}$**. The discrepancy ($20\text{ tCO}_2\text{e}$) is recorded in audit logs as a verification residual $r_{\text{CC}}$.

---

## Issue 11: Temporal Modeling vs. Point/Batch Formulation

- **Document Analysis:** Document 1 Section F and Document 2 Phase 8 introduce iteration-level and aggregation residuals ($L_{\text{iter}}$) for project cycles. However, the governing equations in Section C are strictly static/algebraic point relations for any single reporting period $t$.
- **Risk of Premature Temporal Complexity:** Imposing an LSTM or Transformer on single-step algebraic equations introduces spurious temporal autocorrelation assumptions, memory overhead, and unidentifiable dynamics.
- **Roadmap:**
  - **Base E-PINN (Current Phase):** Point/batch multi-head architecture evaluating algebraic conservation laws on each measurement period.
  - **Temporal Extension (Phase 8):** Sequence builder module evaluates multi-iteration trajectory consistency ($L_{\text{iter}}$) as an outer aggregation check without altering the core physics head.

---

## Issue 12: Generative Graph for Physically Grounded Synthetic Data

To prevent training on unphysical, uncorrelated random numbers, the synthetic data generator must respect the following causal dependency graph:

```
[Ambient / Process Demands]
        │
        ├──> Fuel Consumption Baseline: FC_baseline ~ Normal(mu_base, sigma)
        │         │
        │         └──> Fuel Consumption Actual: FC_actual = FC_baseline - EfficiencyGain
        │
        ├──> Electricity Baseline: EC_baseline
        │         │
        │         └──> Electricity Actual: EC_actual = EC_baseline - EfficiencyGain
        │
        ├──> Raw Flue Gas Volume & CO2 Concentration
        │         │
        │         ├──> Captured Mass: M_captured = Q_gas * rho_CO2 * [CO2_in - CO2_out]
        │         │         │
        │         │         ├──> Product Mass: M_byproduct = M_captured * (1 - Slip)
        │         │         └──> Transport Mass: M_trans_k = M_byproduct
        │         │
        │         ├──> Scrubber Pressure Drop & Flow
        │         │         │
        │         │         ├──> Fan Power: P_fan = (Q_gas * Delta_P) / eta_fan
        │         │         └──> Pump Power: P_pump = (rho_liq * g * Q_liq * H) / eta_pump
        │         │
        │         └──> Exothermic Reaction Heat: H_recovered = M_captured * Delta_H_rxn
        │
        └──> Solvent Degradation: M_solvent_makeup = M_captured * DegradationRate
```

---

## Summary of Reconciled Mathematical Contract
All 12 issues have been isolated, documented, and given rigorous scientific resolutions. The detailed reconciled schema, equations, and architecture are codified in the companion documents:
- `docs/RECONCILED_SCHEMA.md`
- `docs/RECONCILED_EQUATIONS.md`
- `docs/RECONCILED_ARCHITECTURE.md`
