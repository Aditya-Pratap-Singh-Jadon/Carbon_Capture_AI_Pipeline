# Industrial Carbon Capture E-PINN Verification & Carbon-Credit AI Pipeline

[![Tests](https://img.shields.io/badge/tests-21%20passed-brightgreen.svg)](#testing)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](#installation)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](#)

A production-grade, mathematically rigorous scientific AI pipeline for verifying industrial carbon capture telemetry and calculating authoritative carbon credits. Built upon the **Extended Physics-Informed Neural Network (E-PINN)** framework and closed-form deterministic carbon accounting.

---

## Key System Architecture

```
                    ┌──────────────────────────────┐
                    │    INDUSTRIAL SENSOR DATA    │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │     CANONICAL INPUT ORDER    │
                    │  (Enforced 30-Variable Map)  │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   DATA QUALITY & HYGIENE     │
                    │  (Range / NaN / Sanity Checks)
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │      E-PINN CORE ENGINE      │
                    │  - Multi-Head Reconstruction │
                    │  - 15 Conservation Residuals │
                    │  - Augmented Lagrangian Opt  │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ SCIENTIFIC VALIDATION ENGINE │
                    │ (Physics / Data / Bounds)    │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │  DETERMINISTIC CREDIT ENGINE │
                    │   Authoritative Calculation: │
                    │   CC_T = [Benefits-Penalties]│
                    │          x F_multiplier      │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ AUDIT TRAIL & VERIFICATION   │
                    │ - JSON Machine-Readable      │
                    │ - CSV Records                │
                    │ - HTML Visual Dossier        │
                    │ - PDF Regulatory Report      │
                    └──────────────────────────────┘
```

---

## 1. Mathematical Specifications & Core Formulation

The authoritative carbon-credit accounting equation is:
$$CC_T = [E_{\text{reduced}} + E_{\text{removed}} + E_{\text{displaced}} - PE_{\text{aux}} - PE_{\text{lifecycle}} - L_{\text{leakage}}] \times (F_{\text{valid}} \times F_{\text{SME}} \times F_{\text{perm}} \times \alpha)$$

### Core Architectural Separation
- **The E-PINN is the primary scientific validation engine**, reconstructing all intermediate physical states and evaluating governing equations.
- **The Deterministic Carbon Credit Engine is the sole authoritative calculation engine**. The neural network never directly outputs an unverified credit number.
- **Q-Learning** is retained as an independent process-control optimization loop for scrubber fan speed (1000–2000 RPM).
- **Statistical anomaly detection** (Isolation Forest, One-Class SVM) is kept strictly as optional non-authoritative diagnostics.

---

## 2. Canonical 30-Variable Input Order

Regardless of the original column order in incoming CSV files, the pipeline automatically detects, validates, and reorders all features into the canonical schema:

1. `FC_actual` - Actual fuel consumption (t or m³)
2. `FC_baseline` - Baseline fuel consumption (t or m³)
3. `NCV` - Net calorific value (MJ/kg or MJ/m³)
4. `EF_CO2` - Fuel CO₂ emission factor (tCO₂/t)
5. `OF` - Fuel oxidation factor ([0, 1])
6. `EC_actual` - Actual electricity consumption (kWh)
7. `EC_baseline` - Baseline electricity consumption (kWh)
8. `EF_grid` - Grid emission factor (tCO₂/kWh)
9. `M_captured` - Captured product mass (t)
10. `P_CO2` - Stoichiometric carbon ratio ([0, 1])
11. `eta_purity` - Product purity fraction ([0, 1])
12. `M_byproduct` - Displaced chemical byproduct mass (t)
13. `EF_virgin_displace` - Virgin chemical avoidance factor (tCO₂e/t)
14. `H_recovered` - Reclaimed exothermic/waste heat (MJ)
15. `eta_boiler` - Boiler efficiency ([0, 1])
16. `NCV_fuel` - Displaced boiler fuel net calorific value (MJ/kg)
17. `EF_CO2_fuel` - Boiler fuel CO₂ emission factor (tCO₂/MJ)
18. `P_fan` - Auxiliary fan power (kW)
19. `P_pump` - Auxiliary pump power (kW)
20. `t_op` - Operating duration (seconds)
21. `M_solvent_makeup` - Sorbent replenishment mass (t)
22. `EF_solvent_LCA` - Solvent lifecycle emission factor (tCO₂e/t)
23. `gamma_slip` - Gas/liquid carbon slippage fraction ([0, 1])
24. `D_k` - Transport transit distance (km)
25. `EF_vehicle_k` - Transport vehicle emission factor (tCO₂e/t-km)
26. `M_trans_k` - Transported mass (t)
27. `F_valid` - AI-driven validation multiplier ([0, 1])
28. `F_SME` - Enterprise scale factor ($\ge 1.0$)
29. `F_perm` - Containment permanence factor ([0, 1])
30. `alpha` - Precision conservatism discount ([0, 1])

---

## 3. Quick Start & Execution

### Installation
```bash
git clone <repo_url>
cd CUBEX

# 1. Create a virtual environment
python -m venv venv

# 2. Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# 3. Install dependencies
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 1. Generate Controlled Synthetic Datasets
Generates physically consistent steady-state, peak-load, turndown regimes, plus intentionally corrupted anomaly test sets:
```bash
python scripts/generate_data.py
```

### 2. Train the E-PINN Verification Engine
Trains the multi-head neural network using physics losses, uncertainty-normalized residuals, and dual-ascent augmented Lagrangian constraint optimization:
```bash
python scripts/train_epinn.py
```

### 3. Train the Q-Learning Process Controller
Trains tabular Q-learning for dynamic fan speed control:
```bash
python scripts/train_rl.py
```

### 4. Execute the End-to-End Verification Pipeline
Runs ingestion, canonical ordering, E-PINN physics validation, deterministic credit calculation, and generates PDF, HTML, JSON, and CSV reports:
```bash
python scripts/run_pipeline.py --input data/synthetic/test.csv --run-name industrial_clean_run
```

To test column-order resilience on a deliberately scrambled CSV:
```bash
python scripts/run_pipeline.py --input data/synthetic/test_shuffled_columns.csv --run-name industrial_shuffled_run
```

To test strict rejection of sensor corruption:
```bash
python scripts/run_pipeline.py --input data/synthetic/test_corrupted_purity.csv --run-name industrial_corrupted_run
```

---

## 4. Test Suite Execution

Run all 21 automated unit and integration tests:
```bash
python -m pytest tests/ -v
```

---

## 5. Documented Scientific Ambiguities and Traceability

Per specification instructions, all documented ambiguities are explicitly disclosed:
1. **$PE_{\text{aux}} = EC_{\text{hardware}} - EF_{\text{grid}}$ Dimensional Inconsistency**:
   $EC_{\text{hardware}}$ has units of energy (kWh), while $EF_{\text{grid}}$ has units of emission rate ($t\text{CO}_2/\text{kWh}$). Preserved verbatim as default per mathematical contract, with a configurable physical option ($EC_{\text{hardware}} \times EF_{\text{grid}}$) in `configs/physics.yaml`.
2. **Provenance of $F_{\text{valid}}$**:
   Documented as an AI-driven validation multiplier in $[0, 1]$. Handled modularly as observed telemetry input, co-prediction, or consistency check.
3. **$F_{\text{SME}} \ge 1.0$ One-Sided Bound**:
   Enforced as an inequality constraint without imposing an arbitrary upper bound.
