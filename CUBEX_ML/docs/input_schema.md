# Canonical Input Schema & Ordering

## 1. Principle of Canonical Ordering
The model and preprocessing pipelines strictly require inputs in **one immutable canonical order** ($X \in \mathbb{R}^{30}$). Regardless of column order in incoming CSV files, the `DataLoader` identifies columns via case-insensitive matching and aliases, validating and reordering them into canonical order prior to scaling or model ingestion.

## 2. Canonical Column Specification (30 Variables)
| Index | Variable Name | Physical Meaning | Canonical Unit | Admissible Range |
|-------|---------------|------------------|----------------|------------------|
| 0 | `FC_actual` | Actual fuel consumption | tonnes or m³ | $\ge 0$ |
| 1 | `FC_baseline` | Baseline fuel consumption | tonnes or m³ | $\ge 0$ |
| 2 | `NCV` | Net calorific value | MJ/kg or MJ/m³ | $> 0$ |
| 3 | `EF_CO2` | Fuel CO₂ emission factor | tCO₂/t fuel | $\ge 0$ |
| 4 | `OF` | Fuel oxidation factor | dimensionless | $[0, 1]$ |
| 5 | `EC_actual` | Actual electricity consumption | kWh | $\ge 0$ |
| 6 | `EC_baseline` | Baseline electricity consumption | kWh | $\ge 0$ |
| 7 | `EF_grid` | Grid emission factor | tCO₂/kWh | $\ge 0$ |
| 8 | `M_captured` | Captured/synthesized product mass | tonnes | $\ge 0$ |
| 9 | `P_CO2` | Stoichiometric carbon ratio | dimensionless | $[0, 1]$ |
| 10 | `eta_purity` | Chemical product purity | dimensionless | $[0, 1]$ |
| 11 | `M_byproduct` | Displaced chemical byproduct mass | tonnes | $\ge 0$ |
| 12 | `EF_virgin_displace`| Virgin chemical avoidance factor | tCO₂e/t | $\ge 0$ |
| 13 | `H_recovered` | Reclaimed exothermic/waste heat | MJ | $\ge 0$ |
| 14 | `eta_boiler` | Boiler efficiency | dimensionless | $[0, 1]$ |
| 15 | `NCV_fuel` | Net calorific value of displaced fuel | MJ/kg | $> 0$ |
| 16 | `EF_CO2_fuel` | Boiler fuel emission factor | tCO₂/MJ | $\ge 0$ |
| 17 | `P_fan` | Auxiliary fan power | kW | $\ge 0$ |
| 18 | `P_pump` | Auxiliary pump power | kW | $\ge 0$ |
| 19 | `t_op` | Operating duration | seconds | $\ge 0$ |
| 20 | `M_solvent_makeup` | Solvent replenishment mass | tonnes | $\ge 0$ |
| 21 | `EF_solvent_LCA` | Solvent lifecycle carbon intensity | tCO₂e/t | $\ge 0$ |
| 22 | `gamma_slip` | Gas/liquid carbon slippage fraction | dimensionless | $[0, 1]$ |
| 23 | `D_k` | Transport transit distance | km | $\ge 0$ |
| 24 | `EF_vehicle_k` | Vehicle emission factor | tCO₂e/t-km | $\ge 0$ |
| 25 | `M_trans_k` | Transported product mass | tonnes | $\ge 0$ |
| 26 | `F_valid` | AI validation multiplier | dimensionless | $[0, 1]$ |
| 27 | `F_SME` | Enterprise scale factor | dimensionless | $\ge 1.0$ |
| 28 | `F_perm` | Permanence factor | dimensionless | $[0, 1]$ |
| 29 | `alpha` | Precision conservatism discount | dimensionless | $[0, 1]$ |
