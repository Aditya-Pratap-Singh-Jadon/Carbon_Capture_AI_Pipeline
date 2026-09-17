"""Authoritative Canonical Schema and Input Orderings (Frozen Reconciliation Contract).

Derived from docs/RECONCILED_SCHEMA.md.
- Core E-PINN Input: X_core in R^26 (strictly physical and operational telemetry)
- Policy Parameters: [F_SME, F_perm, alpha] in R^3 (administrative crediting factors)
- F_valid is strictly an OUTPUT of VerificationEngine, never an input.
"""

from typing import Dict, List, Tuple

# 1. Core Physical Telemetry Vector (Input to E-PINN: X_core in R^26)
CORE_PHYSICAL_INPUT_ORDER: List[str] = [
    # Fuel and Combustion
    "FC_actual",          # 0: Actual fuel consumption (tonnes)
    "FC_baseline",        # 1: Baseline fuel consumption (tonnes)
    "NCV",                # 2: Net Calorific Value of fuel (MJ/kg)
    "EF_CO2",             # 3: Fuel CO2 emission factor (tCO2/t)
    "OF",                 # 4: Oxidation factor (dimensionless [0, 1])

    # Scope 2 Facility Electricity
    "EC_actual",          # 5: Actual electricity consumption (kWh)
    "EC_baseline",        # 6: Baseline electricity consumption (kWh)
    "EF_grid",            # 7: Grid electricity emission factor (tCO2/kWh)

    # Capture and Chemical Synthesis
    "M_captured",         # 8: Captured / synthesized product mass (tonnes)
    "P_CO2",              # 9: Stoichiometric carbon mass fraction ([0, 1])
    "eta_purity",         # 10: Chemical purity fraction ([0, 1])

    # Circular Byproducts and Waste Heat Recovery
    "M_byproduct",        # 11: Displaced circular byproduct mass (tonnes)
    "EF_virgin_displace", # 12: Virgin chemical avoidance factor (tCO2e/t)
    "H_recovered",        # 13: Reclaimed waste/exothermic heat (MJ)
    "eta_boiler",         # 14: Boiler efficiency ([0, 1])
    "NCV_fuel",           # 15: Net calorific value of displaced fuel (MJ/kg)
    "EF_CO2_fuel",        # 16: Emission factor of displaced fuel (tCO2/MJ)

    # Capture Hardware Telemetry
    "P_fan",              # 17: Scrubber auxiliary fan power (kW)
    "P_pump",             # 18: Solvent recirculation pump power (kW)
    "t_op",               # 19: Hardware active operating duration (seconds)

    # Solvent Degradation & Lifecycle
    "M_solvent_makeup",   # 20: Solvent replenishment mass (tonnes)
    "EF_solvent_LCA",     # 21: Solvent cradle-to-gate carbon intensity (tCO2e/t)

    # Slippage and Logistics
    "gamma_slip",         # 22: Carbon slippage fraction ([0, 1])
    "D_k",                # 23: Transit distance (km)
    "EF_vehicle_k",       # 24: Freight vehicle emission factor (tCO2e/t-km)
    "M_trans_k",          # 25: Transported cargo mass (tonnes)
]

# 2. Administrative Policy Context Factors (R^3)
POLICY_PARAMETER_ORDER: List[str] = [
    "F_SME",              # 26: Enterprise scale factor (>= 1.0)
    "F_perm",             # 27: Containment permanence factor ([0, 1])
    "alpha",              # 28: Precision conservatism discount ([0, 1])
]

# Full Tabular Dataset Order (29 columns in CSV files)
CANONICAL_TABULAR_ORDER: List[str] = CORE_PHYSICAL_INPUT_ORDER + POLICY_PARAMETER_ORDER

# Alias for backward compatibility
CANONICAL_INPUT_ORDER = CANONICAL_TABULAR_ORDER

# Column aliases for resilient CSV ingestion
COLUMN_ALIASES: Dict[str, str] = {
    "fc_actual": "FC_actual",
    "fcactual": "FC_actual",
    "fc_baseline": "FC_baseline",
    "fcbaseline": "FC_baseline",
    "ncv": "NCV",
    "ef_co2": "EF_CO2",
    "efco2": "EF_CO2",
    "of": "OF",
    "fof": "OF",
    "ec_actual": "EC_actual",
    "ecactual": "EC_actual",
    "ec_baseline": "EC_baseline",
    "ecbaseline": "EC_baseline",
    "ef_grid": "EF_grid",
    "efgrid": "EF_grid",
    "m_captured": "M_captured",
    "mcaptured": "M_captured",
    "p_co2": "P_CO2",
    "pco2": "P_CO2",
    "eta_purity": "eta_purity",
    "n_purity": "eta_purity",
    "purity": "eta_purity",
    "m_byproduct": "M_byproduct",
    "mbyproduct": "M_byproduct",
    "ef_virgin_displace": "EF_virgin_displace",
    "efvirgindisplace": "EF_virgin_displace",
    "h_recovered": "H_recovered",
    "hrecovered": "H_recovered",
    "eta_boiler": "eta_boiler",
    "n_boiler": "eta_boiler",
    "boiler_efficiency": "eta_boiler",
    "ncv_fuel": "NCV_fuel",
    "ef_co2_fuel": "EF_CO2_fuel",
    "p_fan": "P_fan",
    "p_pump": "P_pump",
    "t_op": "t_op",
    "top": "t_op",
    "m_solvent_makeup": "M_solvent_makeup",
    "msolvent_makeup": "M_solvent_makeup",
    "ef_solvent_lca": "EF_solvent_LCA",
    "efsolvent_lca": "EF_solvent_LCA",
    "gamma_slip": "gamma_slip",
    "gammaslip": "gamma_slip",
    "slip": "gamma_slip",
    "d_k": "D_k",
    "dk": "D_k",
    "distance": "D_k",
    "ef_vehicle_k": "EF_vehicle_k",
    "efvehicle": "EF_vehicle_k",
    "m_trans_k": "M_trans_k",
    "mtrans": "M_trans_k",
    "f_sme": "F_SME",
    "fsme": "F_SME",
    "f_perm": "F_perm",
    "fperm": "F_perm",
    "alpha": "alpha",
    "a": "alpha",
}

# Domain bounds for data quality validation
BOUNDS: Dict[str, Tuple[float, float]] = {
    "FC_actual": (0.0, 1.0e7),
    "FC_baseline": (0.0, 1.0e7),
    "NCV": (0.0, 200.0),
    "EF_CO2": (0.0, 10.0),
    "OF": (0.0, 1.0),
    "EC_actual": (0.0, 1.0e8),
    "EC_baseline": (0.0, 1.0e8),
    "EF_grid": (0.0, 2.0),
    "M_captured": (0.0, 1.0e6),
    "P_CO2": (0.0, 1.0),
    "eta_purity": (0.0, 1.0),
    "M_byproduct": (0.0, 1.0e6),
    "EF_virgin_displace": (0.0, 20.0),
    "H_recovered": (0.0, 1.0e9),
    "eta_boiler": (0.0, 1.0),
    "NCV_fuel": (0.0, 200.0),
    "EF_CO2_fuel": (0.0, 10.0),
    "P_fan": (0.0, 1.0e4),
    "P_pump": (0.0, 1.0e4),
    "t_op": (0.0, 86400.0 * 365),
    "M_solvent_makeup": (0.0, 1.0e5),
    "EF_solvent_LCA": (0.0, 50.0),
    "gamma_slip": (0.0, 1.0),
    "D_k": (0.0, 50000.0),
    "EF_vehicle_k": (0.0, 5.0),
    "M_trans_k": (0.0, 1.0e6),
    "F_SME": (1.0, 10.0),
    "F_perm": (0.0, 1.0),
    "alpha": (0.0, 1.0),
}
