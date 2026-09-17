"""Physics, stoichiometry, thermodynamics, equations, and residuals."""

from carbon_capture.physics.units import convert, CONVERSIONS
from carbon_capture.physics.equations import (
    calc_e_actual,
    calc_e_baseline,
    calc_e_reduced,
    calc_e_removed,
    calc_e_displaced_chem,
    calc_e_displaced_heat,
    calc_e_displaced,
    calc_ec_hardware,
    calc_pe_aux,
    calc_pe_lifecycle,
    calc_l_slip,
    calc_l_transport,
    calc_l_leakage,
    calc_f_multiplier,
    calc_cc_t,
)
from carbon_capture.physics.stoichiometry import (
    get_stoichiometric_ratio,
    calculate_theoretical_yield,
    STOICHIOMETRIC_P_CO2,
)
from carbon_capture.physics.thermodynamics import (
    PengRobinsonEOS,
    calculate_scrubber_fan_power,
    calculate_solvent_pump_power,
)
from carbon_capture.physics.residuals import (
    compute_raw_residuals,
    normalize_residuals,
    ResidualReport,
)

__all__ = [
    "convert",
    "CONVERSIONS",
    "calc_e_actual",
    "calc_e_baseline",
    "calc_e_reduced",
    "calc_e_removed",
    "calc_e_displaced_chem",
    "calc_e_displaced_heat",
    "calc_e_displaced",
    "calc_ec_hardware",
    "calc_pe_aux",
    "calc_pe_lifecycle",
    "calc_l_slip",
    "calc_l_transport",
    "calc_l_leakage",
    "calc_f_multiplier",
    "calc_cc_t",
    "get_stoichiometric_ratio",
    "calculate_theoretical_yield",
    "STOICHIOMETRIC_P_CO2",
    "PengRobinsonEOS",
    "calculate_scrubber_fan_power",
    "calculate_solvent_pump_power",
    "compute_raw_residuals",
    "normalize_residuals",
    "ResidualReport",
]
