"""Authoritative Governing Equations (verbatim, unmodified, Section C).

All equations from Document 1 Section C are implemented here in closed form.
Documented ambiguities (e.g. PE_aux dimensional inconsistency) are preserved
verbatim as default, while providing an inspectable configurable physical mode.
"""

from typing import Dict, Union
import numpy as np
from carbon_capture.utils.logging import get_logger

logger = get_logger(__name__)

def calc_e_actual(
    fc_actual: Union[float, np.ndarray],
    ncv: Union[float, np.ndarray],
    ef_co2: Union[float, np.ndarray],
    of: Union[float, np.ndarray],
    ec_actual: Union[float, np.ndarray],
    ef_grid: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """E_actual = (FC_actual * NCV * EF_CO2 * OF) + (EC_actual * EF_grid)"""
    return (fc_actual * ncv * ef_co2 * of) + (ec_actual * ef_grid)

def calc_e_baseline(
    fc_baseline: Union[float, np.ndarray],
    ncv: Union[float, np.ndarray],
    ef_co2: Union[float, np.ndarray],
    of: Union[float, np.ndarray],
    ec_baseline: Union[float, np.ndarray],
    ef_grid: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """E_baseline = (FC_baseline * NCV * EF_CO2 * OF) + (EC_baseline * EF_grid)"""
    return (fc_baseline * ncv * ef_co2 * of) + (ec_baseline * ef_grid)

def calc_e_reduced(
    e_baseline: Union[float, np.ndarray],
    e_actual: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """E_reduced = max(0, E_baseline - E_actual)"""
    diff = e_baseline - e_actual
    if hasattr(diff, "__iter__") or isinstance(diff, np.ndarray):
        return np.maximum(0.0, diff)
    return max(0.0, float(diff))

def calc_e_removed(
    m_captured: Union[float, np.ndarray],
    p_co2: Union[float, np.ndarray],
    eta_purity: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """E_removed = M_captured * P_CO2 * eta_purity"""
    return m_captured * p_co2 * eta_purity

def calc_e_displaced_chem(
    m_byproduct: Union[float, np.ndarray],
    ef_virgin_displace: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """E_displaced,chemical = M_byproduct * EF_virgin_displace"""
    return m_byproduct * ef_virgin_displace

def calc_e_displaced_heat(
    h_recovered: Union[float, np.ndarray],
    eta_boiler: Union[float, np.ndarray],
    ncv_fuel: Union[float, np.ndarray],
    ef_co2_fuel: Union[float, np.ndarray],
    eps: float = 1e-8,
) -> Union[float, np.ndarray]:
    """E_displaced,heat = (H_recovered / (eta_boiler * NCV_fuel)) * EF_CO2,fuel"""
    denom = eta_boiler * ncv_fuel
    if hasattr(denom, "__iter__") or isinstance(denom, np.ndarray):
        denom = np.maximum(denom, eps)
    else:
        denom = max(float(denom), eps)
    return (h_recovered / denom) * ef_co2_fuel

def calc_e_displaced(
    e_displaced_chem: Union[float, np.ndarray],
    e_displaced_heat: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """E_displaced = E_displaced,chemical + E_displaced,heat"""
    return e_displaced_chem + e_displaced_heat

def calc_ec_hardware(
    p_fan: Union[float, np.ndarray],
    p_pump: Union[float, np.ndarray],
    t_op: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """EC_hardware = (P_fan + P_pump) * (t_op / 3600)"""
    return (p_fan + p_pump) * (t_op / 3600.0)

def calc_pe_aux(
    ec_hardware: Union[float, np.ndarray],
    ef_grid: Union[float, np.ndarray],
    mode: str = "verbatim",
) -> Union[float, np.ndarray]:
    """PE_aux calculation.
    
    WARNING: In Section C verbatim: PE_aux = EC_hardware - EF_grid
    This is dimensionally inconsistent as EC_hardware is energy (kWh) and
    EF_grid is an emission rate (tCO2/kWh).
    In 'physical' mode, it evaluates: EC_hardware * EF_grid.
    Default is 'verbatim' to preserve specification contract.
    """
    if mode == "physical":
        return ec_hardware * ef_grid
    return ec_hardware - ef_grid

def calc_pe_lifecycle(
    m_solvent_makeup: Union[float, np.ndarray],
    ef_solvent_lca: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """PE_lifecycle = M_solvent_makeup * EF_solvent_LCA"""
    return m_solvent_makeup * ef_solvent_lca

def calc_l_slip(
    m_captured: Union[float, np.ndarray],
    gamma_slip: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """L_slip = M_captured * gamma_slip"""
    return m_captured * gamma_slip

def calc_l_transport(
    d_k: Union[float, np.ndarray],
    ef_vehicle_k: Union[float, np.ndarray],
    m_trans_k: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """L_transport = D_k * EF_vehicle_k * M_trans_k"""
    return d_k * ef_vehicle_k * m_trans_k

def calc_l_leakage(
    l_slip: Union[float, np.ndarray],
    l_transport: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """L_leakage = L_slip + L_transport"""
    return l_slip + l_transport

def calc_f_multiplier(
    f_valid: Union[float, np.ndarray],
    f_sme: Union[float, np.ndarray],
    f_perm: Union[float, np.ndarray],
    alpha: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """F_multiplier = F_valid * F_SME * F_perm * alpha"""
    return f_valid * f_sme * f_perm * alpha

def calc_cc_t(
    e_reduced: Union[float, np.ndarray],
    e_removed: Union[float, np.ndarray],
    e_displaced: Union[float, np.ndarray],
    pe_aux: Union[float, np.ndarray],
    pe_lifecycle: Union[float, np.ndarray],
    l_leakage: Union[float, np.ndarray],
    f_multiplier: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """CC_T = [E_reduced + E_removed + E_displaced - PE_aux - PE_lifecycle - L_leakage] * F_multiplier"""
    bracket = e_reduced + e_removed + e_displaced - pe_aux - pe_lifecycle - l_leakage
    return bracket * f_multiplier
