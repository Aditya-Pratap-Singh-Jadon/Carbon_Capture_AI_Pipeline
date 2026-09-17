"""Physics Residuals and Normalization (verbatim from Document 1 Sections D-G & I-J).

Calculates exact equality residuals between model predictions and governing equations.
Supports both raw residuals and principled normalized residuals.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Union
import numpy as np
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

@dataclass
class ResidualReport:
    raw_residuals: Dict[str, Union[float, np.ndarray]]
    normalized_residuals: Dict[str, Union[float, np.ndarray]]
    normalization_scales: Dict[str, float]
    max_normalized_violation: float
    is_valid_physics: bool

def compute_raw_residuals(
    preds: Dict[str, Union[float, np.ndarray]],
    inputs: Dict[str, Union[float, np.ndarray]],
    pe_aux_mode: str = "verbatim",
) -> Dict[str, Union[float, np.ndarray]]:
    """Compute all 15 raw physics residuals (Document 1, Sections D & G)."""
    residuals = {}

    # 1. r_actual
    eq_e_actual = calc_e_actual(
        inputs["FC_actual"], inputs["NCV"], inputs["EF_CO2"], inputs["OF"],
        inputs["EC_actual"], inputs["EF_grid"]
    )
    residuals["r_actual"] = preds["E_actual"] - eq_e_actual

    # 2. r_baseline
    eq_e_baseline = calc_e_baseline(
        inputs["FC_baseline"], inputs["NCV"], inputs["EF_CO2"], inputs["OF"],
        inputs["EC_baseline"], inputs["EF_grid"]
    )
    residuals["r_baseline"] = preds["E_baseline"] - eq_e_baseline

    # 3. r_reduced
    eq_e_reduced = calc_e_reduced(preds["E_baseline"], preds["E_actual"])
    residuals["r_reduced"] = preds["E_reduced"] - eq_e_reduced

    # 4. r_removed
    eq_e_removed = calc_e_removed(inputs["M_captured"], inputs["P_CO2"], inputs["eta_purity"])
    residuals["r_removed"] = preds["E_removed"] - eq_e_removed

    # 5. r_disp_chem
    eq_e_disp_chem = calc_e_displaced_chem(inputs["M_byproduct"], inputs["EF_virgin_displace"])
    residuals["r_disp_chem"] = preds["E_displaced_chem"] - eq_e_disp_chem

    # 6. r_disp_heat
    eq_e_disp_heat = calc_e_displaced_heat(
        inputs["H_recovered"], inputs["eta_boiler"], inputs["NCV_fuel"], inputs["EF_CO2_fuel"]
    )
    residuals["r_disp_heat"] = preds["E_displaced_heat"] - eq_e_disp_heat

    # 7. r_displaced (compositional)
    eq_e_disp = calc_e_displaced(preds["E_displaced_chem"], preds["E_displaced_heat"])
    residuals["r_displaced"] = preds["E_displaced"] - eq_e_disp

    # 8. r_EChw
    eq_ec_hw = calc_ec_hardware(inputs["P_fan"], inputs["P_pump"], inputs["t_op"])
    residuals["r_EChw"] = preds["EC_hardware"] - eq_ec_hw

    # 9. r_aux
    eq_pe_aux = calc_pe_aux(preds["EC_hardware"], inputs["EF_grid"], mode=pe_aux_mode)
    residuals["r_aux"] = preds["PE_aux"] - eq_pe_aux

    # 10. r_lifecycle
    eq_pe_lifecycle = calc_pe_lifecycle(inputs["M_solvent_makeup"], inputs["EF_solvent_LCA"])
    residuals["r_lifecycle"] = preds["PE_lifecycle"] - eq_pe_lifecycle

    # 11. r_slip
    eq_l_slip = calc_l_slip(inputs["M_captured"], inputs["gamma_slip"])
    residuals["r_slip"] = preds["L_slip"] - eq_l_slip

    # 12. r_transport
    eq_l_trans = calc_l_transport(inputs["D_k"], inputs["EF_vehicle_k"], inputs["M_trans_k"])
    residuals["r_transport"] = preds["L_transport"] - eq_l_trans

    # 13. r_leakage (compositional)
    eq_l_leakage = calc_l_leakage(preds["L_slip"], preds["L_transport"])
    residuals["r_leakage"] = preds["L_leakage"] - eq_l_leakage

    # 14. r_mult (compositional)
    f_valid_in = inputs["F_valid"] if "F_valid" in inputs else 1.0
    eq_f_mult = calc_f_multiplier(f_valid_in, inputs["F_SME"], inputs["F_perm"], inputs["alpha"])
    residuals["r_mult"] = preds["F_multiplier"] - eq_f_mult

    # 15. r_CC (top-level accounting consistency)
    eq_cc_t = calc_cc_t(
        preds["E_reduced"], preds["E_removed"], preds["E_displaced"],
        preds["PE_aux"], preds["PE_lifecycle"], preds["L_leakage"],
        preds["F_multiplier"]
    )
    residuals["r_CC"] = preds["CC_T"] - eq_cc_t

    return residuals

def normalize_residuals(
    raw_residuals: Dict[str, Union[float, np.ndarray]],
    preds: Dict[str, Union[float, np.ndarray]],
    characteristic_scales: Dict[str, float],
    eps: float = 1e-6,
) -> Tuple[Dict[str, Union[float, np.ndarray]], Dict[str, float]]:
    """Normalize residuals per Section I & J."""
    norm_residuals = {}
    scales_used = {}

    for name, r_val in raw_residuals.items():
        if name in ["r_displaced", "r_leakage", "r_mult"]:
            # Compositional residuals: normalize by scale of larger operand (Section I.4)
            if name == "r_displaced":
                s = np.maximum(
                    np.abs(preds["E_displaced"]),
                    np.abs(preds["E_displaced_chem"] + preds["E_displaced_heat"])
                )
            elif name == "r_leakage":
                s = np.maximum(
                    np.abs(preds["L_leakage"]),
                    np.abs(preds["L_slip"] + preds["L_transport"])
                )
            else:  # r_mult
                s = np.maximum(np.abs(preds["F_multiplier"]), 1.0)
            
            s_val = float(np.mean(s)) if isinstance(s, np.ndarray) else float(s)
            s_val = max(s_val, eps)
        elif name == "r_CC":
            s_val = characteristic_scales.get("CC_T", 100.0)
        else:
            s_val = characteristic_scales.get(name, 10.0)

        s_val = max(s_val, eps)
        norm_residuals[name] = r_val / s_val
        scales_used[name] = s_val

    return norm_residuals, scales_used
