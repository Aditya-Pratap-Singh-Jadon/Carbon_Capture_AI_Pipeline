"""Authoritative Deterministic Carbon-Credit Calculation Engine.

Per Section 14 of specification: The deterministic engine is the sole authoritative
credit calculation authority. The neural network never directly outputs the authoritative
credit volume.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
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
class CarbonCreditRecord:
    record_id: int
    E_reduced: float
    E_removed: float
    E_displaced: float
    E_displaced_chem: float
    E_displaced_heat: float
    PE_aux: float
    PE_lifecycle: float
    L_leakage: float
    L_slip: float
    L_transport: float
    F_valid: float
    F_SME: float
    F_perm: float
    alpha: float
    F_multiplier: float
    net_benefit_bracket: float
    CC_T: float

@dataclass
class CarbonCreditSummary:
    total_CC_T: float
    total_E_reduced: float
    total_E_removed: float
    total_E_displaced: float
    total_PE_aux: float
    total_PE_lifecycle: float
    total_L_leakage: float
    mean_F_multiplier: float
    mean_F_valid: float
    record_count: int
    records: List[CarbonCreditRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_CC_T": self.total_CC_T,
            "total_E_reduced": self.total_E_reduced,
            "total_E_removed": self.total_E_removed,
            "total_E_displaced": self.total_E_displaced,
            "total_PE_aux": self.total_PE_aux,
            "total_PE_lifecycle": self.total_PE_lifecycle,
            "total_L_leakage": self.total_L_leakage,
            "mean_F_multiplier": self.mean_F_multiplier,
            "mean_F_valid": self.mean_F_valid,
            "record_count": self.record_count,
        }

class DeterministicCarbonCreditCalculator:
    """Calculates authoritative carbon credits from validated physical process telemetry."""

    def __init__(self, pe_aux_mode: str = "verbatim"):
        self.pe_aux_mode = pe_aux_mode

    def calculate_from_dataframe(
        self,
        df: pd.DataFrame,
        f_valid: Optional[Union[float, np.ndarray]] = None,
    ) -> CarbonCreditSummary:
        """Evaluate deterministic carbon-credit formula on canonical DataFrame."""
        # Benefit terms
        e_act = calc_e_actual(
            df["FC_actual"].to_numpy(), df["NCV"].to_numpy(), df["EF_CO2"].to_numpy(),
            df["OF"].to_numpy(), df["EC_actual"].to_numpy(), df["EF_grid"].to_numpy()
        )
        e_base = calc_e_baseline(
            df["FC_baseline"].to_numpy(), df["NCV"].to_numpy(), df["EF_CO2"].to_numpy(),
            df["OF"].to_numpy(), df["EC_baseline"].to_numpy(), df["EF_grid"].to_numpy()
        )
        e_red = calc_e_reduced(e_base, e_act)
        e_rem = calc_e_removed(
            df["M_captured"].to_numpy(), df["P_CO2"].to_numpy(), df["eta_purity"].to_numpy()
        )
        e_disp_chem = calc_e_displaced_chem(
            df["M_byproduct"].to_numpy(), df["EF_virgin_displace"].to_numpy()
        )
        e_disp_heat = calc_e_displaced_heat(
            df["H_recovered"].to_numpy(), df["eta_boiler"].to_numpy(),
            df["NCV_fuel"].to_numpy(), df["EF_CO2_fuel"].to_numpy()
        )
        e_disp = calc_e_displaced(e_disp_chem, e_disp_heat)

        # Penalty terms
        ec_hw = calc_ec_hardware(
            df["P_fan"].to_numpy(), df["P_pump"].to_numpy(), df["t_op"].to_numpy()
        )
        pe_aux = calc_pe_aux(ec_hw, df["EF_grid"].to_numpy(), mode=self.pe_aux_mode)
        pe_life = calc_pe_lifecycle(
            df["M_solvent_makeup"].to_numpy(), df["EF_solvent_LCA"].to_numpy()
        )
        l_slip = calc_l_slip(df["M_captured"].to_numpy(), df["gamma_slip"].to_numpy())
        l_trans = calc_l_transport(
            df["D_k"].to_numpy(), df["EF_vehicle_k"].to_numpy(), df["M_trans_k"].to_numpy()
        )
        l_leak = calc_l_leakage(l_slip, l_trans)

        # Multipliers
        n = len(df)
        if f_valid is not None:
            if isinstance(f_valid, (int, float)):
                f_val = np.full(n, float(f_valid))
            else:
                f_val = np.asarray(f_valid, dtype=float)
        elif "F_valid" in df.columns:
            f_val = df["F_valid"].to_numpy(dtype=float)
        else:
            f_val = np.ones(n, dtype=float)

        f_sme = df["F_SME"].to_numpy(dtype=float) if "F_SME" in df.columns else np.full(n, 1.05)
        f_perm = df["F_perm"].to_numpy(dtype=float) if "F_perm" in df.columns else np.full(n, 0.98)
        alpha = df["alpha"].to_numpy(dtype=float) if "alpha" in df.columns else np.full(n, 0.90)
        f_mult = calc_f_multiplier(f_val, f_sme, f_perm, alpha)

        # CC_T
        cc_t = calc_cc_t(e_red, e_rem, e_disp, pe_aux, pe_life, l_leak, f_mult)

        records = []
        n = len(df)
        for i in range(n):
            bracket = float(e_red[i] + e_rem[i] + e_disp[i] - pe_aux[i] - pe_life[i] - l_leak[i])
            records.append(
                CarbonCreditRecord(
                    record_id=i,
                    E_reduced=float(e_red[i]),
                    E_removed=float(e_rem[i]),
                    E_displaced=float(e_disp[i]),
                    E_displaced_chem=float(e_disp_chem[i]),
                    E_displaced_heat=float(e_disp_heat[i]),
                    PE_aux=float(pe_aux[i]),
                    PE_lifecycle=float(pe_life[i]),
                    L_leakage=float(l_leak[i]),
                    L_slip=float(l_slip[i]),
                    L_transport=float(l_trans[i]),
                    F_valid=float(f_val[i]),
                    F_SME=float(f_sme[i]),
                    F_perm=float(f_perm[i]),
                    alpha=float(alpha[i]),
                    F_multiplier=float(f_mult[i]),
                    net_benefit_bracket=bracket,
                    CC_T=float(cc_t[i]),
                )
            )

        return CarbonCreditSummary(
            total_CC_T=float(np.sum(cc_t)),
            total_E_reduced=float(np.sum(e_red)),
            total_E_removed=float(np.sum(e_rem)),
            total_E_displaced=float(np.sum(e_disp)),
            total_PE_aux=float(np.sum(pe_aux)),
            total_PE_lifecycle=float(np.sum(pe_life)),
            total_L_leakage=float(np.sum(l_leak)),
            mean_F_multiplier=float(np.mean(f_mult)),
            mean_F_valid=float(np.mean(f_val)),
            record_count=n,
            records=records,
        )
