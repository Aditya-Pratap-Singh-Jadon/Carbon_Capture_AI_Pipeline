"""Controlled Synthetic Industrial Dataset Generator for Carbon Capture.

Generates realistic operating regimes, physics-consistent ground truth,
and controlled fault/anomaly injection scenarios for rigorous pipeline testing.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from carbon_capture.input.canonical_order import CANONICAL_INPUT_ORDER
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
from carbon_capture.utils.logging import get_logger

logger = get_logger(__name__)

class SyntheticDataGenerator:
    """Generates physically consistent industrial carbon capture telemetry."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_normal_regime(
        self,
        n_samples: int = 1000,
        regime: str = "steady_state",
    ) -> pd.DataFrame:
        """Generate physically consistent normal operating telemetry."""
        # 1. Direct combustion & fuel
        if regime == "steady_state":
            fc_actual = self.rng.uniform(40.0, 60.0, n_samples)
            fc_baseline = fc_actual + self.rng.uniform(10.0, 25.0, n_samples)
        elif regime == "peak_load":
            fc_actual = self.rng.uniform(70.0, 95.0, n_samples)
            fc_baseline = fc_actual + self.rng.uniform(15.0, 35.0, n_samples)
        else: # turndown
            fc_actual = self.rng.uniform(15.0, 30.0, n_samples)
            fc_baseline = fc_actual + self.rng.uniform(5.0, 15.0, n_samples)

        ncv = self.rng.uniform(42.0, 48.0, n_samples) # MJ/kg (e.g. natural gas)
        ef_co2 = self.rng.uniform(0.055, 0.058, n_samples) # tCO2/MJ
        of = self.rng.uniform(0.98, 1.0, n_samples) # Oxidation factor

        # 2. Scope 2 / Facility electricity
        ec_actual = self.rng.uniform(200.0, 500.0, n_samples) # kWh
        ec_baseline = ec_actual + self.rng.uniform(50.0, 150.0, n_samples) # kWh
        ef_grid = self.rng.uniform(0.40, 0.55, n_samples) # tCO2/MWh -> ~0.00045 tCO2/kWh
        # Note: in typical industrial grids ~0.0005 tCO2/kWh
        ef_grid = self.rng.uniform(0.0004, 0.0006, n_samples)

        # 3. Direct Capture & Chemical Synthesis
        # Product capture mass (e.g. K2CO3 or CaCO3)
        m_captured = self.rng.uniform(80.0, 150.0, n_samples) # tonnes
        p_co2 = np.full(n_samples, 0.4397) # Stoichiometric ratio for CaCO3
        eta_purity = self.rng.uniform(0.92, 0.98, n_samples) # Product purity

        # 4. Circular Byproducts & Waste Heat
        m_byproduct = m_captured * self.rng.uniform(0.85, 0.95, n_samples)
        ef_virgin_displace = self.rng.uniform(0.15, 0.25, n_samples) # tCO2e/t

        h_recovered = self.rng.uniform(15000.0, 35000.0, n_samples) # MJ
        eta_boiler = self.rng.uniform(0.82, 0.88, n_samples)
        ncv_fuel = self.rng.uniform(40.0, 45.0, n_samples) # MJ/kg
        ef_co2_fuel = self.rng.uniform(0.055, 0.060, n_samples) # tCO2/MJ

        # 5. Auxiliary Hardware Telemetry
        p_fan = self.rng.uniform(30.0, 75.0, n_samples) # kW
        p_pump = self.rng.uniform(15.0, 35.0, n_samples) # kW
        t_op = self.rng.uniform(3600.0, 7200.0, n_samples) # seconds (1 to 2 hours)

        # 6. Sorbents & Solvent Lifecycle
        m_solvent_makeup = self.rng.uniform(0.05, 0.25, n_samples) # tonnes
        ef_solvent_lca = self.rng.uniform(1.8, 2.4, n_samples) # tCO2e/t

        # 7. Slippage & Transport
        gamma_slip = self.rng.uniform(0.01, 0.03, n_samples) # 1-3% slippage
        d_k = self.rng.uniform(20.0, 120.0, n_samples) # km
        ef_vehicle_k = self.rng.uniform(0.00012, 0.00018, n_samples) # tCO2e/t-km
        m_trans_k = m_byproduct.copy()

        # 8. Carbon Credit Multipliers (Policy Context Only - F_valid is derived post-model)
        f_sme = self.rng.uniform(1.02, 1.10, n_samples) # >= 1.0
        f_perm = self.rng.uniform(0.95, 0.99, n_samples)
        alpha = self.rng.uniform(0.88, 0.95, n_samples)

        data = {
            "FC_actual": fc_actual,
            "FC_baseline": fc_baseline,
            "NCV": ncv,
            "EF_CO2": ef_co2,
            "OF": of,
            "EC_actual": ec_actual,
            "EC_baseline": ec_baseline,
            "EF_grid": ef_grid,
            "M_captured": m_captured,
            "P_CO2": p_co2,
            "eta_purity": eta_purity,
            "M_byproduct": m_byproduct,
            "EF_virgin_displace": ef_virgin_displace,
            "H_recovered": h_recovered,
            "eta_boiler": eta_boiler,
            "NCV_fuel": ncv_fuel,
            "EF_CO2_fuel": ef_co2_fuel,
            "P_fan": p_fan,
            "P_pump": p_pump,
            "t_op": t_op,
            "M_solvent_makeup": m_solvent_makeup,
            "EF_solvent_LCA": ef_solvent_lca,
            "gamma_slip": gamma_slip,
            "D_k": d_k,
            "EF_vehicle_k": ef_vehicle_k,
            "M_trans_k": m_trans_k,
            "F_SME": f_sme,
            "F_perm": f_perm,
            "alpha": alpha,
        }

        df = pd.DataFrame(data)[CANONICAL_INPUT_ORDER]
        # Metadata flag
        df.attrs["is_synthetic"] = True
        return df

    def generate_temporal_regime(
        self,
        n_steps: int = 300,
        scenario: str = "stable",
    ) -> pd.DataFrame:
        """Generate chronologically ordered, physically dependent temporal trajectories.
        
        Scenarios:
        - stable: baseline steady state.
        - gradual_fan_change: Fan speed gradually increases, driving higher flow and capture.
        - gradual_flow_change: Flow changes independently (e.g. process load shift).
        - temp_transient: Temperature/heat recovered transient shift.
        - co2_transient: Feed CO2 concentration (EF_CO2 equivalent) transient.
        - legitimate_transition: Normal operating regime shift (e.g. baseline to peak).
        """
        # Base static values
        ncv = np.full(n_steps, 45.0)
        ef_co2 = np.full(n_steps, 0.056)
        of = np.full(n_steps, 0.99)
        ef_grid = np.full(n_steps, 0.0005)
        p_co2 = np.full(n_steps, 0.4397)
        ef_virgin_displace = np.full(n_steps, 0.20)
        eta_boiler = np.full(n_steps, 0.85)
        ncv_fuel = np.full(n_steps, 42.5)
        ef_co2_fuel = np.full(n_steps, 0.058)
        t_op = np.full(n_steps, 3600.0) # 1 hour per step
        ef_solvent_lca = np.full(n_steps, 2.1)
        gamma_slip = np.full(n_steps, 0.02)
        d_k = np.full(n_steps, 50.0)
        ef_vehicle_k = np.full(n_steps, 0.00015)
        f_sme = np.full(n_steps, 1.05)
        f_perm = np.full(n_steps, 0.98)
        alpha = np.full(n_steps, 0.90)

        # Dynamic state initialization (random walk baselines)
        p_fan = np.zeros(n_steps)
        p_fan[0] = 50.0
        
        # Physics driver dependencies:
        # Flow (FC_actual) is heavily influenced by fan speed.
        # Capture (M_captured) is influenced by Flow and Purity.

        for i in range(1, n_steps):
            p_fan[i] = p_fan[i-1] + self.rng.normal(0, 0.5)
            
            # Inject transients based on scenario
            if scenario == "gradual_fan_change" and i > n_steps // 4 and i < 3 * n_steps // 4:
                p_fan[i] += 0.2 # Steady upward drift
            elif scenario == "legitimate_transition" and i == n_steps // 2:
                # Big legitimate step up that smooths out
                p_fan[i] = 80.0
                
        # Ensure bounds
        p_fan = np.clip(p_fan, 30.0, 100.0)
        
        # Flow is a function of fan power + noise
        fc_actual = 20.0 + 0.6 * p_fan + self.rng.normal(0, 1.0, n_steps)
        
        if scenario == "gradual_flow_change":
            drift = np.zeros(n_steps)
            drift[n_steps//3:] = np.linspace(0, 15.0, n_steps - n_steps//3)
            fc_actual += drift
            
        fc_baseline = fc_actual + 20.0 + self.rng.normal(0, 1.0, n_steps)
        ec_actual = 150.0 + 3.0 * p_fan + self.rng.normal(0, 5.0, n_steps)
        ec_baseline = ec_actual + 100.0
        
        # Purity and CO2 transients
        eta_purity = np.full(n_steps, 0.95) + self.rng.normal(0, 0.005, n_steps)
        if scenario == "co2_transient":
            eta_purity[n_steps//2:] -= 0.05 # sudden drop in efficiency/purity
            
        m_captured = 1.2 * fc_actual * eta_purity + self.rng.normal(0, 2.0, n_steps)
        m_byproduct = m_captured * 0.9 + self.rng.normal(0, 1.0, n_steps)
        
        h_recovered = 500.0 * fc_actual + self.rng.normal(0, 100.0, n_steps)
        if scenario == "temp_transient":
            h_recovered[n_steps//3:2*n_steps//3] += 10000.0 # Heat spike
            
        p_pump = 10.0 + 0.2 * fc_actual + self.rng.normal(0, 0.5, n_steps)
        m_solvent_makeup = 0.001 * m_captured + self.rng.normal(0, 0.01, n_steps)
        m_trans_k = m_byproduct.copy()

        data = {
            "timestamp": np.arange(n_steps), # Add explicit chronological ordering
            "FC_actual": fc_actual,
            "FC_baseline": fc_baseline,
            "NCV": ncv,
            "EF_CO2": ef_co2,
            "OF": of,
            "EC_actual": ec_actual,
            "EC_baseline": ec_baseline,
            "EF_grid": ef_grid,
            "M_captured": m_captured,
            "P_CO2": p_co2,
            "eta_purity": eta_purity,
            "M_byproduct": m_byproduct,
            "EF_virgin_displace": ef_virgin_displace,
            "H_recovered": h_recovered,
            "eta_boiler": eta_boiler,
            "NCV_fuel": ncv_fuel,
            "EF_CO2_fuel": ef_co2_fuel,
            "P_fan": p_fan,
            "P_pump": p_pump,
            "t_op": t_op,
            "M_solvent_makeup": m_solvent_makeup,
            "EF_solvent_LCA": ef_solvent_lca,
            "gamma_slip": gamma_slip,
            "D_k": d_k,
            "EF_vehicle_k": ef_vehicle_k,
            "M_trans_k": m_trans_k,
            "F_SME": f_sme,
            "F_perm": f_perm,
            "alpha": alpha,
        }

        df = pd.DataFrame(data)
        df.attrs["is_synthetic"] = True
        return df

    def compute_ground_truth_targets(self, df: pd.DataFrame, pe_aux_mode: str = "verbatim") -> pd.DataFrame:
        """Compute exact closed-form targets for all 15 physical and credit quantities."""
        e_actual = calc_e_actual(
            df["FC_actual"], df["NCV"], df["EF_CO2"], df["OF"],
            df["EC_actual"], df["EF_grid"]
        )
        e_baseline = calc_e_baseline(
            df["FC_baseline"], df["NCV"], df["EF_CO2"], df["OF"],
            df["EC_baseline"], df["EF_grid"]
        )
        e_reduced = calc_e_reduced(e_baseline, e_actual)
        e_removed = calc_e_removed(df["M_captured"], df["P_CO2"], df["eta_purity"])
        e_disp_chem = calc_e_displaced_chem(df["M_byproduct"], df["EF_virgin_displace"])
        e_disp_heat = calc_e_displaced_heat(
            df["H_recovered"], df["eta_boiler"], df["NCV_fuel"], df["EF_CO2_fuel"]
        )
        e_displaced = calc_e_displaced(e_disp_chem, e_disp_heat)
        ec_hw = calc_ec_hardware(df["P_fan"], df["P_pump"], df["t_op"])
        pe_aux = calc_pe_aux(ec_hw, df["EF_grid"], mode=pe_aux_mode)
        pe_lifecycle = calc_pe_lifecycle(df["M_solvent_makeup"], df["EF_solvent_LCA"])
        l_slip = calc_l_slip(df["M_captured"], df["gamma_slip"])
        l_trans = calc_l_transport(df["D_k"], df["EF_vehicle_k"], df["M_trans_k"])
        l_leakage = calc_l_leakage(l_slip, l_trans)
        f_valid = df["F_valid"] if "F_valid" in df.columns else 0.95
        f_mult = calc_f_multiplier(f_valid, df["F_SME"], df["F_perm"], df["alpha"])
        cc_t = calc_cc_t(e_reduced, e_removed, e_displaced, pe_aux, pe_lifecycle, l_leakage, f_mult)

        targets = pd.DataFrame({
            "E_actual": e_actual,
            "E_baseline": e_baseline,
            "E_reduced": e_reduced,
            "E_removed": e_removed,
            "E_displaced_chem": e_disp_chem,
            "E_displaced_heat": e_disp_heat,
            "E_displaced": e_displaced,
            "EC_hardware": ec_hw,
            "PE_aux": pe_aux,
            "PE_lifecycle": pe_lifecycle,
            "L_slip": l_slip,
            "L_transport": l_trans,
            "L_leakage": l_leakage,
            "F_multiplier": f_mult,
            "CC_T": cc_t,
        })
        return targets

    def generate_corrupted_dataset(
        self,
        base_df: pd.DataFrame,
        corruption_type: str = "sensor_bias",
    ) -> Tuple[pd.DataFrame, str]:
        """Inject specific, controlled physical or data corruptions."""
        corrupted = base_df.copy()
        description = ""

        if corruption_type == "sensor_bias":
            # Intentionally corrupt purity sensor above 1.0 (violates physical bound)
            corrupted.loc[0:15, "eta_purity"] = 1.45
            description = "Purity sensor biased to 145% (violates eta_purity in [0, 1])."
        elif corruption_type == "mass_balance_violation":
            # Byproduct mass greater than captured mass (impossible physical creation)
            corrupted.loc[0:20, "M_byproduct"] = corrupted.loc[0:20, "M_captured"] * 3.5
            description = "Mass byproduct 3.5x higher than total captured mass."
        elif corruption_type == "negative_physical_quantity":
            # Impossible negative power
            corrupted.loc[0:10, "P_fan"] = -45.0
            description = "Negative fan power (-45 kW) violating non-negativity constraint."
        elif corruption_type == "nan_injection":
            # Missing sensor readings
            corrupted.loc[5:10, "FC_actual"] = np.nan
            description = "Injected NaN values into FC_actual."
        elif corruption_type == "slip_violation":
            # Slippage exceeds 100%
            corrupted.loc[0:10, "gamma_slip"] = 1.25
            description = "Slippage fraction exceeds 1.0."
        elif corruption_type == "co2_sensor_bias":
            # CO2 emission factor impossible
            corrupted.loc[0:10, "EF_CO2"] = 10.0
            description = "Unrealistic CO2 emission factor."
        elif corruption_type == "flow_sensor_bias":
            # Huge flow
            corrupted.loc[0:10, "FC_actual"] = 1e6
            description = "Flow sensor reads 1e6."
        elif corruption_type == "captured_mass_bias":
            # Captured mass negative
            corrupted.loc[0:10, "M_captured"] = -10.0
            description = "Captured mass is negative."
        elif corruption_type == "impossible_temperature":
            # H_recovered negative
            corrupted.loc[0:10, "H_recovered"] = -5000.0
            description = "Impossible negative recovered heat."
        elif corruption_type == "multiple_failures":
            corrupted.loc[0:10, "P_fan"] = -10.0
            corrupted.loc[0:10, "eta_purity"] = 1.5
            corrupted.loc[0:10, "gamma_slip"] = 1.1
            description = "Multiple simultaneous constraint violations."
        elif corruption_type == "measurement_noise":
            # Add 5% noise to M_captured
            noise = self.rng.normal(0, 0.05, len(corrupted)) * corrupted["M_captured"]
            corrupted["M_captured"] += noise
            description = "Added 5% Gaussian noise to M_captured."
        elif corruption_type == "extreme_valid":
            # Very high but valid values
            corrupted["FC_actual"] = 1000.0
            corrupted["FC_baseline"] = 1200.0
            corrupted["M_captured"] = 1500.0
            corrupted["M_byproduct"] = 1400.0
            corrupted["P_fan"] = 500.0
            corrupted["P_pump"] = 200.0
            corrupted["H_recovered"] = 300000.0
            description = "Extreme but physically valid operating condition."
        elif corruption_type == "shuffled_columns":
            cols = list(corrupted.columns)
            self.rng.shuffle(cols)
            corrupted = corrupted[cols]
            description = "Shuffled CSV columns."
        elif corruption_type == "missing_column":
            corrupted = corrupted.drop(columns=["M_captured"])
            description = "Missing required column M_captured."
        elif corruption_type == "temporal_sensor_drift":
            # Gradually drift purity sensor over time (last 50 steps)
            drift = np.linspace(0, 0.3, 50)
            corrupted.loc[len(corrupted)-50:, "eta_purity"] += drift
            description = "Gradual sensor drift on eta_purity violating bounds at the end."
        elif corruption_type == "temporal_sensor_bias":
            # Sudden constant bias halfway through
            corrupted.loc[len(corrupted)//2:, "P_fan"] += 50.0 
            # Note: Legitimate transitions would see FC_actual and M_captured respond.
            # A sensor bias does not update the physics dependents, causing physical inconsistency!
            description = "Sudden constant bias on P_fan without corresponding flow changes."
        elif corruption_type == "sudden_inconsistent_transition":
            # Physically impossible jump in capture without flow
            corrupted.loc[len(corrupted)//2, "M_captured"] += 500.0
            description = "Sudden physically inconsistent spike in M_captured for 1 timestep."
        elif corruption_type == "persistent_fault":
            corrupted.loc[len(corrupted)//2:, "gamma_slip"] = 1.5
            description = "Persistent slip violation starting halfway through."
        elif corruption_type == "recovery_from_fault":
            # Fault for 20 steps, then recovery
            mid = len(corrupted)//2
            corrupted.loc[mid:mid+20, "FC_actual"] = -100.0
            description = "Negative flow fault that resolves after 20 timesteps."
        else:
            raise ValueError(f"Unknown corruption type: {corruption_type}")

        return corrupted, description
