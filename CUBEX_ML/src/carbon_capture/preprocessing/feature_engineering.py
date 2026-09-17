"""Physics-Informed Feature Engineering without hidden assumptions."""

from typing import List, Tuple
import numpy as np
import pandas as pd
from carbon_capture.utils.logging import get_logger

logger = get_logger(__name__)

class PhysicsFeatureExtractor:
    """Extracts derived physical quantities supported strictly by the mathematical formulation."""

    def __init__(self):
        self.engineered_feature_names: List[str] = [
            "P_aux_total",       # P_fan + P_pump (kW)
            "transport_ton_km",  # D_k * M_trans_k (t-km)
            "max_stoich_co2",    # M_captured * P_CO2 (tCO2)
            "heat_input_fuel",   # H_recovered / (eta_boiler * NCV_fuel)
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add first-principles physics features to DataFrame."""
        out = df.copy()

        # 1. Total auxiliary power (kW)
        out["P_aux_total"] = out["P_fan"] + out["P_pump"]

        # 2. Total transport ton-km
        out["transport_ton_km"] = out["D_k"] * out["M_trans_k"]

        # 3. Maximum stoichiometric CO2 capacity
        out["max_stoich_co2"] = out["M_captured"] * out["P_CO2"]

        # 4. Equivalent fuel energy required for recovered heat (kg or m3 equivalent)
        denom = np.maximum(out["eta_boiler"] * out["NCV_fuel"], 1e-6)
        out["heat_input_fuel"] = out["H_recovered"] / denom

        return out
