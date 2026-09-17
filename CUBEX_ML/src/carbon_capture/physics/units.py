"""Centralized Physical Units and Conversions Registry."""

from typing import Dict

# Standard SI / Engineering conversion factors
CONVERSIONS = {
    "kg_to_tonnes": 1e-3,
    "tonnes_to_kg": 1e3,
    "g_to_kg": 1e-3,
    "j_to_mj": 1e-6,
    "mj_to_gj": 1e-3,
    "kwh_to_mj": 3.6,
    "mj_to_kwh": 1.0 / 3.6,
    "hours_to_seconds": 3600.0,
    "seconds_to_hours": 1.0 / 3600.0,
    "bar_to_pa": 1e5,
    "pa_to_bar": 1e-5,
}

def convert(val: float, conversion_name: str) -> float:
    """Convert a value using the standard conversion table."""
    if conversion_name not in CONVERSIONS:
        raise KeyError(f"Conversion factor '{conversion_name}' not recognized.")
    return val * CONVERSIONS[conversion_name]
