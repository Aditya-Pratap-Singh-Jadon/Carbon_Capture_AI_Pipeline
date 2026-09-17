"""Chemical Stoichiometry and Mineralization Calculations."""

from typing import Dict

# Molecular weights in g/mol
MW_CO2 = 44.01
MW_CACO3 = 100.087
MW_K2CO3 = 138.205
MW_NA2CO3 = 105.989
MW_MGCO3 = 84.314

# Stoichiometric carbon dioxide mass fractions (P_CO2) for common mineralization products
STOICHIOMETRIC_P_CO2: Dict[str, float] = {
    "calcium_carbonate": MW_CO2 / MW_CACO3,   # ~0.4397
    "potassium_carbonate": MW_CO2 / MW_K2CO3, # ~0.3184
    "sodium_carbonate": MW_CO2 / MW_NA2CO3,   # ~0.4152
    "magnesium_carbonate": MW_CO2 / MW_MGCO3, # ~0.5220
    "pure_co2_gas": 1.0,
}

def get_stoichiometric_ratio(compound: str) -> float:
    """Retrieve theoretical P_CO2 ratio for a known chemical product."""
    key = compound.lower().replace(" ", "_")
    if key in STOICHIOMETRIC_P_CO2:
        return STOICHIOMETRIC_P_CO2[key]
    raise KeyError(f"Compound '{compound}' not in stoichiometric registry. Known: {list(STOICHIOMETRIC_P_CO2.keys())}")

def calculate_theoretical_yield(feedstock_mass_tonnes: float, compound: str) -> float:
    """Compute maximum stoichiometric yield of carbonate product."""
    p_co2 = get_stoichiometric_ratio(compound)
    return feedstock_mass_tonnes * p_co2
