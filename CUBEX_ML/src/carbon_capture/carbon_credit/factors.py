"""Carbon-Credit Factor Registries, Enterprise Multipliers, and Policy Discounts."""

from typing import Dict, Optional

# Permanence Horizon Table (Document 3 & Project.pdf)
# Final product utilization determines the containment permanence factor F_perm in [0, 1]
PERMANENCE_FACTORS: Dict[str, float] = {
    "geological_storage": 1.00,
    "mineral_carbonation_concrete": 0.98,
    "synthetic_fuel": 0.15,
    "enhanced_oil_recovery": 0.50,
    "chemical_plastics": 0.70,
    "default": 0.95,
}

# Enterprise Scale Factors F_SME >= 1.0 (MSME economic incentive)
ENTERPRISE_SCALE_FACTORS: Dict[str, float] = {
    "micro_enterprise": 1.15,
    "small_enterprise": 1.10,
    "medium_enterprise": 1.05,
    "large_enterprise": 1.00,
    "default": 1.00,
}

def get_permanence_factor(storage_type: str = "default") -> float:
    """Retrieve permanence factor based on carbon utilization route."""
    key = storage_type.lower().replace(" ", "_")
    return PERMANENCE_FACTORS.get(key, PERMANENCE_FACTORS["default"])

def get_enterprise_scale_factor(scale_tier: str = "default") -> float:
    """Retrieve MSME enterprise incentive multiplier F_SME >= 1.0."""
    key = scale_tier.lower().replace(" ", "_")
    return ENTERPRISE_SCALE_FACTORS.get(key, ENTERPRISE_SCALE_FACTORS["default"])
