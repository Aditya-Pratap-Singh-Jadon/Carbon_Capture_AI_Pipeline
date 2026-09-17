"""Thermodynamic Models and Packed Bed Hydraulic Relationships."""

import numpy as np
from typing import Dict, Tuple

# Universal gas constant J / (mol * K)
R_GAS = 8.314462618

# Critical properties for CO2
TC_CO2 = 304.13   # K
PC_CO2 = 7.3773e6 # Pa (73.77 bar)
OMEGA_CO2 = 0.2239 # Acentric factor

class PengRobinsonEOS:
    """Peng-Robinson Equation of State for real gas behavior (CO2)."""

    def __init__(self, tc: float = TC_CO2, pc: float = PC_CO2, omega: float = OMEGA_CO2):
        self.tc = tc
        self.pc = pc
        self.omega = omega
        # EOS constants
        self.kappa = 0.37464 + 1.54226 * self.omega - 0.26992 * (self.omega ** 2)
        self.a_c = 0.45724 * (R_GAS ** 2) * (self.tc ** 2) / self.pc
        self.b = 0.07780 * R_GAS * self.tc / self.pc

    def compute_z_factor(self, temp_k: float, pressure_pa: float) -> float:
        """Compute compressibility factor Z by solving the cubic polynomial."""
        tr = temp_k / self.tc
        alpha = (1.0 + self.kappa * (1.0 - np.sqrt(tr))) ** 2
        a = self.a_c * alpha
        
        A = a * pressure_pa / ((R_GAS * temp_k) ** 2)
        B = self.b * pressure_pa / (R_GAS * temp_k)
        
        # Cubic coefficients: Z^3 - (1-B)Z^2 + (A - 3B^2 - 2B)Z - (AB - B^2 - B^3) = 0
        coeffs = [
            1.0,
            -(1.0 - B),
            A - 3.0 * (B ** 2) - 2.0 * B,
            -(A * B - (B ** 2) - (B ** 3))
        ]
        roots = np.roots(coeffs)
        # Select largest real root for vapor phase
        real_roots = [float(r.real) for r in roots if np.isreal(r) and r.real > B]
        return max(real_roots) if real_roots else 1.0

    def compute_density(self, temp_k: float, pressure_pa: float, molar_mass_kg_mol: float = 0.04401) -> float:
        """Calculate real gas density in kg/m3."""
        z = self.compute_z_factor(temp_k, pressure_pa)
        rho_molar = pressure_pa / (z * R_GAS * temp_k) # mol / m3
        return rho_molar * molar_mass_kg_mol

def calculate_scrubber_fan_power(
    q_gas_m3_s: float,
    delta_p_pa: float,
    eta_fan: float = 0.75,
) -> float:
    """Calculate fan electrical power in kW: P_fan = (Q_gas * Delta_P) / (1000 * eta_fan)."""
    eta = max(eta_fan, 0.1)
    power_watts = (q_gas_m3_s * delta_p_pa) / eta
    return power_watts / 1000.0 # Return in kW

def calculate_solvent_pump_power(
    q_liq_m3_s: float,
    rho_liq_kg_m3: float,
    h_pump_m: float,
    eta_pump: float = 0.70,
    g_accel: float = 9.80665,
) -> float:
    """Calculate solvent recirculation pump power in kW."""
    eta = max(eta_pump, 0.1)
    power_watts = (rho_liq_kg_m3 * g_accel * q_liq_m3_s * h_pump_m) / eta
    return power_watts / 1000.0 # Return in kW
