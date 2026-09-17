"""Unit tests for Physics Equations, Units, and Residuals."""

import pytest
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
from carbon_capture.physics.thermodynamics import PengRobinsonEOS, calculate_scrubber_fan_power
from carbon_capture.physics.stoichiometry import get_stoichiometric_ratio
from carbon_capture.physics.residuals import compute_raw_residuals, normalize_residuals

def test_governing_equations_basic():
    # E_actual
    e_act = calc_e_actual(fc_actual=50.0, ncv=45.0, ef_co2=0.05, of=1.0, ec_actual=300.0, ef_grid=0.0005)
    expected_act = (50.0 * 45.0 * 0.05 * 1.0) + (300.0 * 0.0005)
    assert np.isclose(e_act, expected_act)

    # E_reduced
    e_red = calc_e_reduced(e_baseline=150.0, e_actual=100.0)
    assert e_red == 50.0

    e_red_zero = calc_e_reduced(e_baseline=80.0, e_actual=100.0)
    assert e_red_zero == 0.0

    # E_removed
    e_rem = calc_e_removed(m_captured=100.0, p_co2=0.4397, eta_purity=0.95)
    assert np.isclose(e_rem, 100.0 * 0.4397 * 0.95)

def test_peng_robinson_eos():
    eos = PengRobinsonEOS()
    # At standard temp and atmospheric pressure, Z should be close to 1.0 (ideal behavior)
    z = eos.compute_z_factor(temp_k=298.15, pressure_pa=101325.0)
    assert 0.98 <= z <= 1.01

    density = eos.compute_density(temp_k=298.15, pressure_pa=101325.0)
    # CO2 density at STP is ~1.8 - 1.9 kg/m3
    assert 1.7 <= density <= 2.0

def test_stoichiometric_ratio():
    p_caco3 = get_stoichiometric_ratio("calcium_carbonate")
    assert np.isclose(p_caco3, 44.01 / 100.087, atol=1e-3)
