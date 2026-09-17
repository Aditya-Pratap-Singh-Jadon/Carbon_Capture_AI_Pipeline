"""Unit tests for Deterministic Carbon Credit Engine and Verification Logic."""

import pytest
import pandas as pd
import numpy as np
from carbon_capture.carbon_credit.calculator import DeterministicCarbonCreditCalculator
from carbon_capture.carbon_credit.verification import VerificationEngine
from carbon_capture.input.synthetic import SyntheticDataGenerator
from carbon_capture.input.validator import ValidationResult
from carbon_capture.epinn.inference import EPINNInferenceOutput

def test_deterministic_calculator():
    gen = SyntheticDataGenerator(seed=42)
    df = gen.generate_normal_regime(n_samples=5)

    calc = DeterministicCarbonCreditCalculator()
    summary = calc.calculate_from_dataframe(df)

    assert summary.record_count == 5
    assert summary.total_CC_T > 0.0
    assert len(summary.records) == 5

    # Test audit record breakdown
    rec0 = summary.records[0]
    expected_bracket = rec0.E_reduced + rec0.E_removed + rec0.E_displaced - rec0.PE_aux - rec0.PE_lifecycle - rec0.L_leakage
    assert np.isclose(rec0.net_benefit_bracket, expected_bracket)
    assert np.isclose(rec0.CC_T, expected_bracket * rec0.F_multiplier)

def test_verification_decision_engine():
    engine = VerificationEngine()
    gen = SyntheticDataGenerator()
    df = gen.generate_normal_regime(n_samples=5)
    calc = DeterministicCarbonCreditCalculator()
    credit_summary = calc.calculate_from_dataframe(df)

    # 1. Clean case
    val_res = ValidationResult(valid=True, record_count=5)
    epinn_out = EPINNInferenceOutput(
        predictions={},
        raw_residuals={},
        normalized_residuals={"r_CC": np.array([0.1])},
        residual_scales={},
        constraint_violations={"c1": 0.0},
        is_physics_consistent=True,
        mean_normalized_residual=0.2,
        max_normalized_residual=0.5,
    )
    decision = engine.decide(val_res, epinn_out, credit_summary)
    assert decision.decision == "VERIFIED"
    assert decision.verified_credits_tco2e > 0.0

    # 2. Inconsistent physics case
    epinn_bad = EPINNInferenceOutput(
        predictions={},
        raw_residuals={},
        normalized_residuals={"r_CC": np.array([4.5])},
        residual_scales={},
        constraint_violations={"c1": 0.0},
        is_physics_consistent=False,
        mean_normalized_residual=2.8,
        max_normalized_residual=4.5,
    )
    bad_dec = engine.decide(val_res, epinn_bad, credit_summary)
    assert bad_dec.decision == "NOT VERIFIED"
    assert bad_dec.verified_credits_tco2e == 0.0
