"""Unit tests for Input Schema, Canonical Order, and Validation."""

import pytest
import pandas as pd
import numpy as np
from carbon_capture.input.canonical_order import CANONICAL_INPUT_ORDER
from carbon_capture.input.schema import InputSchema
from carbon_capture.input.validator import InputValidator
from carbon_capture.input.loader import DataLoader
from carbon_capture.utils.exceptions import SchemaValidationError

@pytest.fixture
def valid_row():
    row = {col: 10.0 for col in CANONICAL_INPUT_ORDER}
    row["OF"] = 0.99
    row["P_CO2"] = 0.4397
    row["eta_purity"] = 0.95
    row["eta_boiler"] = 0.85
    row["gamma_slip"] = 0.02
    row["F_SME"] = 1.05
    row["F_perm"] = 0.98
    row["alpha"] = 0.90
    return row

def test_canonical_order_length():
    assert len(CANONICAL_INPUT_ORDER) == 29

def test_loader_shuffled_order(valid_row):
    # Create DataFrame with scrambled column order
    df_orig = pd.DataFrame([valid_row])
    shuffled_cols = list(np.random.permutation(CANONICAL_INPUT_ORDER))
    df_shuffled = df_orig[shuffled_cols]
    
    loader = DataLoader()
    reordered_df, orig_cols, extra = loader.canonicalize_columns(df_shuffled)
    
    assert list(reordered_df.columns[:29]) == CANONICAL_INPUT_ORDER
    assert orig_cols == shuffled_cols

def test_missing_column_rejected(valid_row):
    del valid_row["FC_actual"]
    df_missing = pd.DataFrame([valid_row])
    loader = DataLoader()
    with pytest.raises(SchemaValidationError):
        loader.canonicalize_columns(df_missing)

def test_nan_rejection(valid_row):
    valid_row["eta_purity"] = np.nan
    df_nan = pd.DataFrame([valid_row])
    validator = InputValidator()
    res = validator.validate(df_nan)
    assert not res.valid
    assert any("NaN" in e for e in res.errors)

def test_physical_bound_violation(valid_row):
    # Purity cannot exceed 1.0
    valid_row["eta_purity"] = 1.25
    df_bad = pd.DataFrame([valid_row])
    validator = InputValidator()
    res = validator.validate(df_bad)
    assert not res.valid
    assert any("eta_purity" in e for e in res.errors)
