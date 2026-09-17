"""Unit tests for Preprocessing, Scaling, and Feature Extraction."""

import pytest
import pandas as pd
import numpy as np
from carbon_capture.input.canonical_order import CANONICAL_INPUT_ORDER, CORE_PHYSICAL_INPUT_ORDER
from carbon_capture.preprocessing.cleaning import DataCleaner
from carbon_capture.preprocessing.scaling import CanonicalScaler
from carbon_capture.preprocessing.feature_engineering import PhysicsFeatureExtractor

@pytest.fixture
def sample_df():
    data = {col: np.linspace(10.0, 50.0, 20) for col in CANONICAL_INPUT_ORDER}
    return pd.DataFrame(data)

def test_canonical_scaler_train_only_fitting(sample_df):
    train_df = sample_df.iloc[:15]
    test_df = sample_df.iloc[15:]

    scaler = CanonicalScaler()
    assert not scaler.is_fitted

    scaler.fit(train_df)
    assert scaler.is_fitted

    scaled_train = scaler.transform(train_df)
    scaled_test = scaler.transform(test_df)

    assert scaled_train.shape == (15, 26)
    assert scaled_test.shape == (5, 26)
    # Check mean of train scaled is ~0
    assert np.allclose(scaled_train.mean(axis=0), 0.0, atol=1e-5)

def test_scaler_inverse_transform(sample_df):
    scaler = CanonicalScaler()
    scaler.fit(sample_df)
    scaled = scaler.transform(sample_df)
    recovered = scaler.inverse_transform(scaled)

    for col in CORE_PHYSICAL_INPUT_ORDER:
        assert np.allclose(sample_df[col], recovered[col], atol=1e-5)

def test_feature_extractor(sample_df):
    extractor = PhysicsFeatureExtractor()
    feat_df = extractor.transform(sample_df)

    assert "P_aux_total" in feat_df.columns
    assert "transport_ton_km" in feat_df.columns
    assert np.allclose(feat_df["P_aux_total"], sample_df["P_fan"] + sample_df["P_pump"])
