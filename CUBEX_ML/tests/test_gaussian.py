import os
import tempfile
import numpy as np
import pytest
from pathlib import Path

from carbon_capture.anomaly.gaussian import (
    MultivariateGaussianModel, 
    GaussianResult,
    GAUSSIAN_FEATURES
)
from carbon_capture.anomaly.gaussian_epinn_verification import (
    GaussianEPINNVerifier,
    EvidenceAggregationResult
)
from carbon_capture.epinn.inference import EPINNInferenceOutput
from carbon_capture.carbon_credit.verification import VerificationDecision

# ---------------------------------------------------------------------------
# PHASE 13 TESTS
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_data():
    np.random.seed(42)
    # 100 samples, 6 features
    return np.random.randn(100, 6) * 2.0 + 10.0

@pytest.fixture
def correlated_data():
    np.random.seed(42)
    x1 = np.random.randn(100)
    x2 = 2.0 * x1 + np.random.randn(100) * 0.1
    data = np.random.randn(100, 6)
    data[:, 0] = x1
    data[:, 1] = x2
    return data

def make_record(values):
    return {feat: val for feat, val in zip(GAUSSIAN_FEATURES, values)}

def test_known_mean_covariance(clean_data):
    model = MultivariateGaussianModel()
    model.calibrate(clean_data)
    assert np.allclose(model.mu, np.mean(clean_data, axis=0))
    assert np.allclose(model.sigma, np.cov(clean_data, rowvar=False))

def test_mahalanobis_distance_and_likelihood(clean_data):
    model = MultivariateGaussianModel()
    model.calibrate(clean_data)
    
    # Test on the mean (distance should be ~0)
    record = make_record(model.mu)
    res = model.evaluate_single(record)
    assert res.is_valid_input
    assert res.mahalanobis_squared < 1e-6
    # log_lik = -0.5 * (k*ln(2pi) + ln|Sigma| + D^2)
    # We can just check it's a valid float
    assert not np.isnan(res.log_likelihood)

def test_correlated_variables(correlated_data):
    model = MultivariateGaussianModel()
    model.calibrate(correlated_data)
    assert model.is_calibrated
    
    rec = make_record(correlated_data[0])
    res = model.evaluate_single(rec)
    assert res.is_valid_input
    assert res.mahalanobis_squared > 0

def test_outlier_observation(clean_data):
    model = MultivariateGaussianModel()
    model.calibrate(clean_data)
    
    normal_rec = make_record(model.mu)
    res_normal = model.evaluate_single(normal_rec)
    
    outlier = model.mu + np.array([100.0, -100.0, 50.0, 0, 0, 0])
    outlier_rec = make_record(outlier)
    res_outlier = model.evaluate_single(outlier_rec)
    
    assert res_outlier.mahalanobis_squared > res_normal.mahalanobis_squared

def test_singular_covariance():
    data = np.ones((100, 6)) # all ones -> 0 variance -> singular
    model = MultivariateGaussianModel(regularization=1e-3)
    model.calibrate(data)
    assert model.was_regularized
    
    rec = make_record(data[0])
    res = model.evaluate_single(rec)
    assert res.is_valid_input

def test_near_singular_covariance(correlated_data):
    # Make feature 2 exactly a multiple of feature 0
    correlated_data[:, 2] = correlated_data[:, 0] * 5.0
    model = MultivariateGaussianModel(regularization=1e-3)
    model.calibrate(correlated_data)
    assert model.was_regularized
    assert model.is_calibrated

def test_insufficient_calibration_samples():
    data = np.random.randn(5, 6) # < 7
    model = MultivariateGaussianModel()
    with pytest.raises(ValueError, match="Insufficient calibration samples"):
        model.calibrate(data)

def test_missing_value(clean_data):
    model = MultivariateGaussianModel()
    model.calibrate(clean_data)
    
    rec = make_record(model.mu)
    rec["CO2_ppm"] = None
    res = model.evaluate_single(rec)
    assert res.is_valid_input
    assert "CO2_ppm" in res.missing_features
    assert "Evaluated on 5 features" in res.diagnostics

def test_dropout_sentinel(clean_data):
    model = MultivariateGaussianModel()
    model.calibrate(clean_data)
    
    rec = make_record(model.mu)
    rec["Temperature_C"] = np.nan
    res = model.evaluate_single(rec)
    assert res.is_valid_input
    assert "Temperature_C" in res.missing_features

def test_model_save_load(clean_data):
    model = MultivariateGaussianModel()
    model.calibrate(clean_data)
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_name = tmp.name
        
    try:
        model.save(tmp_name)
        new_model = MultivariateGaussianModel()
        new_model.load(tmp_name)
        assert np.allclose(model.mu, new_model.mu)
        assert np.allclose(model.sigma, new_model.sigma)
    finally:
        os.remove(tmp_name)

def test_deterministic_calibration(clean_data):
    m1 = MultivariateGaussianModel()
    m1.calibrate(clean_data)
    m2 = MultivariateGaussianModel()
    m2.calibrate(clean_data)
    assert np.allclose(m1.mu, m2.mu)
    assert np.allclose(m1.sigma, m2.sigma)

def test_dimension_mismatch(clean_data):
    model = MultivariateGaussianModel()
    with pytest.raises(ValueError):
        model.calibrate(clean_data[:, :5])

def test_malformed_input(clean_data):
    model = MultivariateGaussianModel()
    model.calibrate(clean_data)
    # Completely missing all
    rec = {feat: None for feat in GAUSSIAN_FEATURES}
    res = model.evaluate_single(rec)
    assert not res.is_valid_input

# ---------------------------------------------------------------------------
# PHASE 14 TESTS
# ---------------------------------------------------------------------------

def make_epinn_output(consistent=True):
    return EPINNInferenceOutput(
        predictions={},
        raw_residuals={},
        normalized_residuals={},
        residual_scales={},
        constraint_violations={} if consistent else {"c1": 1.0},
        is_physics_consistent=consistent,
        mean_normalized_residual=0.0,
        max_normalized_residual=0.0
    )

def make_verification_decision(consistent=True):
    return VerificationDecision(
        decision="VERIFIED" if consistent else "REJECTED",
        verified_credits_tco2e=10.0 if consistent else 0.0,
        provisional_credits_tco2e=10.0,
        confidence_score=0.9,
        F_valid=1.0 if consistent else 0.0,
        physics_consistent=consistent,
        data_quality_passed=True,
        constraints_satisfied=consistent
    )

def test_integration_statistically_normal_physically_consistent():
    verifier = GaussianEPINNVerifier(mahalanobis_threshold=10.0)
    g_res = GaussianResult(True, 5.0, -10.0, 0.9, False, [], [], "")
    epinn = make_epinn_output(True)
    vd = make_verification_decision(True)
    
    res = verifier.aggregate_evidence(g_res, epinn, vd, [])
    assert res.overall_status == "VERIFIED_NORMAL"
    assert res.physics_evidence["f_valid"] == 1.0

def test_integration_statistically_anomalous_physically_consistent():
    verifier = GaussianEPINNVerifier(mahalanobis_threshold=10.0)
    g_res = GaussianResult(True, 15.0, -20.0, 0.1, False, [], [], "")
    epinn = make_epinn_output(True)
    vd = make_verification_decision(True)
    
    res = verifier.aggregate_evidence(g_res, epinn, vd, [])
    assert res.overall_status == "VERIFIED_STATISTICALLY_ANOMALOUS"

def test_integration_statistically_normal_physically_inconsistent():
    verifier = GaussianEPINNVerifier(mahalanobis_threshold=10.0)
    g_res = GaussianResult(True, 5.0, -10.0, 0.9, False, [], [], "")
    epinn = make_epinn_output(False)
    vd = make_verification_decision(False)
    
    res = verifier.aggregate_evidence(g_res, epinn, vd, [])
    assert res.overall_status == "REJECTED_PHYSICALLY_INCONSISTENT"

def test_integration_statistically_anomalous_physically_inconsistent():
    verifier = GaussianEPINNVerifier(mahalanobis_threshold=10.0)
    g_res = GaussianResult(True, 15.0, -20.0, 0.1, False, [], [], "")
    epinn = make_epinn_output(False)
    vd = make_verification_decision(False)
    
    res = verifier.aggregate_evidence(g_res, epinn, vd, [])
    assert res.overall_status == "REJECTED_ANOMALOUS_AND_INCONSISTENT"

def test_integration_incomplete_epinn_input():
    verifier = GaussianEPINNVerifier(mahalanobis_threshold=10.0)
    g_res = GaussianResult(True, 5.0, -10.0, 0.9, False, [], [], "")
    # Missing epinn and verification due to E-PINN 26-variable block
    res = verifier.aggregate_evidence(g_res, None, None, ["FC_actual"])
    assert res.overall_status == "E_PINN_INPUT_INCOMPLETE"
    assert res.statistical_evidence["is_statistically_normal"] == True

def test_integration_sensor_dropout():
    verifier = GaussianEPINNVerifier(mahalanobis_threshold=10.0)
    # Gaussian works but with missing feature
    g_res = GaussianResult(True, 5.0, -10.0, 0.9, False, ["CO2_ppm"], ["Temperature_C"], "")
    epinn = make_epinn_output(True)
    vd = make_verification_decision(True)
    
    res = verifier.aggregate_evidence(g_res, epinn, vd, ["Temperature_C"])
    assert res.overall_status == "INSUFFICIENT_DATA"
    assert not res.data_quality_evidence["is_data_complete"]

def test_no_circular_dependency():
    # Verify that the F_valid remains unchanged and is passed through
    verifier = GaussianEPINNVerifier(mahalanobis_threshold=10.0)
    g_res = GaussianResult(True, 999.0, -20.0, 0.1, False, [], [], "")  # Extremely anomalous
    epinn = make_epinn_output(True)
    vd = make_verification_decision(True)  # F_valid = 1.0
    
    res = verifier.aggregate_evidence(g_res, epinn, vd, [])
    assert res.physics_evidence["f_valid"] == 1.0
    # The anomaly must NOT touch the physics score.
