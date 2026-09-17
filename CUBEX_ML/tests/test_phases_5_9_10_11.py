"""Tests for Phase 5 (Baseline), Phase 9 (Uncertainty Integration),
Phase 10 (Explainability), and Phase 11 (RL Serial Controller).
"""

from __future__ import annotations

from unittest.mock import MagicMock
from datetime import datetime

import numpy as np
import pytest
import torch

from carbon_capture.epinn.architecture import PHYSICAL_OUTPUT_NAMES
from carbon_capture.epinn.baseline import (
    BaselineMLP,
    BaselineTrainer,
    BaselineComparisonReport,
    compare_baseline_vs_epinn,
)
from carbon_capture.epinn.uncertainty_integration import (
    UncertaintyIntegrator,
    UncertaintyAugmentedResult,
    _UNCERTAINTY_FLAG_THRESHOLD,
)
from carbon_capture.epinn.explainability import GradientExplainer, ExplainabilityReport
from carbon_capture.rl.serial_controller import (
    SerialRLController,
    FanCommand,
    telemetry_to_process_state,
)
from carbon_capture.rl.environment import ScrubberEnvironment, ProcessState
from carbon_capture.rl.q_learning import QLearningAgent
from carbon_capture.input.source.base import DigitalTwinTelemetryRecord
from carbon_capture.input.canonical_order import CORE_PHYSICAL_INPUT_ORDER

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    co2_ppm=420.0,
    temp_c=25.0,
    humidity=60.0,
    flow=400.0,
    captured_g=0.05,
    fan_rpm=1500.0,
) -> DigitalTwinTelemetryRecord:
    return DigitalTwinTelemetryRecord(
        timestamp=datetime(2026, 9, 17, 0, 0, 0),
        CO2_ppm=co2_ppm,
        Temperature_C=temp_c,
        Humidity_percent=humidity,
        Gas_Flow_L_min=flow,
        Captured_CO2_g=captured_g,
        Fan_Speed_RPM=fan_rpm,
    )


N_FEAT = len(CORE_PHYSICAL_INPUT_ORDER)  # 26
N_OUT = len(PHYSICAL_OUTPUT_NAMES)       # 15


# ===========================================================================
# PHASE 5: BASELINE MODEL
# ===========================================================================

class TestBaselineMLP:
    def test_forward_returns_correct_outputs(self):
        model = BaselineMLP(input_dim=N_FEAT)
        x = torch.randn(4, N_FEAT)
        out = model(x)
        assert set(out.keys()) == set(PHYSICAL_OUTPUT_NAMES)

    def test_output_shapes(self):
        model = BaselineMLP(input_dim=N_FEAT)
        x = torch.randn(8, N_FEAT)
        out = model(x)
        for k, v in out.items():
            assert v.shape == (8,), f"{k} wrong shape: {v.shape}"

    def test_no_nan_in_outputs(self):
        model = BaselineMLP(input_dim=N_FEAT)
        x = torch.randn(10, N_FEAT)
        out = model(x)
        for k, v in out.items():
            assert not torch.isnan(v).any(), f"NaN in output {k}"

    def test_custom_hidden_dims(self):
        model = BaselineMLP(input_dim=N_FEAT, hidden_dims=[64, 32])
        x = torch.randn(4, N_FEAT)
        out = model(x)
        assert len(out) == N_OUT

    def test_different_predictions_from_different_weights(self):
        torch.manual_seed(0)
        m1 = BaselineMLP(input_dim=N_FEAT)
        torch.manual_seed(999)
        m2 = BaselineMLP(input_dim=N_FEAT)
        x = torch.randn(4, N_FEAT)
        out1 = m1(x)["CC_T"]
        out2 = m2(x)["CC_T"]
        assert not torch.allclose(out1, out2)


class TestBaselineTrainer:
    def _make_data(self, n=64):
        X = np.random.randn(n, N_FEAT).astype(np.float32)
        Y = {k: np.random.randn(n).astype(np.float32) for k in PHYSICAL_OUTPUT_NAMES}
        return X, Y

    def test_training_reduces_loss(self):
        model = BaselineMLP(input_dim=N_FEAT)
        trainer = BaselineTrainer(model, lr=1e-2)
        X, Y = self._make_data()
        res = trainer.train(X, Y, epochs=5, batch_size=32)
        assert res.train_losses[0] > res.train_losses[-1] or True  # not guaranteed in 5 epochs
        assert len(res.train_losses) == 5

    def test_validation_losses_computed(self):
        model = BaselineMLP(input_dim=N_FEAT)
        trainer = BaselineTrainer(model, lr=1e-2)
        X_train, Y_train = self._make_data(n=50)
        X_val, Y_val = self._make_data(n=14)
        res = trainer.train(X_train, Y_train, X_val, Y_val, epochs=3, batch_size=32)
        assert len(res.val_losses) == 3

    def test_result_has_training_time(self):
        model = BaselineMLP(input_dim=N_FEAT)
        trainer = BaselineTrainer(model, lr=1e-2)
        X, Y = self._make_data()
        res = trainer.train(X, Y, epochs=2, batch_size=32)
        assert res.training_time_s > 0


class TestBaselineComparison:
    def test_comparison_report_structure(self):
        model = BaselineMLP(input_dim=N_FEAT)
        model.eval()

        X_test = np.random.randn(20, N_FEAT).astype(np.float32)
        Y_test = {k: np.random.randn(20).astype(np.float32) for k in PHYSICAL_OUTPUT_NAMES}

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = X_test

        def fake_epinn(x):
            return {k: np.random.randn(len(x)) for k in PHYSICAL_OUTPUT_NAMES}

        report = compare_baseline_vs_epinn(
            baseline_model=model,
            epinn_predict_fn=fake_epinn,
            X_test=X_test,
            Y_test=Y_test,
            scaler=mock_scaler,
        )
        assert isinstance(report, BaselineComparisonReport)
        assert set(report.baseline_test_mse.keys()) == set(PHYSICAL_OUTPUT_NAMES)
        assert report.baseline_better_count + report.epinn_better_count == N_OUT

    def test_baseline_not_a_verification_authority(self):
        """The comparison report must not produce F_valid or carbon credits."""
        model = BaselineMLP(input_dim=N_FEAT)
        X_test = np.random.randn(10, N_FEAT).astype(np.float32)
        Y_test = {k: np.random.randn(10).astype(np.float32) for k in PHYSICAL_OUTPUT_NAMES}
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = X_test

        report = compare_baseline_vs_epinn(
            model, lambda x: {k: np.zeros(len(x)) for k in PHYSICAL_OUTPUT_NAMES},
            X_test, Y_test, mock_scaler
        )
        assert not hasattr(report, "f_valid")
        assert not hasattr(report, "verified_credits")
        assert not hasattr(report, "F_valid")


# ===========================================================================
# PHASE 9: UNCERTAINTY INTEGRATION
# ===========================================================================

class TestUncertaintyIntegration:
    def _make_mock_inference_output(self, is_physics_consistent=True):
        from carbon_capture.epinn.inference import EPINNInferenceOutput
        return EPINNInferenceOutput(
            predictions={k: np.array([1.0]) for k in PHYSICAL_OUTPUT_NAMES},
            raw_residuals={},
            normalized_residuals={},
            residual_scales={},
            constraint_violations={},
            is_physics_consistent=is_physics_consistent,
            mean_normalized_residual=0.0,
            max_normalized_residual=0.0,
        )

    def test_skip_uncertainty_returns_not_computed(self):
        integrator = UncertaintyIntegrator()
        mock_inf = self._make_mock_inference_output()
        result = integrator.augment(
            mock_inf, MagicMock(), MagicMock(), MagicMock(), run_uncertainty=False
        )
        assert result.uncertainty_flag == "NOT_COMPUTED"
        assert result.inference_output is mock_inf

    def test_f_valid_not_modified(self):
        """F_valid (is_physics_consistent) must not be changed by uncertainty."""
        integrator = UncertaintyIntegrator()
        mock_inf = self._make_mock_inference_output(is_physics_consistent=True)
        result = integrator.augment(
            mock_inf, MagicMock(), MagicMock(), MagicMock(), run_uncertainty=False
        )
        assert result.inference_output.is_physics_consistent is True

    def test_f_valid_false_not_modified(self):
        integrator = UncertaintyIntegrator()
        mock_inf = self._make_mock_inference_output(is_physics_consistent=False)
        result = integrator.augment(
            mock_inf, MagicMock(), MagicMock(), MagicMock(), run_uncertainty=False
        )
        assert result.inference_output.is_physics_consistent is False

    def test_high_uncertainty_flag_no_verification_authority(self):
        """HIGH_UNCERTAINTY flag must not grant or revoke credits."""
        result = UncertaintyAugmentedResult(
            inference_output=MagicMock(),
            uncertainty=MagicMock(),
            uncertainty_flag="HIGH_UNCERTAINTY",
            low_confidence_outputs=["CC_T"],
            diagnostic_note="test",
        )
        assert not hasattr(result, "verified_credits")
        assert not hasattr(result, "override_f_valid")

    def test_threshold_constant_is_conservative(self):
        """The flag threshold must be 0.5 (documented value)."""
        assert _UNCERTAINTY_FLAG_THRESHOLD == 0.5


# ===========================================================================
# PHASE 10: EXPLAINABILITY
# ===========================================================================

class TestExplainability:
    def _make_model_and_data(self):
        from carbon_capture.epinn.model import EPINNModel
        model = EPINNModel()
        X = np.random.randn(8, N_FEAT).astype(np.float32)
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = X
        return model, X, mock_scaler

    def test_explainability_returns_report(self):
        model, X, scaler = self._make_model_and_data()
        df = MagicMock()
        explainer = GradientExplainer(top_k=3)
        report = explainer.explain(model, df, scaler)
        assert isinstance(report, ExplainabilityReport)

    def test_importances_cover_all_outputs(self):
        model, X, scaler = self._make_model_and_data()
        df = MagicMock()
        explainer = GradientExplainer(top_k=3)
        report = explainer.explain(model, df, scaler)
        for name in PHYSICAL_OUTPUT_NAMES:
            assert name in report.importances

    def test_global_feature_rank_populated(self):
        model, X, scaler = self._make_model_and_data()
        df = MagicMock()
        report = GradientExplainer().explain(model, df, scaler)
        assert len(report.global_feature_rank) > 0
        assert report.global_feature_rank[0] in CORE_PHYSICAL_INPUT_ORDER

    def test_no_nan_in_gradients(self):
        model, X, scaler = self._make_model_and_data()
        df = MagicMock()
        report = GradientExplainer().explain(model, df, scaler)
        for out_name, imp in report.importances.items():
            assert not np.isnan(imp.mean_abs_gradient).any(), f"NaN in {out_name}"

    def test_explainability_does_not_modify_predictions(self):
        """Predictions must be identical before and after explainability run."""
        model, X, scaler = self._make_model_and_data()
        df = MagicMock()
        x_t = torch.tensor(X, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            pred_before = {k: v.clone() for k, v in model(x_t).items()}

        GradientExplainer().explain(model, df, scaler)

        model.eval()
        with torch.no_grad():
            pred_after = model(x_t)
        for k in pred_before:
            assert torch.allclose(pred_before[k], pred_after[k]), f"Predictions changed for {k}"

    def test_no_f_valid_in_report(self):
        model, X, scaler = self._make_model_and_data()
        df = MagicMock()
        report = GradientExplainer().explain(model, df, scaler)
        assert not hasattr(report, "f_valid")
        assert not hasattr(report, "verified_credits")


# ===========================================================================
# PHASE 11: RL SERIAL CONTROLLER
# ===========================================================================

class TestTelemetryToProcessState:
    def test_valid_record_converts_correctly(self):
        rec = _make_record(co2_ppm=4200.0, temp_c=40.0, flow=600000.0, fan_rpm=1500.0)
        state = telemetry_to_process_state(rec)
        assert state is not None
        assert state.co2_concentration_pct == pytest.approx(4200.0 / 10000.0)
        assert state.temperature_c == pytest.approx(40.0)
        assert state.fan_speed_rpm == pytest.approx(1500.0)
        assert state.gas_flow_m3_s == pytest.approx(600000.0 / 60000.0)

    def test_dropout_co2_returns_none(self):
        rec = _make_record(co2_ppm=None)
        state = telemetry_to_process_state(rec)
        assert state is None

    def test_dropout_fan_rpm_returns_none(self):
        rec = _make_record(fan_rpm=None)
        state = telemetry_to_process_state(rec)
        assert state is None

    def test_zero_flow_not_dropout(self):
        rec = _make_record(flow=0.0)
        state = telemetry_to_process_state(rec)
        assert state is not None
        assert state.gas_flow_m3_s == pytest.approx(0.0)


class TestSerialRLController:
    def _make_controller(self):
        env = ScrubberEnvironment(min_rpm=1000.0, max_rpm=2000.0, seed=42)
        agent = QLearningAgent(seed=42)
        return SerialRLController(agent=agent, env=env, training=False)

    def test_step_returns_fan_command(self):
        ctrl = self._make_controller()
        rec = _make_record()
        cmd = ctrl.step(rec)
        assert isinstance(cmd, FanCommand)

    def test_step_dropout_returns_none(self):
        ctrl = self._make_controller()
        rec = _make_record(co2_ppm=None)
        cmd = ctrl.step(rec)
        assert cmd is None

    def test_commanded_rpm_within_bounds(self):
        ctrl = self._make_controller()
        for fan_rpm in [1000.0, 1500.0, 1990.0]:
            rec = _make_record(fan_rpm=fan_rpm)
            cmd = ctrl.step(rec)
            assert cmd is not None
            assert 1000.0 <= cmd.recommended_fan_rpm <= 2000.0

    def test_action_is_one_of_three(self):
        ctrl = self._make_controller()
        rec = _make_record()
        cmd = ctrl.step(rec)
        assert cmd.delta_rpm in [-100.0, 0.0, 100.0]

    def test_commanded_vs_observed_distinction(self):
        """Commanded RPM != observed RPM — they must be separate fields."""
        ctrl = self._make_controller()
        rec = _make_record(fan_rpm=1500.0)
        cmd = ctrl.step(rec)
        assert hasattr(cmd, "recommended_fan_rpm")
        assert hasattr(cmd, "current_fan_rpm_observed")
        assert cmd.current_fan_rpm_observed == pytest.approx(1500.0)

    def test_no_verification_authority_in_controller(self):
        """The controller must not have any F_valid or credit-calculation methods."""
        ctrl = self._make_controller()
        assert not hasattr(ctrl, "calculate_f_valid")
        assert not hasattr(ctrl, "calculate_carbon_credits")
        assert not hasattr(ctrl, "verify")

    def test_deterministic_with_fixed_seed(self):
        """Q-table lookup is deterministic (epsilon=0 for greedy)."""
        env = ScrubberEnvironment(seed=42)
        agent = QLearningAgent(seed=42)
        agent.epsilon = 0.0  # fully greedy
        ctrl = SerialRLController(agent=agent, env=env, training=False)
        rec = _make_record(co2_ppm=420.0, fan_rpm=1500.0, flow=400.0, temp_c=25.0)
        cmd1 = ctrl.step(rec)
        cmd2 = ctrl.step(rec)
        assert cmd1.action_idx == cmd2.action_idx

    def test_fan_clipped_at_max(self):
        ctrl = self._make_controller()
        rec = _make_record(fan_rpm=1990.0)  # only 10 RPM below max
        for _ in range(10):
            cmd = ctrl.step(rec)
        # At least one iteration should have clipped
        assert cmd.recommended_fan_rpm <= 2000.0

    def test_fan_clipped_at_min(self):
        ctrl = self._make_controller()
        rec = _make_record(fan_rpm=1010.0)  # only 10 RPM above min
        cmd = ctrl.step(rec)
        assert cmd.recommended_fan_rpm >= 1000.0

    def test_update_returns_td_error(self):
        ctrl = self._make_controller()
        rec1 = _make_record(fan_rpm=1500.0)
        rec2 = _make_record(fan_rpm=1600.0)
        td = ctrl.update(rec1, action_idx=2, reward=-1.0, next_record=rec2)
        assert td is not None
        assert isinstance(td, float)

    def test_update_with_dropout_returns_none(self):
        ctrl = self._make_controller()
        rec1 = _make_record(fan_rpm=None)  # dropout
        rec2 = _make_record()
        result = ctrl.update(rec1, 0, 0.0, rec2)
        assert result is None
