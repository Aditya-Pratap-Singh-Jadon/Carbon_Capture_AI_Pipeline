"""Phase 14: Gaussian + E-PINN Verification Aggregation Layer.

Combines statistical evidence from the Gaussian Measurement Model with
physics evidence from the E-PINN and deterministic VerificationEngine.

Crucial architectural invariant:
- NO CIRCULAR DEPENDENCY: Gaussian model does not consume F_valid. E-PINN does not
  consume Gaussian likelihood. They are parallel evidence streams.
- The evidence layer exposes the statuses but does NOT invent an arbitrary
  combined verification threshold. F_valid remains the authoritative physics multiplier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from carbon_capture.anomaly.gaussian import GaussianResult
from carbon_capture.carbon_credit.verification import VerificationDecision
from carbon_capture.epinn.inference import EPINNInferenceOutput

@dataclass
class EvidenceAggregationResult:
    """Structured verification evidence representation without arbitrary combined scoring."""
    statistical_evidence: Dict[str, Any]
    physics_evidence: Dict[str, Any]
    data_quality_evidence: Dict[str, Any]
    overall_status: str
    diagnostics: List[str] = field(default_factory=list)


class GaussianEPINNVerifier:
    """
    Evidence aggregation layer for parallel Statistical and Physics verification.
    """

    def __init__(self, mahalanobis_threshold: float = 10.0):
        # NOTE: This threshold is explicitly marked for calibration per the spec.
        # It provides a diagnostic flag but does NOT override F_valid.
        self.mahalanobis_threshold = mahalanobis_threshold
        self.diagnostics = []

    def aggregate_evidence(
        self,
        gaussian_result: GaussianResult,
        epinn_inference: Optional[EPINNInferenceOutput] = None,
        verification_decision: Optional[VerificationDecision] = None,
        missing_fields: Optional[List[str]] = None,
    ) -> EvidenceAggregationResult:
        """
        Aggregate Gaussian and E-PINN evidence. Handles E-PINN incomplete inputs gracefully.
        """
        self.diagnostics = []
        missing_fields = missing_fields or []

        # 1. Statistical Evidence
        is_statistically_normal = False
        if gaussian_result.is_valid_input:
            if gaussian_result.mahalanobis_squared < self.mahalanobis_threshold:
                is_statistically_normal = True
            else:
                self.diagnostics.append(f"Statistical anomaly: Mahalanobis D^2 = {gaussian_result.mahalanobis_squared:.2f} >= {self.mahalanobis_threshold}")
        else:
            self.diagnostics.append(f"Statistical evaluation failed: {gaussian_result.diagnostics}")

        statistical_evidence = {
            "is_statistically_normal": is_statistically_normal,
            "mahalanobis_squared": gaussian_result.mahalanobis_squared,
            "log_likelihood": gaussian_result.log_likelihood,
            "statistical_score": gaussian_result.statistical_score,
            "covariance_regularized": gaussian_result.covariance_regularized,
            "evaluated_features": gaussian_result.observed_features,
            "diagnostics": gaussian_result.diagnostics,
            "threshold_requires_calibration": True,  # Explicitly required by Phase 14 spec
        }

        # 2. Physics Evidence
        physics_evidence = {
            "physics_evaluated": False,
            "is_physics_consistent": False,
            "f_valid": None,
            "verified_credits": None,
            "constraint_violations_count": None,
        }
        
        if epinn_inference is not None and verification_decision is not None:
            physics_evidence.update({
                "physics_evaluated": True,
                "is_physics_consistent": epinn_inference.is_physics_consistent,
                "f_valid": verification_decision.F_valid,
                "verified_credits": verification_decision.verified_credits_tco2e,
                "constraint_violations_count": len(epinn_inference.constraint_violations),
            })
            if not epinn_inference.is_physics_consistent:
                self.diagnostics.append("Physics inconsistency detected in E-PINN residuals/constraints.")
        else:
            self.diagnostics.append("E-PINN inference missing or blocked by incomplete input telemetry.")

        # 3. Data Quality Evidence
        data_quality_evidence = {
            "missing_fields": missing_fields,
            "gaussian_missing_features": gaussian_result.missing_features,
            "is_data_complete": len(missing_fields) == 0,
        }

        # 4. Overall Status Determination
        # We do NOT invent a "Gaussian x F_valid" score. We just classify the scenario.
        if epinn_inference is None or verification_decision is None:
            overall_status = "E_PINN_INPUT_INCOMPLETE"
        elif not data_quality_evidence["is_data_complete"]:
            overall_status = "INSUFFICIENT_DATA"
        elif is_statistically_normal and physics_evidence["is_physics_consistent"]:
            overall_status = "VERIFIED_NORMAL"
        elif not is_statistically_normal and physics_evidence["is_physics_consistent"]:
            overall_status = "VERIFIED_STATISTICALLY_ANOMALOUS"
        elif is_statistically_normal and not physics_evidence["is_physics_consistent"]:
            overall_status = "REJECTED_PHYSICALLY_INCONSISTENT"
        else:
            overall_status = "REJECTED_ANOMALOUS_AND_INCONSISTENT"

        return EvidenceAggregationResult(
            statistical_evidence=statistical_evidence,
            physics_evidence=physics_evidence,
            data_quality_evidence=data_quality_evidence,
            overall_status=overall_status,
            diagnostics=self.diagnostics,
        )
