"""Scientific Verification Decision Engine and F_valid Derivation (Frozen Architecture).

The E-PINN verifies physics on X_core in R^26.
VerificationEngine derives F_valid in [0, 1] post-E-PINN from residual normality
and constraint satisfaction. F_valid is never an E-PINN input.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
from carbon_capture.input.validator import ValidationResult
from carbon_capture.epinn.inference import EPINNInferenceOutput
from carbon_capture.carbon_credit.calculator import CarbonCreditSummary, DeterministicCarbonCreditCalculator

@dataclass
class VerificationDecision:
    decision: str  # "VERIFIED", "NOT VERIFIED", "INSUFFICIENT DATA"
    verified_credits_tco2e: float
    provisional_credits_tco2e: float
    confidence_score: float
    F_valid: float  # AI-driven validation multiplier derived post-E-PINN
    physics_consistent: bool
    data_quality_passed: bool
    constraints_satisfied: bool
    rejection_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    audit_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "verified_credits_tco2e": self.verified_credits_tco2e,
            "provisional_credits_tco2e": self.provisional_credits_tco2e,
            "confidence_score": self.confidence_score,
            "F_valid": self.F_valid,
            "physics_consistent": self.physics_consistent,
            "data_quality_passed": self.data_quality_passed,
            "constraints_satisfied": self.constraints_satisfied,
            "rejection_reasons": self.rejection_reasons,
            "warnings": self.warnings,
            "audit_metadata": self.audit_metadata,
        }

class VerificationEngine:
    """Combines Data Hygiene, E-PINN Physics Validation, and Derives F_valid for Credit Accounting."""

    def __init__(
        self,
        max_residual_tol: float = 3.0,
        mean_residual_tol: float = 1.5,
        constraint_tol: float = 1.0e-3,
        kappa_penalty: float = 0.5,
    ):
        self.max_residual_tol = max_residual_tol
        self.mean_residual_tol = mean_residual_tol
        self.constraint_tol = constraint_tol
        self.kappa_penalty = kappa_penalty

    def calculate_f_valid(
        self,
        val_result: ValidationResult,
        epinn_output: EPINNInferenceOutput,
    ) -> float:
        """Derive F_valid in [0, 1] post-E-PINN without circularity."""
        if not val_result.valid:
            return 0.0

        max_viol = max(epinn_output.constraint_violations.values()) if epinn_output.constraint_violations else 0.0
        if max_viol > self.constraint_tol:
            return 0.0

        # Continuous penalty for residual deviation beyond 1.0 sigma
        mean_norm_res = epinn_output.mean_normalized_residual
        excess = max(0.0, mean_norm_res - 1.0)
        f_val = float(np.exp(-self.kappa_penalty * excess))
        return float(np.clip(f_val, 0.0, 1.0))

    def decide(
        self,
        val_result: ValidationResult,
        epinn_output: EPINNInferenceOutput,
        credit_summary: CarbonCreditSummary,
        run_metadata: Optional[Dict[str, Any]] = None,
    ) -> VerificationDecision:
        rejection_reasons = []
        warnings = list(val_result.warnings)

        # 1. Data Hygiene Check
        if not val_result.valid:
            rejection_reasons.extend(val_result.errors)
            return VerificationDecision(
                decision="NOT VERIFIED",
                verified_credits_tco2e=0.0,
                provisional_credits_tco2e=credit_summary.total_CC_T,
                confidence_score=0.0,
                F_valid=0.0,
                physics_consistent=False,
                data_quality_passed=False,
                constraints_satisfied=False,
                rejection_reasons=rejection_reasons,
                warnings=warnings,
                audit_metadata=run_metadata or {},
            )

        if val_result.record_count == 0:
            return VerificationDecision(
                decision="INSUFFICIENT DATA",
                verified_credits_tco2e=0.0,
                provisional_credits_tco2e=0.0,
                confidence_score=0.0,
                F_valid=0.0,
                physics_consistent=False,
                data_quality_passed=True,
                constraints_satisfied=False,
                rejection_reasons=["Input dataset is empty."],
                warnings=warnings,
                audit_metadata=run_metadata or {},
            )

        # 2. Derive F_valid
        f_valid_derived = self.calculate_f_valid(val_result, epinn_output)

        # 3. E-PINN Physics Residual Checks
        physics_consistent = epinn_output.is_physics_consistent
        if epinn_output.max_normalized_residual > self.max_residual_tol:
            rejection_reasons.append(
                f"Max normalized physics residual ({epinn_output.max_normalized_residual:.2f}) "
                f"exceeds tolerance ({self.max_residual_tol})."
            )
        if epinn_output.mean_normalized_residual > self.mean_residual_tol:
            rejection_reasons.append(
                f"Mean normalized physics residual ({epinn_output.mean_normalized_residual:.2f}) "
                f"exceeds tolerance ({self.mean_residual_tol})."
            )

        # 4. Constraint Satisfaction Checks
        max_viol = max(epinn_output.constraint_violations.values()) if epinn_output.constraint_violations else 0.0
        constraints_satisfied = max_viol <= self.constraint_tol
        if not constraints_satisfied:
            violating = {k: v for k, v in epinn_output.constraint_violations.items() if v > self.constraint_tol}
            rejection_reasons.append(f"Physical constraint violations detected: {violating}")

        # Confidence score
        norm_penalty = min(1.0, epinn_output.mean_normalized_residual / self.mean_residual_tol)
        viol_penalty = min(1.0, max_viol / 0.1)
        conf = max(0.0, 1.0 - 0.5 * norm_penalty - 0.5 * viol_penalty)

        if len(rejection_reasons) == 0 and f_valid_derived > 0.0:
            decision_status = "VERIFIED"
            # Authoritative credits scaled by the derived F_valid
            verified_credits = max(0.0, credit_summary.total_CC_T * (f_valid_derived / max(credit_summary.mean_F_valid, 1e-6)))
        else:
            decision_status = "NOT VERIFIED"
            verified_credits = 0.0

        return VerificationDecision(
            decision=decision_status,
            verified_credits_tco2e=verified_credits,
            provisional_credits_tco2e=credit_summary.total_CC_T,
            confidence_score=conf,
            F_valid=f_valid_derived,
            physics_consistent=physics_consistent,
            data_quality_passed=val_result.valid,
            constraints_satisfied=constraints_satisfied,
            rejection_reasons=rejection_reasons,
            warnings=warnings,
            audit_metadata=run_metadata or {},
        )
