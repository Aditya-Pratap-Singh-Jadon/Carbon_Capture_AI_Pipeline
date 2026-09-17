"""Audit Trail and Verification Record Builder."""

import datetime
import hashlib
import json
from typing import Any, Dict, List, Optional
from carbon_capture.input.validator import ValidationResult
from carbon_capture.epinn.inference import EPINNInferenceOutput
from carbon_capture.carbon_credit.calculator import CarbonCreditSummary
from carbon_capture.carbon_credit.verification import VerificationDecision

class AuditReportBuilder:
    """Constructs comprehensive auditable verification packages."""

    def __init__(self, pipeline_version: str = "1.0.0", model_version: str = "1.0.0"):
        self.pipeline_version = pipeline_version
        self.model_version = model_version

    def build_audit_record(
        self,
        val_result: ValidationResult,
        epinn_output: EPINNInferenceOutput,
        credit_summary: CarbonCreditSummary,
        decision: VerificationDecision,
        input_file: Optional[str] = None,
        pe_aux_mode: str = "verbatim",
    ) -> Dict[str, Any]:
        """Assemble full audit structure for regulatory and third-party inspection."""
        run_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        run_id = hashlib.sha256(f"{run_timestamp}_{input_file}".encode()).hexdigest()[:16]

        record = {
            "audit_header": {
                "run_id": run_id,
                "timestamp_utc": run_timestamp,
                "pipeline_version": self.pipeline_version,
                "model_version": self.model_version,
                "input_source": str(input_file),
            },
            "verification_decision": {
                "status": decision.decision,
                "verified_credits_tco2e": decision.verified_credits_tco2e,
                "provisional_credits_tco2e": decision.provisional_credits_tco2e,
                "confidence_score": decision.confidence_score,
                "physics_consistent": decision.physics_consistent,
                "data_quality_passed": decision.data_quality_passed,
                "constraints_satisfied": decision.constraints_satisfied,
                "rejection_reasons": decision.rejection_reasons,
                "warnings": decision.warnings,
            },
            "input_data_integrity": {
                "record_count": val_result.record_count,
                "original_columns": val_result.original_columns,
                "canonical_columns": val_result.canonical_columns,
                "data_valid": val_result.valid,
                "hygiene_corrections": val_result.corrections,
                "hygiene_errors": val_result.errors,
            },
            "epinn_physics_validation": {
                "mean_normalized_residual": epinn_output.mean_normalized_residual,
                "max_normalized_residual": epinn_output.max_normalized_residual,
                "is_physics_consistent": epinn_output.is_physics_consistent,
                "raw_residuals_mean": {
                    k: float(epinn_output.raw_residuals[k].mean())
                    for k in epinn_output.raw_residuals
                },
                "normalized_residuals_mean": {
                    k: float(epinn_output.normalized_residuals[k].mean())
                    for k in epinn_output.normalized_residuals
                },
                "residual_normalization_scales": epinn_output.residual_scales,
                "constraint_violations": epinn_output.constraint_violations,
            },
            "deterministic_credit_accounting": {
                "CC_T_total": credit_summary.total_CC_T,
                "benefit_terms": {
                    "E_reduced": credit_summary.total_E_reduced,
                    "E_removed": credit_summary.total_E_removed,
                    "E_displaced": credit_summary.total_E_displaced,
                },
                "penalty_terms": {
                    "PE_aux": credit_summary.total_PE_aux,
                    "PE_lifecycle": credit_summary.total_PE_lifecycle,
                    "L_leakage": credit_summary.total_L_leakage,
                },
                "mean_multiplier_product": credit_summary.mean_F_multiplier,
            },
            "scientific_assumptions_and_ambiguities": [
                {
                    "issue": "PE_aux dimensional inconsistency",
                    "status": "Documented & Preserved",
                    "mode_active": pe_aux_mode,
                    "note": "Section C specifies PE_aux = EC_hardware - EF_grid. Dimensionally inconsistent (kWh vs tCO2/kWh). Preserved verbatim per mathematical specification.",
                },
                {
                    "issue": "Provenance of F_valid",
                    "status": "Documented & Configurable",
                    "note": "Documented as AI-Driven Validation Multiplier in [0, 1]. Handled modularly as observed telemetry input or co-prediction.",
                },
                {
                    "issue": "F_SME One-sided bound",
                    "status": "Documented",
                    "note": "Bound enforced as F_SME >= 1.0 without arbitrary upper bound.",
                },
            ],
        }
        return record
