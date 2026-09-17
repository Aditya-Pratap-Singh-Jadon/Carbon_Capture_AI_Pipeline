# Gaussian + E-PINN Verification Layer

**Phase 14 Component**

## Architecture

The verification layer aggregates parallel evidence streams from two independent scientific engines without creating circular dependencies.

```
Observed Telemetry (7 variables)
       │
       ├──────────────► Multivariate Gaussian Model (Phase 13)
       │                    │
       │                    ▼
       │              Statistical Evidence (Mahalanobis D², Likelihood)
       │
       └──────────────► E-PINN (Phase 6)
                            │
                            ▼
                      Physics Evidence (15 Residuals, 23 Constraints)
                            │
                            ▼
                    VerificationEngine (Phase 2)
                            │
                            ▼
                      F_valid ∈ {0, 1}
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼
   EvidenceAggregationLayer        Carbon Credit Engine
```

## Evidence Object (`EvidenceAggregationResult`)

Instead of inventing an arbitrary unified score (e.g., $w_1 \times \text{Gaussian} + w_2 \times \text{E-PINN}$), the aggregation layer exposes a structured `EvidenceAggregationResult`:

1. **`statistical_evidence`**: From the Gaussian model (Mahalanobis distance, likelihood, anomaly flag).
2. **`physics_evidence`**: From the E-PINN and VerificationEngine (Residual consistency, constraint violations, `F_valid`, verified credits).
3. **`data_quality_evidence`**: Tracks dropped sensors, incomplete telemetry schemas.
4. **`overall_status`**: A non-quantitative categorical classification.

## Non-Circular Dependency Invariant

- The Gaussian model **does not consume `F_valid`**.
- The E-PINN **does not use the Gaussian log-likelihood** as a physics input.
- `F_valid` is entirely downstream of the E-PINN residuals and remains the sole authoritative multiplier for carbon credits.

## Graceful Degradation (Incomplete Input)

The Digital Twin currently provides only 7 physical variables, whereas the E-PINN strictly requires 26.
When processing Digital Twin telemetry:
- E-PINN evaluation is blocked (`physics_evidence` = null).
- Gaussian evaluation proceeds normally on its legitimate 6-variable boundary.
- `overall_status` degrades gracefully to `E_PINN_INPUT_INCOMPLETE`.

The system refuses to fabricate the 19 missing E-PINN variables just to force an integrated evaluation.

## Overall Status Enumeration

| Status | Meaning |
|---|---|
| `VERIFIED_NORMAL` | Statistically normal AND physically consistent. |
| `VERIFIED_STATISTICALLY_ANOMALOUS` | Statistically anomalous BUT physically consistent (`F_valid=1`). |
| `REJECTED_PHYSICALLY_INCONSISTENT` | Statistically normal BUT physically inconsistent (`F_valid=0`). |
| `REJECTED_ANOMALOUS_AND_INCONSISTENT` | Both statistically anomalous and physically inconsistent. |
| `INSUFFICIENT_DATA` | Telemetry dropped required fields. |
| `E_PINN_INPUT_INCOMPLETE` | Telemetry schema lacks the 26 E-PINN variables (e.g., Digital Twin). |

## Calibration Requirements

The Gaussian anomaly boundary (`mahalanobis_threshold=10.0`) is a diagnostic parameter. It is explicitly flagged as `threshold_requires_calibration=True`. It does not mathematically override the E-PINN `F_valid`.
