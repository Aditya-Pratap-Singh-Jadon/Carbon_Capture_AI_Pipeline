# Final System Architecture

**CUBEX — Industrial Carbon-Capture Verification & Carbon-Credit AI Pipeline**  
**Version:** 1.0.0 | **Status:** Production-Ready (Phase 12)

---

## Three Execution Paths

```
PATH A — Batch Verification (CSV / DataFrame)
══════════════════════════════════════════════
CSV File / DataFrame
    ↓
DataLoader (canonical column ordering, alias resolution)
    ↓
InputValidator (26-variable schema, bound checks, NaN detection)
    ↓
CanonicalScaler (StandardScaler fitted on training data only)
    ↓
EPINNMultiHead (26-dimensional input → 15 physical outputs)
    ↓
EPINNInferenceEngine.evaluate_from_preds()
    ↓
compute_raw_residuals() → 15 physics residuals (Group A/B/C)
    ↓
normalize_residuals() → dimensionless residual vector
    ↓
AugmentedLagrangianConstraints (23 physical inequality constraints)
    ↓
VerificationEngine → F_valid ∈ {0, 1} (deterministic)
    ↓
DeterministicCarbonCreditCalculator → CC_T (tCO₂e)
    ↓
VerificationDecisionEngine → VERIFIED / REJECTED / CONDITIONAL
    ↓
ReportGenerator → JSON, CSV, HTML, PDF audit reports
    ↓
UncertaintyIntegrator → MC-Dropout diagnostic (DOES NOT MODIFY F_valid)
    ↓
GradientExplainer → Feature attribution (DOES NOT MODIFY PREDICTIONS)


PATH B — Real-Time Digital Twin / Serial Telemetry
═══════════════════════════════════════════════════
Digital Twin (CUBEX_COM project)
    ↓ Virtual COM (COM10 → COM11, or any pair)
SerialSource (pyserial; configurable port/baud/timeout)
    ↓
DigitalTwinParser → DigitalTwinTelemetryRecord (7 fields; -1.0 → None)
    ↓
TelemetryClassificationEngine → TelemetryClassification
    ↓                           (26-var availability report)
[STOP — E-PINN inference requires additional sensors; see limitations]


PATH C — Q-Learning Fan Control Loop
══════════════════════════════════════
DigitalTwinTelemetryRecord (from PATH B)
    ↓
telemetry_to_process_state() [unit conversion only; no fabrication]
    ↓
QLearningAgent.select_action() → action ∈ {-100, 0, +100} RPM
    ↓
FanCommand (recommended_fan_rpm; bounded to [1000, 2000] RPM)
    ↓
Digital Twin / future Arduino [EXTERNAL — receives command via COM]
    ↓
New telemetry observation [PATH B repeats]
```

---

## Component Inventory

| Module | Path | Role |
|---|---|---|
| `DataLoader` | `input/loader.py` | CSV ingestion, canonical ordering |
| `InputValidator` | `input/validator.py` | Schema/bounds validation |
| `CanonicalScaler` | `preprocessing/scaling.py` | Feature scaling |
| `EPINNMultiHead` | `epinn/architecture.py` | Physics-constrained neural network |
| `TemporalEncoder` | `epinn/architecture.py` | LSTM sequence encoder (Phase 8) |
| `SequenceBuilder` | `epinn/architecture.py` | Sliding window builder |
| `EPINNLoss` | `epinn/losses.py` | 15 residual losses + Augmented Lagrangian |
| `EPINNInferenceEngine` | `epinn/inference.py` | Residual evaluation + constraint checking |
| `VerificationEngine` | `validation/` | F_valid computation |
| `DeterministicCCCalculator` | `carbon_credit/calculator.py` | CC_T calculation |
| `ReportGenerator` | `reporting/report_generator.py` | JSON/CSV/HTML/PDF |
| `UncertaintyEstimator` | `epinn/uncertainty.py` | MC-Dropout |
| `UncertaintyIntegrator` | `epinn/uncertainty_integration.py` | Diagnostic wrapper |
| `GradientExplainer` | `epinn/explainability.py` | Gradient feature attribution |
| `BaselineMLP` | `epinn/baseline.py` | Non-physics comparison baseline |
| `SerialSource` | `input/source/serial_source.py` | pyserial ingestion |
| `DigitalTwinParser` | `input/source/digital_twin_parser.py` | 7-field parser |
| `TelemetryClassificationEngine` | `input/source/digital_twin_parser.py` | 26-var availability |
| `QLearningAgent` | `rl/q_learning.py` | Tabular Q-learning |
| `SerialRLController` | `rl/serial_controller.py` | Source-agnostic fan controller |
| `ScrubberEnvironment` | `rl/environment.py` | RL training environment |

---

## Scientific Boundaries

| Boundary | Rule |
|---|---|
| F_valid | Computed by VerificationEngine; not modifiable by uncertainty or Q-learning |
| Carbon Credits | Downstream of F_valid; deterministic formula only |
| Q-learning | Controls fan speed only; no access to F_valid or CC engine |
| Uncertainty | Diagnostic flag only; never overrides verification |
| Explainability | Observational; never modifies predictions or residuals |
| Digital Twin state | Private; only 7 contracted fields cross the COM boundary |
