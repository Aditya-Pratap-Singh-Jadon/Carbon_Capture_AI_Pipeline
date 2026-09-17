# Audit Trail and Regulatory Traceability

## 1. Traceability Architecture
Every verification evaluation generates a complete cryptographically hashed audit package:
- **Run Header**: Unique Run ID, UTC timestamp, pipeline version, model version, and input file source.
- **Input Integrity Record**: Original column order, canonical column mapping, total record counts, missing/imputed values, hygiene flags.
- **E-PINN Physics Validation Record**: Mean and maximum normalized residuals, raw residual values per equation, characteristic normalization scales, and constraint violation states.
- **Deterministic Accounting Record**: Step-by-step breakdown of all benefit terms ($E_{\text{reduced}}, E_{\text{removed}}, E_{\text{displaced}}$), penalty deductions ($PE_{\text{aux}}, PE_{\text{lifecycle}}, L_{\text{leakage}}$), multiplier terms, and net $CC_T$.
- **Disclosures**: Explicit documentation of open ambiguities (e.g., $PE_{\text{aux}}$ dimensional inconsistency, $F_{\text{valid}}$ provenance, $F_{\text{SME}}$ one-sided bound).

## 2. Multi-Format Deliverables
For every verified batch, the system automatically exports:
1. `*_verification_report.json`: Machine-readable audit payload for blockchain or registry API ingestion.
2. `*_credit_records.csv`: Per-record intermediate quantities and credit allocations.
3. `*_verification_report.html`: Interactive human-readable verification dossier with embedded analytical plots.
4. `*_verification_report.pdf`: Publication-quality executive and regulatory PDF report.
