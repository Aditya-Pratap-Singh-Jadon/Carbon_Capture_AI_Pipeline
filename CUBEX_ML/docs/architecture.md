# Architecture Documentation: Industrial Carbon Capture AI & E-PINN Verification

## 1. System Overview
The pipeline integrates an **Extended Physics-Informed Neural Network (E-PINN)** and an authoritative **Deterministic Carbon-Credit Accounting Engine** to verify industrial carbon capture processes and issue auditable carbon credits.

```
INDUSTRIAL SENSOR TELEMETRY
            │
            ▼
┌───────────────────────────────────────┐
│ Input Layer: Schema & Canonical Order │
└───────────────────┬───────────────────┘
                    ▼
┌───────────────────────────────────────┐
│ Preprocessing & Data Hygiene          │
└───────────────────┬───────────────────┘
                    ▼
┌───────────────────────────────────────┐
│ E-PINN Scientific Physics Engine      │
│  - Multi-head physical reconstruction │
│  - 15 Governing Conservation Residuals│
│  - Augmented Lagrangian Constraints   │
└───────────────────┬───────────────────┘
                    ▼
┌───────────────────────────────────────┐
│ Physical Validation Decision          │
└───────────────────┬───────────────────┘
                    ▼
┌───────────────────────────────────────┐
│ Deterministic Carbon-Credit Engine    │
│  CC_T = [Benefits - Penalties] x Mult │
└───────────────────┬───────────────────┘
                    ▼
┌───────────────────────────────────────┐
│ Audit Trail & Report Generation       │
│  (JSON, CSV, HTML, PDF Deliverables)  │
└───────────────────────────────────────┘
```

## 2. Component Separation & Responsibilities
- **E-PINN Engine**: Acts as the scientific validation authority. Rather than predicting a black-box scalar, it reconstructs all intermediate physical quantities and verifies whether measured telemetry adheres to fundamental conservation laws.
- **Deterministic Calculator**: Evaluates the closed-form credit accounting formula without ML approximation.
- **Q-Learning Controller**: Operates as a separate loop optimizing fan speed (1000–2000 RPM) to minimize power and CO2 deviations.
- **Statistical Diagnostics**: Isolation Forest and One-Class SVM are available strictly as non-authoritative diagnostics.
