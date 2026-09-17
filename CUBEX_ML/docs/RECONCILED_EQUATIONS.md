# Reconciled Governing Equations, Residuals & Constraints

**Document Version:** 1.0.0  
**Status:** Frozen Reconciled Mathematical Specification  
**Related Documents:** `docs/SPECIFICATION_RECONCILIATION.md`, `docs/RECONCILED_SCHEMA.md`

---

## 1. Equation Reconciliation Decisions

### Modification 1: $PE_{\text{aux}}$ Dual-Mode Formulation
- **Original specification:** $PE_{\text{aux}} = EC_{\text{hardware}} - EF_{\text{grid}}$.
- **Issue:** Dimensional inconsistency ($\text{kWh} - \text{tCO}_2/\text{kWh}$).
- **Decision:** Implement both `SPECIFICATION_MODE` (verbatim subtraction, flagged as dimensionally inconsistent) and `PHYSICAL_MODE` ($EC_{\text{hardware}} \times EF_{\text{grid}}$, dimensionally valid energy $\times$ emission intensity).
- **Reason:** Preserves strict mathematical contract traceability while enabling real-world physical correctness.
- **Implementation Consequence:** The mode is user-configurable in `configs/physics.yaml`, inspectable, and reported explicitly in audit logs.

### Modification 2: Smooth Hinge Relaxation for $E_{\text{reduced}}$
- **Original specification:** $E_{\text{reduced}} = \max(0, E_{\text{baseline}} - E_{\text{actual}})$.
- **Issue:** Non-differentiable kink at $E_{\text{baseline}} = E_{\text{actual}}$ causing gradient discontinuities during neural-network backpropagation.
- **Decision:** Use exact $\max(0, \cdot)$ in the deterministic engine and evaluation. During E-PINN training-time residual backprop, use the smooth softplus relaxation $\frac{1}{\beta} \ln(1 + e^{\beta(E_{\text{baseline}} - E_{\text{actual}})})$ with $\beta = 10.0$.
- **Reason:** Guarantees numerical stability and smooth gradient descent without altering the authoritative accounting formula.
- **Implementation Consequence:** Implemented in `EPINNLoss` without mutating closed-form reference calculations.

### Modification 3: Residual Hierarchy Taxonomy (14 Section D + 1 Section G)
- **Original specification:** Stated that $L_{\text{physics}}$ has 14 residuals, but 15 residuals are named.
- **Issue:** Potential ambiguity in whether $r_{\text{CC}}$ is part of $L_{\text{physics}}$.
- **Decision:** Explicitly partition:
  - 11 first-principles residuals $\to L_{\text{physics}}$
  - 3 compositional residuals $\to L_{\text{consistency}}$
  - 1 top-level credit consistency residual $\to L_{CC}$
- **Reason:** Prevents high-magnitude $CC_T$ errors from masking low-magnitude intermediate component drift (preserves identifiability, Section K).
- **Implementation Consequence:** All 15 residuals are tracked independently in diagnostics.

---

## 2. Authoritative Governing Equations

### Benefit Terms
1. **$E_{\text{actual}}$ (Actual Direct Emissions)**:
   $$E_{\text{actual}} = \sum_i \left( FC_{\text{actual},i} \times NCV_i \times EF_{\text{CO2},i} \times OF_i \right) + \left( EC_{\text{actual}} \times EF_{\text{grid}} \right)$$

2. **$E_{\text{baseline}}$ (Baseline Counterfactual Emissions)**:
   $$E_{\text{baseline}} = \sum_i \left( FC_{\text{baseline},i} \times NCV_i \times EF_{\text{CO2},i} \times OF_i \right) + \left( EC_{\text{baseline}} \times EF_{\text{grid}} \right)$$

3. **$E_{\text{reduced}}$ (Emissions Reduction Benefit)**:
   $$E_{\text{reduced}} = \max(0, E_{\text{baseline}} - E_{\text{actual}})$$

4. **$E_{\text{removed}}$ (Permanent Direct Chemical Removal)**:
   $$E_{\text{removed}} = M_{\text{captured}} \times P_{\text{CO2}} \times \eta_{\text{purity}}$$

5. **$E_{\text{displaced,chem}}$ (Virgin Chemical Avoidance Benefit)**:
   $$E_{\text{displaced,chem}} = M_{\text{byproduct}} \times EF_{\text{virgin\_displace}}$$

6. **$E_{\text{displaced,heat}}$ (Exothermic / Waste Heat Recovery Benefit)**:
   $$E_{\text{displaced,heat}} = \left( \frac{H_{\text{recovered}}}{\eta_{\text{boiler}} \times NCV_{\text{fuel}}} \right) \times EF_{\text{CO2,fuel}}$$

7. **$E_{\text{displaced}}$ (Combined Circular Displacement Benefit)**:
   $$E_{\text{displaced}} = E_{\text{displaced,chem}} + E_{\text{displaced,heat}}$$

### Penalty Terms
8. **$EC_{\text{hardware}}$ (Auxiliary Electrical Energy)**:
   $$EC_{\text{hardware}} = (P_{\text{fan}} + P_{\text{pump}}) \times \left( \frac{t_{\text{op}}}{3600} \right)$$

9. **$PE_{\text{aux}}$ (Auxiliary Operating Penalty)**:
   $$\begin{cases}
   EC_{\text{hardware}} - EF_{\text{grid}} & \text{in SPECIFICATION\_MODE (Verbatim)} \\
   EC_{\text{hardware}} \times EF_{\text{grid}} & \text{in PHYSICAL\_MODE (Dimensionally Consistent)}
   \end{cases}$$

10. **$PE_{\text{lifecycle}}$ (Sorbent Sourcing Lifecycle Penalty)**:
    $$PE_{\text{lifecycle}} = M_{\text{solvent\_makeup}} \times EF_{\text{solvent\_LCA}}$$

11. **$L_{\text{slip}}$ (Process Slippage Leakage)**:
    $$L_{\text{slip}} = M_{\text{captured}} \times \gamma_{\text{slip}}$$

12. **$L_{\text{transport}}$ (Logistics Transit Leakage)**:
    $$L_{\text{transport}} = \sum_k \left( D_k \times EF_{\text{vehicle},k} \times M_{\text{trans},k} \right)$$

13. **$L_{\text{leakage}}$ (Total Scope 3 Supply Chain Leakage)**:
    $$L_{\text{leakage}} = L_{\text{slip}} + L_{\text{transport}}$$

### Multipliers & Net Authoritative Credit
14. **$F_{\text{multiplier}}$ (Combined Risk & Policy Multiplier)**:
    $$F_{\text{multiplier}} = F_{\text{valid}} \times F_{\text{SME}} \times F_{\text{perm}} \times \alpha$$

15. **$CC_T$ (Authoritative Net Carbon Credits)**:
    $$CC_T = \left[ E_{\text{reduced}} + E_{\text{removed}} + E_{\text{displaced}} - PE_{\text{aux}} - PE_{\text{lifecycle}} - L_{\text{leakage}} \right] \times F_{\text{multiplier}}$$

---

## 3. The 15 Governing Equality Residuals

For every equation above, the model defines a residual $r = \hat{Y}_{\text{pred}} - Y_{\text{physics}}$:

### Group A: First-Principles Physics Residuals ($L_{\text{physics}}$)
1. $r_{\text{actual}} = \hat{E}_{\text{actual}} - \left[ (FC_{\text{actual}} \cdot NCV \cdot EF_{\text{CO2}} \cdot OF) + (EC_{\text{actual}} \cdot EF_{\text{grid}}) \right]$
2. $r_{\text{baseline}} = \hat{E}_{\text{baseline}} - \left[ (FC_{\text{baseline}} \cdot NCV \cdot EF_{\text{CO2}} \cdot OF) + (EC_{\text{baseline}} \cdot EF_{\text{grid}}) \right]$
3. $r_{\text{reduced}} = \hat{E}_{\text{reduced}} - \max(0, \hat{E}_{\text{baseline}} - \hat{E}_{\text{actual}})$
4. $r_{\text{removed}} = \hat{E}_{\text{removed}} - \left( M_{\text{captured}} \cdot P_{\text{CO2}} \cdot \eta_{\text{purity}} \right)$
5. $r_{\text{disp\_chem}} = \hat{E}_{\text{displaced,chem}} - \left( M_{\text{byproduct}} \cdot EF_{\text{virgin\_displace}} \right)$
6. $r_{\text{disp\_heat}} = \hat{E}_{\text{displaced,heat}} - \left( \frac{H_{\text{recovered}}}{\eta_{\text{boiler}} \cdot NCV_{\text{fuel}}} \cdot EF_{\text{CO2,fuel}} \right)$
7. $r_{\text{EChw}} = \hat{EC}_{\text{hardware}} - \left[ (P_{\text{fan}} + P_{\text{pump}}) \cdot \frac{t_{\text{op}}}{3600} \right]$
8. $r_{\text{aux}} = \hat{PE}_{\text{aux}} - \text{calc\_PE\_aux}(\hat{EC}_{\text{hardware}}, EF_{\text{grid}})$
9. $r_{\text{lifecycle}} = \hat{PE}_{\text{lifecycle}} - \left( M_{\text{solvent\_makeup}} \cdot EF_{\text{solvent\_LCA}} \right)$
10. $r_{\text{slip}} = \hat{L}_{\text{slip}} - \left( M_{\text{captured}} \cdot \gamma_{\text{slip}} \right)$
11. $r_{\text{transport}} = \hat{L}_{\text{transport}} - \left( D_k \cdot EF_{\text{vehicle},k} \cdot M_{\text{trans},k} \right)$

### Group B: Compositional Consistency Residuals ($L_{\text{consistency}}$)
12. $r_{\text{displaced}} = \hat{E}_{\text{displaced}} - (\hat{E}_{\text{displaced,chem}} + \hat{E}_{\text{displaced,heat}})$
13. $r_{\text{leakage}} = \hat{L}_{\text{leakage}} - (\hat{L}_{\text{slip}} + \hat{L}_{\text{transport}})$
14. $r_{\text{mult}} = \hat{F}_{\text{multiplier}} - (F_{\text{valid}} \cdot F_{\text{SME}} \cdot F_{\text{perm}} \cdot \alpha)$

### Group C: Top-Level Credit Accounting Residual ($L_{CC}$)
15. $r_{\text{CC}} = \hat{CC}_T - \left[ \hat{E}_{\text{red}} + \hat{E}_{\text{rem}} + \hat{E}_{\text{disp}} - \hat{PE}_{\text{aux}} - \hat{PE}_{\text{life}} - \hat{L}_{\text{leak}} \right] \times \hat{F}_{\text{multiplier}}$

---

## 4. Augmented Lagrangian Inequality Constraints

The system monitors 23 physical inequality constraints $g_c \le 0$:

| Constraint ID | Expression $g_c \le 0$ | Physical / Mathematical Basis |
|---|---|---|
| $c_1$ | $-\hat{E}_{\text{reduced}} \le 0$ | Reductions are definitionally non-negative |
| $c_2, c_3$ | $\eta_{\text{purity}} - 1 \le 0, \quad -\eta_{\text{purity}} \le 0$ | Purity is a fraction in $[0, 1]$ |
| $c_4, c_5$ | $\gamma_{\text{slip}} - 1 \le 0, \quad -\gamma_{\text{slip}} \le 0$ | Slippage is a fraction in $[0, 1]$ |
| $c_6, c_7$ | $\eta_{\text{boiler}} - 1 \le 0, \quad -\eta_{\text{boiler}} \le 0$ | Efficiency is a fraction in $[0, 1]$ |
| $c_8, c_9$ | $F_{\text{valid}} - 1 \le 0, \quad -F_{\text{valid}} \le 0$ | AI validation factor in $[0, 1]$ |
| $c_{10}, c_{11}$ | $F_{\text{perm}} - 1 \le 0, \quad -F_{\text{perm}} \le 0$ | Permanence factor in $[0, 1]$ |
| $c_{12}, c_{13}$ | $\alpha - 1 \le 0, \quad -\alpha \le 0$ | Conservatism discount in $[0, 1]$ |
| $c_{14}$ | $1.0 - F_{\text{SME}} \le 0$ | MSME incentive is one-sided $\ge 1.0$ |
| $c_{15} \dots c_{23}$| Non-negativity ($-\hat{M}_{\text{captured}} \le 0, -\hat{M}_{\text{byproduct}} \le 0$, etc.)| Masses, powers, durations, and heat cannot be negative |

Enforced via the Augmented Lagrangian loss:
$$L_{\text{constraints}} = \sum_{c=1}^{23} \left[ \mu_c \max(0, g_c)^2 + \lambda_c g_c \right]$$
with epoch-wise dual ascent updates $\lambda_c \leftarrow \max(0, \lambda_c + \mu_c \max(0, g_c))$.
