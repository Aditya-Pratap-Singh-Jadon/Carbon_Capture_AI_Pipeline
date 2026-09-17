# Extended Physics-Informed Neural Network (E-PINN)

## 1. Network Architecture
The E-PINN model consists of:
- **Shared Representation Backbone**: Multi-layer perceptron with smooth SiLU activations, Layer Normalization, and Dropout.
- **Physical Head**: Predicts all 15 intermediate physical quantities:
  - $E_{\text{actual}}$, $E_{\text{baseline}}$, $E_{\text{reduced}}$
  - $E_{\text{removed}}$, $E_{\text{displaced,chem}}$, $E_{\text{displaced,heat}}$, $E_{\text{displaced}}$
  - $EC_{\text{hardware}}$, $PE_{\text{aux}}$, $PE_{\text{lifecycle}}$
  - $L_{\text{slip}}$, $L_{\text{transport}}$, $L_{\text{leakage}}$
  - $F_{\text{multiplier}}$, $CC_T$

## 2. Loss Function Formulation
$$L_{\text{total}} = L_{\text{data}} + L_{\text{physics}} + L_{\text{consistency}} + w_{CC} \cdot L_{CC} + L_{\text{constraints}}$$

- **$L_{\text{data}}$**: Supervises observable quantities against audited reference data (using Gaussian NLL when uncertainty $\sigma_q$ is known, or normalized residuals).
- **$L_{\text{physics}}$**: Enforces first-principles conservation equations ($r_{\text{actual}}, r_{\text{baseline}}, r_{\text{reduced}}, r_{\text{removed}}, r_{\text{disp\_chem}}, r_{\text{disp\_heat}}, r_{\text{EChw}}, r_{\text{aux}}, r_{\text{lifecycle}}, r_{\text{slip}}, r_{\text{transport}}$).
- **$L_{\text{consistency}}$**: Enforces compositional equations ($r_{\text{displaced}}, r_{\text{leakage}}, r_{\text{mult}}$).
- **$L_{CC}$**: Anchors the top-level accounting equation.
- **$L_{\text{constraints}}$**: Enforces physical boundaries via the Augmented Lagrangian method.

## 3. Augmented Lagrangian Constraint Handling
Inequality constraints $g_c(\hat{Y}) \le 0$ are enforced via:
$$L_{\text{constraints}} = \sum_c \left[ \mu_c \max(0, g_c(\hat{Y}))^2 + \lambda_c g_c(\hat{Y}) \right]$$
- Dual ascent updates: $\lambda_c \leftarrow \max(0, \lambda_c + \mu_c \max(0, g_c))$.
- Adaptive penalty updates: $\mu_c$ is inflated when constraint violations fail to shrink by $\ge 10\%$ across epochs.
