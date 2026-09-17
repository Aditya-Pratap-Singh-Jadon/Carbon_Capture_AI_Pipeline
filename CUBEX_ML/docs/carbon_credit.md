# Deterministic Carbon-Credit Accounting Engine

## 1. Authoritative Accounting Engine
Under this architecture, **the neural network never replaces deterministic carbon accounting**. The neural network's purpose is exclusively scientific physical verification. The deterministic engine calculates authoritative credit volumes using closed-form algebra:

$$CC_T = [E_{\text{reduced}} + E_{\text{removed}} + E_{\text{displaced}} - PE_{\text{aux}} - PE_{\text{lifecycle}} - L_{\text{leakage}}] \times F_{\text{multiplier}}$$

where:
$$F_{\text{multiplier}} = F_{\text{valid}} \times F_{\text{SME}} \times F_{\text{perm}} \times \alpha$$

## 2. Term Classifications
- **Benefit Terms (+)**:
  - $E_{\text{reduced}}$: Scope 1 direct combustion reductions.
  - $E_{\text{removed}}$: Stoichiometrically bound carbon mineralization.
  - $E_{\text{displaced}}$: Displaced virgin chemicals and reclaimed thermal energy.
- **Penalty Terms (-)**:
  - $PE_{\text{aux}}$: Hardware power operating penalty.
  - $PE_{\text{lifecycle}}$: Sorbent replenishment cradle-to-gate emissions.
  - $L_{\text{leakage}}$: Carbon slip and transit emissions.
- **Multipliers**:
  - $F_{\text{valid}} \in [0, 1]$: AI-driven data and physics validity multiplier.
  - $F_{\text{SME}} \ge 1.0$: Enterprise scale incentive.
  - $F_{\text{perm}} \in [0, 1]$: Containment permanence over a 100-year horizon.
  - $\alpha \in [0, 1]$: Prescribed safety discount factor.
