# Physics and Thermodynamic Foundations

## 1. Governing Conservation Equations (Section C)
1. **$E_{\text{actual}}$**: $(FC_{\text{actual}} \times NCV \times EF_{\text{CO2}} \times OF) + (EC_{\text{actual}} \times EF_{\text{grid}})$
2. **$E_{\text{baseline}}$**: $(FC_{\text{baseline}} \times NCV \times EF_{\text{CO2}} \times OF) + (EC_{\text{baseline}} \times EF_{\text{grid}})$
3. **$E_{\text{reduced}}$**: $\max(0, E_{\text{baseline}} - E_{\text{actual}})$
4. **$E_{\text{removed}}$**: $M_{\text{captured}} \times P_{\text{CO2}} \times \eta_{\text{purity}}$
5. **$E_{\text{displaced,chem}}$**: $M_{\text{byproduct}} \times EF_{\text{virgin\_displace}}$
6. **$E_{\text{displaced,heat}}$**: $\frac{H_{\text{recovered}}}{\eta_{\text{boiler}} \times NCV_{\text{fuel}}} \times EF_{\text{CO2,fuel}}$
7. **$E_{\text{displaced}}$**: $E_{\text{displaced,chem}} + E_{\text{displaced,heat}}$
8. **$EC_{\text{hardware}}$**: $(P_{\text{fan}} + P_{\text{pump}}) \times \frac{t_{\text{op}}}{3600}$
9. **$PE_{\text{aux}}$**: $EC_{\text{hardware}} - EF_{\text{grid}}$ *(see Section 3 on documented ambiguity)*
10. **$PE_{\text{lifecycle}}$**: $M_{\text{solvent\_makeup}} \times EF_{\text{solvent\_LCA}}$
11. **$L_{\text{slip}}$**: $M_{\text{captured}} \times \gamma_{\text{slip}}$
12. **$L_{\text{transport}}$**: $D_k \times EF_{\text{vehicle},k} \times M_{\text{trans},k}$
13. **$L_{\text{leakage}}$**: $L_{\text{slip}} + L_{\text{transport}}$
14. **$F_{\text{multiplier}}$**: $F_{\text{valid}} \times F_{\text{SME}} \times F_{\text{perm}} \times \alpha$
15. **$CC_T$**: $[E_{\text{reduced}} + E_{\text{removed}} + E_{\text{displaced}} - PE_{\text{aux}} - PE_{\text{lifecycle}} - L_{\text{leakage}}] \times F_{\text{multiplier}}$

## 2. Real Gas Thermodynamics & Hydraulics
- **Peng-Robinson Equation of State**: Computes real gas compressibility factor $Z$ and real gas density $\rho_{\text{CO2}}$ for scrubber gas mixtures.
- **Scrubber Hydraulics**: Models fan electrical load based on gas volumetric flow rate $Q_{\text{gas}}$ and bed pressure drop $\Delta P_{\text{scrub}}$, and pump power for liquid solvent recirculation.

## 3. Documented Ambiguities and Resolutions
- **$PE_{\text{aux}}$ Dimensional Inconsistency**:
  $EC_{\text{hardware}}$ has units of energy (kWh), while $EF_{\text{grid}}$ has units of emission rate ($t\text{CO}_2/\text{kWh}$). Subtracting an emission rate from an energy quantity is dimensionally inconsistent.
  *Resolution*: The equation is preserved verbatim by default per prompt specification contract, and an optional physical mode ($EC_{\text{hardware}} \times EF_{\text{grid}}$) is provided and configurable in `configs/physics.yaml`.
