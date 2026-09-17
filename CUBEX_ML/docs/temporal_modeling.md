# Phase 8: Temporal and Iteration-Aware Modeling

## Objective
Extend the E-PINN verification system to process temporal sequences and dynamic industrial transitions without breaking backward compatibility for pointwise validation.

## Sequence Construction
The `SequenceBuilder` (`preprocessing/sequence_builder.py`) automatically sorts tabular data chronologically using an identified `timestamp` or `iteration` field. It constructs contiguous sliding windows of size $W$, preventing future-to-past temporal leakage. 

## Dynamic vs. Static Variables
All 26 core physics features are processed through the temporal encoder to form a comprehensive representation. The true dynamic drivers of the process are identified as:
- **Control / Hardware**: `P_fan`, `P_pump`
- **Thermodynamic Flow**: `FC_actual`, `EC_actual`, `H_recovered`
- **Chemical State**: `eta_purity`, `M_captured`, `M_byproduct`, `P_CO2`

Constants like `NCV`, `EF_CO2`, and administrative vectors (`F_SME`) remain static across the window. 

## Architecture
To natively support sequence processing while preserving the E-PINN physics engine, an LSTM-based `TemporalEncoder` was integrated into `EPINNMultiHead`.

```
Sequence (W, 26) → Temporal Encoder (LSTM) → Latent State (26)
                                                     ↓
                                             EPINNBackbone (MLP)
                                                     ↓
                                             15 Physics Heads
```
This enables the network to reconstruct the current time-step physics conditioned on historical process transients.

## Temporal Consistency Loss (Pending)
As per the Phase 8 requirements, the original formulation called for a temporal consistency loss to enforce dynamic physics transitions. However, because the existing frozen specification only defines static algebraic relationships (e.g. $M_{captured} \propto FC_{actual}$), arbitrary rate-of-change thresholds ($\|\frac{d\hat{y}}{dt}\|^2 < \tau_{max}$) were rejected. Imposing maximum derivative penalties is unscientific for processes like gas scrubbing where a sudden fan speed adjustment ($1200 \rightarrow 1800$ RPM) legitimately creates a rapid step-change in flow and capture.

**Therefore, the `l_temporal` component in `losses.py` is implemented as a modular structural hook but is explicitly disabled (`0.0`) by default.** A rigorous process-specific dynamic equation (e.g., differential equations for solvent mass accumulation or thermal inertia) is required before a temporal physics loss can be scientifically justified.

## Transient Legitimization vs Anomaly Detection
With temporal features integrated, the network learns the dependency of states over time.
- **Legitimate Transition**: Fan speed increases $\rightarrow$ Flow increases $\rightarrow$ Capture increases. The temporal encoder predicts this, and the E-PINN residuals remain low.
- **Sensor Bias / Anomaly**: Fan speed increases $\rightarrow$ Flow remains flat. The static residuals will fail, and the temporal sequence will conflict with the LSTM's learned dynamics, strongly triggering the verification rejection threshold.
