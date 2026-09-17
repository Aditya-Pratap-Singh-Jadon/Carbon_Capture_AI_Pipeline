# Q-Learning Process-Control Optimization

## 1. Objective and Boundary
The Q-learning agent is strictly responsible for real-time process optimization (e.g. fan speed adjustments in the capture scrubber). It operates in a separate loop from scientific validation:
- **Q-Learning Question**: "What control action should be taken to balance power consumption and capture performance?"
- **E-PINN Question**: "Can I trust this process measurement physically according to governing conservation laws?"

## 2. State, Action, and Operational Bounds
- **State Vector**:
  - $\text{CO}_2$ concentration (%)
  - Temperature (°C)
  - Fan speed (RPM)
  - Gas volumetric flow rate ($\text{m}^3/\text{s}$)
- **Action Space**:
  - `0`: $-100$ RPM
  - `1`: $0$ RPM
  - `2`: $+100$ RPM
- **Hardware Bounds**:
  - $\text{Fan Speed} \in [1000, 2000]$ RPM
- **Reward Function**:
  $$R = -|\text{CO}_{2,\text{actual}} - \text{CO}_{2,\text{target}}| - 0.5 \left( \frac{\text{RPM}}{\text{RPM}_{\max}} \right)^2$$
