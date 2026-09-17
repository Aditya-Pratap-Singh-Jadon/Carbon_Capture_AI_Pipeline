# Digital Twin Integration Guide

## Overview

The Digital Twin (located in `CUBEX_COM/`) simulates a carbon-capture scrubber and emits telemetry via stdout or a virtual COM port. The AI pipeline (located in `CUBEX/`) consumes this telemetry through a source-agnostic serial interface.

---

## Telemetry Contract

```
timestamp,CO2_ppm,Temperature_C,Humidity_percent,Gas_Flow_L_min,Captured_CO2_g,Fan_Speed_RPM
2026-09-16T16:30:01.250,421.3,29.4,61.2,487.5,0.18,1500
```

| Field | Unit | Semantic |
|---|---|---|
| `timestamp` | ISO-8601 | Record timestamp |
| `CO2_ppm` | ppm | Ambient/exhaust CO₂ concentration |
| `Temperature_C` | °C | Process temperature |
| `Humidity_percent` | % | Relative humidity |
| `Gas_Flow_L_min` | L/min | Process gas volumetric flow rate |
| `Captured_CO2_g` | g/timestep | CO₂ captured during current timestep (**NOT cumulative**) |
| `Fan_Speed_RPM` | RPM | Actual observed fan speed (**NOT the commanded value**) |

**Dropout sentinel:** `-1.0` means the sensor is unavailable. Parsed as `None`. Never imputed.

---

## Running the Digital Twin

```bash
cd CUBEX_COM/carbon_capture_digital_twin

# Stdout mode (for testing/inspection)
python scripts/run_twin.py --scenario normal --mode stdout

# Virtual COM mode (for AI pipeline integration)
python scripts/run_twin.py --scenario normal --mode com --port COM10
```

---

## Connecting the AI Pipeline

### Configuration

Edit `configs/input.yaml` (or pass arguments to `SerialSource`):

```yaml
input:
  source: serial
  port: COM11        # The READ side of the virtual COM pair
  baud_rate: 115200
  timeout: 1.0
```

**Do NOT hardcode port names.** Use the configuration file.

### Virtual COM Pair Setup (Windows)

Use `com0com` or a similar null-modem emulator:
- COM10 = Digital Twin writes here
- COM11 = AI pipeline reads here

### Python Usage

```python
from carbon_capture.input.source import SerialSource, DigitalTwinParser, TelemetryClassificationEngine

# Connect
src = SerialSource(port="COM11", baud_rate=115200, timeout=1.0)
parser = DigitalTwinParser()
clf_engine = TelemetryClassificationEngine()

with src:
    for raw_line in src.read_stream():
        result = parser.parse_line(raw_line)
        if result.ok:
            record = result.record
            classification = clf_engine.classify(record)
            # Use classification.variables to inspect availability
            # Proceed to RL controller if process state can be derived
```

---

## Q-Learning Fan Control

```python
from carbon_capture.rl.environment import ScrubberEnvironment
from carbon_capture.rl.q_learning import QLearningAgent
from carbon_capture.rl.serial_controller import SerialRLController

env = ScrubberEnvironment(min_rpm=1000.0, max_rpm=2000.0)
agent = QLearningAgent()
controller = SerialRLController(agent=agent, env=env, training=False)

# For each telemetry record:
cmd = controller.step(record)
if cmd is not None:
    print(f"Commanded RPM: {cmd.recommended_fan_rpm:.1f}")
    # Send cmd.recommended_fan_rpm to Digital Twin / Arduino
```

---

## Availability Status (26 E-PINN Variables)

From the current 7-field contract:

| Category | Count |
|---|---|
| DIRECTLY_OBSERVED | 0 |
| LEGITIMATELY_DERIVED | 2 (`M_captured`, `t_op`) — conditional |
| CONFIGURATION_OR_POLICY | 16 |
| UNAVAILABLE | 11 |

**E-PINN inference from Digital Twin telemetry is NOT currently possible.** See `docs/KNOWN_LIMITATIONS.md` (L-01).
