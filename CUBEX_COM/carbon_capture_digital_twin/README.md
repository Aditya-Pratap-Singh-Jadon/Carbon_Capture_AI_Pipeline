# Digital Twin + Virtual COM Sensor Simulator

This project is a standalone Windows-based Digital Twin that simulates an industrial carbon-capture process, generates realistic sensor observations (including configurable noise, drift, and faults), and exposes these observations through a virtual COM interface.

## Purpose

The Digital Twin is designed to test downstream AI pipelines (such as E-PINN and Carbon Credit Verification Engines) by providing a realistic stream of sensor telemetry without requiring physical hardware (like an Arduino or real sensors).

The integration contract between this Digital Twin and the downstream pipeline is purely via a Serial (COM) connection transmitting CSV telemetry.

## Architecture

```text
                    ┌─────────────────────────┐
                    │   Digital Twin Process  │
                    │     True Process State  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      Sensor Model       │
                    │ noise / drift / faults  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │       Telemetry         │
                    │       CSV contract      │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
                  stdout        CSV       Virtual COM
                                             │
                                             ▼
                                  Main AI Pipeline
```

**IMPORTANT**: The *True Process State* is internal. It is logged for debugging but never transmitted to the downstream AI. The AI receives only the *Simulated Observations*.

## Telemetry Contract

The serial interface outputs one telemetry record per line, as comma-separated values (CSV) in UTF-8.

**Fields (fixed order):**
`timestamp,CO2_ppm,Temperature_C,Humidity_percent,Gas_Flow_L_min,Captured_CO2_g,Fan_Speed_RPM`

**Example:**
`2026-09-16T16:30:01.250,421.3,29.4,61.2,487.5,0.18,1500`

## Installation

1. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Virtual COM Setup (Windows)

To route the telemetry to your main AI pipeline on the same machine, you need a Virtual COM Port pair.

1. **Install com0com**: Download and install the open-source `com0com` null-modem emulator for Windows.
2. **Create a Pair**: Open the com0com setup and create a new port pair (e.g., `COM10` and `COM11`).
3. **Configure**: 
   - Set the Digital Twin to output to `COM10` (editable in `config.yaml` or via code).
   - Set the downstream AI pipeline to read from `COM11`.

**Note**: The simulator can be run in `stdout` or `csv` mode without any virtual COM software installed.

## Usage

### 1. Normal Mode (stdout)
Print telemetry directly to the console:
```bash
python scripts/run_twin.py --scenario normal --mode stdout
```

### 2. Fault Scenarios
Run a specific fault scenario to test your downstream AI's anomaly detection:
```bash
python scripts/run_twin.py --scenario sensor_drift --mode stdout
python scripts/run_twin.py --scenario fan_failure --mode stdout
python scripts/run_twin.py --scenario low_flow --mode stdout
python scripts/run_twin.py --scenario sensor_spike --mode stdout
```

### 3. Generate a CSV Dataset
Log the telemetry to a timestamped CSV file in the `logs/` folder:
```bash
python scripts/run_twin.py --scenario normal --mode csv
```

### 4. Serial Mode (Virtual COM)
Output the telemetry to the COM port specified in `config.yaml` (`COM10` by default):
```bash
python scripts/run_twin.py --scenario normal --mode serial
```

### 5. Listing Available Ports
Check which COM ports are currently visible to the system:
```bash
python scripts/list_ports.py
```

### 6. Testing the Serial Connection
Run the Digital Twin in serial mode on `COM10`. Then, in a separate terminal, run the mock receiver on the other end of the pair (`COM11`):
```bash
python scripts/test_serial.py --port COM11
```

## Future Hardware Integration

The separation of concerns means that the downstream AI pipeline only cares about the serial telemetry contract.

When physical hardware is ready:
1. Connect the real Arduino with physical sensors to the PC.
2. Note the physical COM port assigned by Windows (e.g., `COM4`).
3. Point your Main AI Pipeline to `COM4` instead of `COM11`.
4. Stop running the Digital Twin.

No changes to the AI pipeline code (E-PINN, Verification Engine) are required.

## Limitations
*This Digital Twin generates simulated development/test data. It is not a substitute for calibrated industrial instrumentation or real-world physical measurements. The process model uses simplified heuristics to represent flow and carbon capture.*
