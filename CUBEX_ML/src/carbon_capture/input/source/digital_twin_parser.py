"""DigitalTwinParser and TelemetryClassificationEngine.

Responsibilities
----------------
1. Parse a raw comma-separated telemetry line into DigitalTwinTelemetryRecord
   (strict 7-field contract; -1.0 → None; no imputation).
2. Build a TelemetryClassification that covers all 26 canonical E-PINN
   physical variables and 3 policy parameters, assigning each a precise
   AvailabilityCategory with no fabricated values.

Telemetry contract (order matters in the serial line):
    timestamp, CO2_ppm, Temperature_C, Humidity_percent,
    Gas_Flow_L_min, Captured_CO2_g, Fan_Speed_RPM

Units (contracted, not transmitted in the wire format):
    timestamp        ISO-8601
    CO2_ppm          ppm
    Temperature_C    °C
    Humidity_percent %
    Gas_Flow_L_min   L/min
    Captured_CO2_g   grams CO2 captured during current timestep (NOT cumulative)
    Fan_Speed_RPM    RPM (actual observed, not commanded)

Dropout sentinel: -1.0 → parsed as None inside DigitalTwinTelemetryRecord.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from carbon_capture.input.source.base import (
    AvailabilityCategory,
    DigitalTwinTelemetryRecord,
    TelemetryClassification,
    TelemetryVariableStatus,
)
from carbon_capture.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Wire-format contract
# ---------------------------------------------------------------------------

DIGITAL_TWIN_FIELDS: List[str] = [
    "timestamp",
    "CO2_ppm",
    "Temperature_C",
    "Humidity_percent",
    "Gas_Flow_L_min",
    "Captured_CO2_g",
    "Fan_Speed_RPM",
]

DROPOUT_SENTINEL = -1.0


# ---------------------------------------------------------------------------
# Parse result  (success / failure + diagnostics)
# ---------------------------------------------------------------------------

class ParseResult:
    """Thin wrapper returned by DigitalTwinParser.parse_line()."""

    def __init__(
        self,
        record: Optional[DigitalTwinTelemetryRecord],
        reason: Optional[str] = None,
    ) -> None:
        self.record = record
        self.ok = record is not None
        self.failure_reason = reason

    def __bool__(self) -> bool:
        return self.ok


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class DigitalTwinParser:
    """
    Parses raw serial telemetry lines from the Digital Twin (or any compatible
    source) into DigitalTwinTelemetryRecord.

    Rules
    -----
    - Exactly 7 comma-separated fields are required; any other count → reject.
    - Timestamp must be ISO-8601 parseable.
    - Each numeric field must be a valid float.
    - The value -1.0 is the dropout sentinel → stored as None, not 0.0.
    - Zero (0.0) is a legitimate measurement and is NOT treated as dropout.
    - Empty / whitespace-only lines → silently ignored (ParseResult with None).
    - No data is imputed, interpolated, or fabricated.
    """

    def parse_line(self, line: str) -> ParseResult:
        """Parse a single telemetry line.  Returns ParseResult."""

        # --- empty line ---
        stripped = line.strip()
        if not stripped:
            return ParseResult(None, "empty line")

        parts = [p.strip() for p in stripped.split(",")]

        # --- field count ---
        if len(parts) != len(DIGITAL_TWIN_FIELDS):
            reason = (
                f"field count mismatch: expected {len(DIGITAL_TWIN_FIELDS)}, "
                f"got {len(parts)}"
            )
            logger.warning(f"[INPUT] Schema invalid — {reason}: '{stripped}'")
            return ParseResult(None, reason)

        # --- timestamp ---
        try:
            ts_raw = parts[0]
            if not ts_raw:
                raise ValueError("empty timestamp field")
            timestamp = pd.to_datetime(ts_raw).to_pydatetime()
            if pd.isna(timestamp):
                raise ValueError("timestamp parsed as NaT")
        except Exception as exc:
            reason = f"invalid timestamp: '{parts[0]}' ({exc})"
            logger.warning(f"[INPUT] {reason}")
            return ParseResult(None, reason)

        # --- numeric sensor fields ---
        sensor_names = DIGITAL_TWIN_FIELDS[1:]   # CO2_ppm … Fan_Speed_RPM
        values: Dict[str, Optional[float]] = {}
        dropped_fields: List[str] = []
        numeric_valid = True

        for field_name, raw_str in zip(sensor_names, parts[1:]):
            try:
                val = float(raw_str)
                if val == DROPOUT_SENTINEL:
                    values[field_name] = None
                    dropped_fields.append(field_name)
                    logger.debug(f"[INPUT] Sensor dropout: {field_name}")
                else:
                    values[field_name] = val
            except ValueError:
                reason = f"invalid numeric value for {field_name}: '{raw_str}'"
                logger.warning(f"[INPUT] Numeric parse failure — {reason}")
                return ParseResult(None, reason)

        if dropped_fields:
            logger.info(f"[INPUT] Dropouts in record at {parts[0]}: {dropped_fields}")

        record = DigitalTwinTelemetryRecord(
            timestamp=timestamp,
            CO2_ppm=values["CO2_ppm"],
            Temperature_C=values["Temperature_C"],
            Humidity_percent=values["Humidity_percent"],
            Gas_Flow_L_min=values["Gas_Flow_L_min"],
            Captured_CO2_g=values["Captured_CO2_g"],
            Fan_Speed_RPM=values["Fan_Speed_RPM"],
            parse_success=True,
            schema_valid=True,
            timestamp_valid=True,
            numeric_valid=numeric_valid,
            dropped_fields=dropped_fields,
        )

        logger.debug(f"[INPUT] Record accepted: {timestamp}, dropouts={dropped_fields}")
        return ParseResult(record)

    def parse_batch(self, lines: List[str]) -> Tuple[List[DigitalTwinTelemetryRecord], List[ParseResult]]:
        """Parse multiple lines; return (accepted_records, all_results)."""
        results = [self.parse_line(ln) for ln in lines]
        accepted = [r.record for r in results if r.ok]
        return accepted, results


# ---------------------------------------------------------------------------
# Availability classification engine
# ---------------------------------------------------------------------------

class TelemetryClassificationEngine:
    """
    Builds a TelemetryClassification for a given DigitalTwinTelemetryRecord,
    covering all 26 physical E-PINN variables and 3 policy parameters.

    Classification rules (per variable) are derived from the frozen
    RECONCILED_SCHEMA.md contract.  No value is fabricated.

    Only DIRECTLY_OBSERVED means the value came from telemetry.
    Only LEGITIMATELY_DERIVED means the value was computed using accepted
      physics and observed telemetry alone.
    CONFIGURATION_OR_POLICY means the value comes from explicit system config
      (not measurement, not assumption, not default).
    UNAVAILABLE means the pipeline cannot obtain the value without inventing it.
    """

    def classify(self, record: DigitalTwinTelemetryRecord) -> TelemetryClassification:
        clf = TelemetryClassification(source_record=record)

        def _obs(var: str, telem_field: str, unit: str, value: Optional[float]) -> None:
            if value is None:
                clf.add(TelemetryVariableStatus(
                    variable_name=var,
                    category=AvailabilityCategory.UNAVAILABLE,
                    unit=unit,
                    source=telem_field,
                    value=None,
                    unavailability_reason="Sensor dropout (-1.0 sentinel received).",
                ))
            else:
                clf.add(TelemetryVariableStatus(
                    variable_name=var,
                    category=AvailabilityCategory.DIRECTLY_OBSERVED,
                    unit=unit,
                    source=telem_field,
                    value=value,
                ))

        def _unavail(var: str, unit: str, reason: str) -> None:
            clf.add(TelemetryVariableStatus(
                variable_name=var,
                category=AvailabilityCategory.UNAVAILABLE,
                unit=unit,
                source="none",
                value=None,
                unavailability_reason=reason,
            ))

        # ----------------------------------------------------------------
        # GROUP A — Directly observed from 7-field telemetry contract
        # ----------------------------------------------------------------
        # Note: CO2_ppm and Gas_Flow_L_min are related to but NOT identical
        #       to FC_actual, EC_actual, or M_captured. They are raw sensor
        #       readings, not the processed E-PINN physical variables.

        # There are no exact 1:1 matches to the 26 E-PINN variables because
        # the Digital Twin telemetry uses sensor-level units (ppm, L/min, g)
        # whereas the E-PINN requires process-level units (tonnes fuel, kWh,
        # tonnes CO2).  Mapping would require derivation or conversion factors
        # that are not part of the telemetry contract.

        # Fan_Speed_RPM is the closest to P_fan, but P_fan is power (kW),
        # not speed (RPM).  A fan curve is required to derive P_fan.

        # ----------------------------------------------------------------
        # GROUP B — UNAVAILABLE: all 26 physical E-PINN variables
        # ----------------------------------------------------------------
        _unavail("FC_actual",          "tonnes",         "Not in Digital Twin telemetry contract. Requires fuel flow meter (not Gas_Flow_L_min which is process gas, not fuel).")
        _unavail("FC_baseline",        "tonnes",         "Baseline fuel consumption is a process design parameter; not transmitted by Digital Twin.")
        _unavail("NCV",                "MJ/kg",          "Net calorific value of fuel is a fuel property constant; not transmitted by Digital Twin.")
        _unavail("EF_CO2",             "tCO2/t",         "Fuel CO2 emission factor is a fuel property constant; not transmitted by Digital Twin.")
        _unavail("OF",                 "dimensionless",  "Oxidation factor is a combustion process constant; not transmitted by Digital Twin.")
        _unavail("EC_actual",          "kWh",            "Electricity consumption requires energy meter; not transmitted by Digital Twin.")
        _unavail("EC_baseline",        "kWh",            "Baseline electricity consumption is a process design parameter; not transmitted.")
        _unavail("EF_grid",            "tCO2/kWh",       "Grid emission factor is a regional configuration constant; not transmitted.")
        _unavail("M_captured",         "tonnes",         "Captured_CO2_g is per-timestep grams and requires timestep duration and unit conversion to tonnes. Without the exact timestep definition this cannot be safely derived.")
        _unavail("P_CO2",              "dimensionless",  "Stoichiometric carbon mass fraction of product is a chemical property; not transmitted.")
        _unavail("eta_purity",         "dimensionless",  "Chemical purity fraction requires lab analysis; not transmitted.")
        _unavail("M_byproduct",        "tonnes",         "Displaced circular byproduct mass is a process chemistry output; not transmitted.")
        _unavail("EF_virgin_displace", "tCO2e/t",        "Virgin chemical avoidance factor is a LCA constant; not transmitted.")
        _unavail("H_recovered",        "MJ",             "Recovered heat requires calorimetric measurement; not transmitted.")
        _unavail("eta_boiler",         "dimensionless",  "Boiler efficiency is a process design parameter; not transmitted.")
        _unavail("NCV_fuel",           "MJ/kg",          "Net calorific value of displaced fuel is a fuel property; not transmitted.")
        _unavail("EF_CO2_fuel",        "tCO2/MJ",        "Emission factor of displaced fuel is a fuel property constant; not transmitted.")
        _unavail("P_fan",              "kW",             "Fan electrical power (kW) cannot be derived from Fan_Speed_RPM without a fan power curve — not part of the telemetry contract.")
        _unavail("P_pump",             "kW",             "Pump electrical power is not transmitted by Digital Twin.")
        _unavail("t_op",               "seconds",        "Hardware operating duration can be computed from consecutive timestamps but requires two valid adjacent records and explicit timestep semantic agreement.")
        _unavail("M_solvent_makeup",   "tonnes",         "Solvent replenishment mass is not transmitted by Digital Twin.")
        _unavail("EF_solvent_LCA",     "tCO2e/t",        "Solvent lifecycle carbon intensity is a LCA constant; not transmitted.")
        _unavail("gamma_slip",         "dimensionless",  "Carbon slippage fraction is a process characterisation parameter; not transmitted.")
        _unavail("D_k",                "km",             "Transit distance is a logistics parameter; not transmitted by Digital Twin.")
        _unavail("EF_vehicle_k",       "tCO2e/t-km",     "Freight vehicle emission factor is a logistics parameter; not transmitted.")
        _unavail("M_trans_k",          "tonnes",         "Transported cargo mass is a logistics quantity; not transmitted.")

        # ----------------------------------------------------------------
        # GROUP C — UNAVAILABLE: policy parameters
        # ----------------------------------------------------------------
        _unavail("F_SME",  "dimensionless", "Enterprise scale factor is a policy/registry constant; not transmitted.")
        _unavail("F_perm", "dimensionless", "Containment permanence factor is a project certification parameter; not transmitted.")
        _unavail("alpha",  "dimensionless", "Precision conservatism discount is a policy parameter; not transmitted.")

        # ----------------------------------------------------------------
        # Supplementary observed-field record (for completeness in report)
        # ----------------------------------------------------------------
        # These Digital Twin fields have NO direct counterpart in the 26 E-PINN
        # variables but are preserved in DigitalTwinTelemetryRecord.
        # We add them to the classification with a note.
        _obs("DT:CO2_ppm",          "CO2_ppm",         "ppm",    record.CO2_ppm)
        _obs("DT:Temperature_C",    "Temperature_C",   "°C",     record.Temperature_C)
        _obs("DT:Humidity_percent", "Humidity_percent","%",      record.Humidity_percent)
        _obs("DT:Gas_Flow_L_min",   "Gas_Flow_L_min",  "L/min",  record.Gas_Flow_L_min)
        _obs("DT:Captured_CO2_g",   "Captured_CO2_g",  "g/step", record.Captured_CO2_g)
        _obs("DT:Fan_Speed_RPM",    "Fan_Speed_RPM",   "RPM",    record.Fan_Speed_RPM)

        return clf
