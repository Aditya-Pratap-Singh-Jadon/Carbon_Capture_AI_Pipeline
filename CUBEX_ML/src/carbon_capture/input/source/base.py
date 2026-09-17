"""Base abstractions for source-agnostic telemetry ingestion.

Separation of concerns:
- TelemetrySource: abstract source (serial, file, mock)
- DigitalTwinTelemetryRecord: exactly the 7 contracted observation fields
- TelemetryVariableStatus: one classified entry per required downstream variable
- TelemetryClassification: availability report across all required variables
- AvailabilityCategory: four canonical semantic categories (DIRECTLY_OBSERVED /
  LEGITIMATELY_DERIVED / CONFIGURATION_OR_POLICY / UNAVAILABLE)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, Iterator, List, Optional


# ---------------------------------------------------------------------------
# Semantic Category
# ---------------------------------------------------------------------------

class AvailabilityCategory(Enum):
    """Four mutually exclusive categories for every downstream-required variable."""
    DIRECTLY_OBSERVED = auto()        # Present in the telemetry record
    LEGITIMATELY_DERIVED = auto()     # Computable from observed telemetry + established physics/timestamps
    CONFIGURATION_OR_POLICY = auto()  # Supplied from explicit system configuration, not fabricated
    UNAVAILABLE = auto()              # Cannot be obtained without fabricating information


# ---------------------------------------------------------------------------
# Raw 7-field Telemetry Observation
# ---------------------------------------------------------------------------

@dataclass
class DigitalTwinTelemetryRecord:
    """
    Strict 7-field observation contract between the Digital Twin (or any
    compatible telemetry source) and the AI pipeline input layer.

    All Optional[float] fields are None when the telemetry source emitted the
    dropout sentinel (-1.0).  Zero is a legitimate measurement.

    This class is intentionally NOT the 26-variable E-PINN representation.
    """
    timestamp: datetime

    # Sensor fields; None = dropout / unavailable (sentinel was -1.0)
    CO2_ppm: Optional[float]            # ppm
    Temperature_C: Optional[float]      # °C
    Humidity_percent: Optional[float]   # %
    Gas_Flow_L_min: Optional[float]     # L/min
    Captured_CO2_g: Optional[float]     # grams CO2 captured during current timestep (NOT cumulative)
    Fan_Speed_RPM: Optional[float]      # RPM (actual observed, not commanded)

    # Ingestion / data-quality diagnostics (not scientific validity)
    parse_success: bool = True
    schema_valid: bool = True
    timestamp_valid: bool = True
    numeric_valid: bool = True
    dropped_fields: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-variable availability entry
# ---------------------------------------------------------------------------

@dataclass
class TelemetryVariableStatus:
    """Availability classification for a single downstream-required variable."""
    variable_name: str
    category: AvailabilityCategory
    unit: str
    source: str                              # telemetry field name, derivation rule, or policy key
    value: Optional[float] = None            # None if UNAVAILABLE; not a placeholder
    derivation_rule: Optional[str] = None    # human-readable formula if LEGITIMATELY_DERIVED
    unavailability_reason: Optional[str] = None  # human-readable if UNAVAILABLE


# ---------------------------------------------------------------------------
# Full 26-variable availability report
# ---------------------------------------------------------------------------

@dataclass
class TelemetryClassification:
    """
    Explicit availability report across all required downstream variables.

    Generated from a DigitalTwinTelemetryRecord.  Contains NO fabricated
    values: any variable that cannot be observed, derived, or supplied from
    configuration is classified UNAVAILABLE with value=None.
    """
    source_record: DigitalTwinTelemetryRecord
    variables: Dict[str, TelemetryVariableStatus] = field(default_factory=dict)

    def add(self, status: TelemetryVariableStatus) -> None:
        self.variables[status.variable_name] = status

    def all_available(self) -> bool:
        return all(
            s.category != AvailabilityCategory.UNAVAILABLE
            for s in self.variables.values()
        )

    def unavailable_names(self) -> List[str]:
        return [
            name for name, s in self.variables.items()
            if s.category == AvailabilityCategory.UNAVAILABLE
        ]

    def directly_observed_names(self) -> List[str]:
        return [
            name for name, s in self.variables.items()
            if s.category == AvailabilityCategory.DIRECTLY_OBSERVED
        ]

    def summary_table(self) -> str:
        """Human-readable availability table for logging/reporting."""
        lines = [
            f"{'Variable':<22} {'Category':<30} {'Source':<25} {'Unit':<15} {'Value'}",
            "-" * 110,
        ]
        for var_name, status in self.variables.items():
            val = f"{status.value:.6g}" if status.value is not None else "None"
            lines.append(
                f"{var_name:<22} {status.category.name:<30} "
                f"{status.source:<25} {status.unit:<15} {val}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Abstract telemetry source
# ---------------------------------------------------------------------------

class TelemetrySource(ABC):
    """
    Source-agnostic abstract base for telemetry ingestion.

    Any producer (Digital Twin serial, Arduino serial, CSV replay) must
    satisfy this interface.  The downstream pipeline must not contain
    source-specific branching logic.
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the physical or virtual source."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release the connection gracefully."""

    @abstractmethod
    def read_stream(self) -> Iterator[str]:
        """Yield raw telemetry strings, one per record, until stopped."""
