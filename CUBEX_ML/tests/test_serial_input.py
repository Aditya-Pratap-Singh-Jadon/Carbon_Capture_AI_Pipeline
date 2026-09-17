"""Tests for the Digital Twin serial telemetry ingestion layer.

Coverage
--------
Parser tests:
  - valid telemetry line
  - incorrect field count (too few / too many)
  - malformed timestamp
  - malformed numeric field
  - dropout sentinel -1.0 for each sensor field
  - multiple simultaneous dropouts
  - empty / whitespace-only line
  - header line (text in numeric position)

Serial tests:
  - mocked serial connection
  - timeout / no data
  - connection failure
  - continuous stream of valid records
  - clean shutdown / stop

Canonical representation tests:
  - all 7 fields observed → all classified DIRECTLY_OBSERVED
  - single dropout → UNAVAILABLE for that field, others unchanged
  - multiple dropouts
  - zero value is NOT treated as dropout

Availability report tests:
  - all 26 E-PINN physical variables present in classification
  - all 3 policy parameters present
  - all 26 physical + 3 policy → UNAVAILABLE (Digital Twin cannot supply them)
  - 6 DT:* supplementary fields classified as DIRECTLY_OBSERVED / UNAVAILABLE
  - no fabricated value for any UNAVAILABLE variable (value must be None)

Temporal / ordering tests:
  - timestamps parsed correctly
  - out-of-order detection helper
  - duplicate timestamp detection helper
  - Captured_CO2_g preserved per-timestep (not accumulated)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from carbon_capture.input.source.base import (
    AvailabilityCategory,
    DigitalTwinTelemetryRecord,
    TelemetryClassification,
)
from carbon_capture.input.source.digital_twin_parser import (
    DIGITAL_TWIN_FIELDS,
    DROPOUT_SENTINEL,
    DigitalTwinParser,
    TelemetryClassificationEngine,
)
from carbon_capture.input.source.serial_source import SerialSource
from carbon_capture.input.canonical_order import CORE_PHYSICAL_INPUT_ORDER, POLICY_PARAMETER_ORDER

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_LINE = "2026-09-16T16:30:01.250,421.3,29.4,61.2,487.5,0.18,1500"

@pytest.fixture
def parser() -> DigitalTwinParser:
    return DigitalTwinParser()


@pytest.fixture
def clf_engine() -> TelemetryClassificationEngine:
    return TelemetryClassificationEngine()


@pytest.fixture
def valid_record(parser) -> DigitalTwinTelemetryRecord:
    result = parser.parse_line(VALID_LINE)
    assert result.ok, f"Fixture setup failed: {result.failure_reason}"
    return result.record


# ---------------------------------------------------------------------------
# Parser: valid line
# ---------------------------------------------------------------------------

def test_valid_line_parses(parser):
    result = parser.parse_line(VALID_LINE)
    assert result.ok
    rec = result.record
    assert rec.CO2_ppm == pytest.approx(421.3)
    assert rec.Temperature_C == pytest.approx(29.4)
    assert rec.Humidity_percent == pytest.approx(61.2)
    assert rec.Gas_Flow_L_min == pytest.approx(487.5)
    assert rec.Captured_CO2_g == pytest.approx(0.18)
    assert rec.Fan_Speed_RPM == pytest.approx(1500.0)
    assert rec.parse_success
    assert rec.schema_valid
    assert rec.timestamp_valid
    assert rec.numeric_valid
    assert rec.dropped_fields == []


def test_timestamp_parsed_correctly(parser):
    result = parser.parse_line(VALID_LINE)
    assert result.ok
    ts: datetime = result.record.timestamp
    assert ts.year == 2026
    assert ts.month == 9
    assert ts.day == 16


# ---------------------------------------------------------------------------
# Parser: field count
# ---------------------------------------------------------------------------

def test_too_few_fields_rejected(parser):
    result = parser.parse_line("2026-09-16T16:30:01.250,421.3,29.4")
    assert not result.ok
    assert "field count" in result.failure_reason


def test_too_many_fields_rejected(parser):
    result = parser.parse_line(VALID_LINE + ",99.9,88.8")
    assert not result.ok
    assert "field count" in result.failure_reason


def test_exact_seven_fields_required(parser):
    for n_extra in [1, 2, 5]:
        line = VALID_LINE + "," + ",".join(["1.0"] * n_extra)
        assert not parser.parse_line(line).ok


# ---------------------------------------------------------------------------
# Parser: malformed timestamp
# ---------------------------------------------------------------------------

def test_malformed_timestamp_rejected(parser):
    line = "NOT_A_TIMESTAMP,421.3,29.4,61.2,487.5,0.18,1500"
    result = parser.parse_line(line)
    assert not result.ok
    assert "timestamp" in result.failure_reason


def test_missing_timestamp_field_rejected(parser):
    # Simulates a line where the first field is empty
    line = ",421.3,29.4,61.2,487.5,0.18,1500"
    result = parser.parse_line(line)
    assert not result.ok


# ---------------------------------------------------------------------------
# Parser: malformed numeric field
# ---------------------------------------------------------------------------

def test_malformed_numeric_co2_rejected(parser):
    line = "2026-09-16T16:30:01.250,NOT_A_NUMBER,29.4,61.2,487.5,0.18,1500"
    result = parser.parse_line(line)
    assert not result.ok
    assert "CO2_ppm" in result.failure_reason


def test_malformed_numeric_fan_speed_rejected(parser):
    line = "2026-09-16T16:30:01.250,421.3,29.4,61.2,487.5,0.18,FAST"
    result = parser.parse_line(line)
    assert not result.ok
    assert "Fan_Speed_RPM" in result.failure_reason


# ---------------------------------------------------------------------------
# Parser: dropout sentinel -1.0
# ---------------------------------------------------------------------------

def test_dropout_co2_becomes_none(parser):
    line = "2026-09-16T16:30:01.250,-1.0,29.4,61.2,487.5,0.18,1500"
    result = parser.parse_line(line)
    assert result.ok
    assert result.record.CO2_ppm is None
    assert "CO2_ppm" in result.record.dropped_fields


def test_dropout_temperature_becomes_none(parser):
    line = "2026-09-16T16:30:01.250,421.3,-1.0,61.2,487.5,0.18,1500"
    result = parser.parse_line(line)
    assert result.ok
    assert result.record.Temperature_C is None


def test_multiple_dropouts(parser):
    line = "2026-09-16T16:30:01.250,-1.0,-1.0,-1.0,487.5,0.18,1500"
    result = parser.parse_line(line)
    assert result.ok
    rec = result.record
    assert rec.CO2_ppm is None
    assert rec.Temperature_C is None
    assert rec.Humidity_percent is None
    assert rec.Gas_Flow_L_min == pytest.approx(487.5)
    assert len(rec.dropped_fields) == 3


def test_all_sensor_dropout(parser):
    line = "2026-09-16T16:30:01.250,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0"
    result = parser.parse_line(line)
    assert result.ok
    rec = result.record
    assert rec.CO2_ppm is None
    assert rec.Fan_Speed_RPM is None
    assert len(rec.dropped_fields) == 6  # all 6 sensor fields dropped


def test_zero_is_not_dropout(parser):
    """Zero (0.0) is a legitimate measurement; must NOT be treated as dropout."""
    line = "2026-09-16T16:30:01.250,0.0,0.0,0.0,0.0,0.0,0"
    result = parser.parse_line(line)
    assert result.ok
    rec = result.record
    assert rec.CO2_ppm == pytest.approx(0.0)
    assert rec.Fan_Speed_RPM == pytest.approx(0.0)
    assert rec.dropped_fields == []


def test_dropout_not_converted_to_zero(parser):
    """None from dropout must NEVER become 0.0."""
    line = "2026-09-16T16:30:01.250,-1.0,29.4,61.2,487.5,0.18,1500"
    result = parser.parse_line(line)
    assert result.ok
    assert result.record.CO2_ppm is not 0.0
    assert result.record.CO2_ppm is None


# ---------------------------------------------------------------------------
# Parser: empty / whitespace lines
# ---------------------------------------------------------------------------

def test_empty_line_returns_none(parser):
    result = parser.parse_line("")
    assert not result.ok
    assert result.record is None


def test_whitespace_line_returns_none(parser):
    result = parser.parse_line("   \t  ")
    assert not result.ok


# ---------------------------------------------------------------------------
# Parser: batch parsing
# ---------------------------------------------------------------------------

def test_batch_parse_filters_valid(parser):
    lines = [
        VALID_LINE,
        "",                                  # empty → rejected
        "bad,line",                          # wrong field count → rejected
        "2026-09-16T16:31:00.000,400.0,25.0,55.0,450.0,0.20,1450",  # valid
    ]
    accepted, results = parser.parse_batch(lines)
    assert len(accepted) == 2
    assert len(results) == 4  # all attempted


# ---------------------------------------------------------------------------
# Canonical representation / classification
# ---------------------------------------------------------------------------

def test_all_fields_observed_classification(valid_record, clf_engine):
    clf: TelemetryClassification = clf_engine.classify(valid_record)
    for field_name in DIGITAL_TWIN_FIELDS[1:]:  # skip timestamp
        dt_key = f"DT:{field_name}"
        assert dt_key in clf.variables
        assert clf.variables[dt_key].category == AvailabilityCategory.DIRECTLY_OBSERVED
        assert clf.variables[dt_key].value is not None


def test_single_dropout_classified_unavailable(parser, clf_engine):
    line = "2026-09-16T16:30:01.250,-1.0,29.4,61.2,487.5,0.18,1500"
    rec = parser.parse_line(line).record
    clf = clf_engine.classify(rec)
    assert clf.variables["DT:CO2_ppm"].category == AvailabilityCategory.UNAVAILABLE
    assert clf.variables["DT:CO2_ppm"].value is None
    # Others should still be observed
    assert clf.variables["DT:Temperature_C"].category == AvailabilityCategory.DIRECTLY_OBSERVED


def test_no_fabricated_value_for_unavailable(parser, clf_engine):
    """All UNAVAILABLE entries must have value=None — never a fabricated float."""
    line = "2026-09-16T16:30:01.250,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0"
    rec = parser.parse_line(line).record
    clf = clf_engine.classify(rec)
    for name, status in clf.variables.items():
        if status.category == AvailabilityCategory.UNAVAILABLE:
            assert status.value is None, (
                f"Variable '{name}' is UNAVAILABLE but has fabricated value: {status.value}"
            )


# ---------------------------------------------------------------------------
# 26-variable availability report
# ---------------------------------------------------------------------------

def test_all_26_physical_variables_present_in_classification(valid_record, clf_engine):
    clf = clf_engine.classify(valid_record)
    for var in CORE_PHYSICAL_INPUT_ORDER:
        assert var in clf.variables, f"E-PINN variable '{var}' missing from classification"


def test_all_3_policy_variables_present_in_classification(valid_record, clf_engine):
    clf = clf_engine.classify(valid_record)
    for var in POLICY_PARAMETER_ORDER:
        assert var in clf.variables, f"Policy variable '{var}' missing from classification"


def test_all_26_epinn_variables_unavailable(valid_record, clf_engine):
    """Digital Twin's 7 fields cannot directly supply any of the 26 E-PINN variables."""
    clf = clf_engine.classify(valid_record)
    for var in CORE_PHYSICAL_INPUT_ORDER:
        status = clf.variables[var]
        assert status.category == AvailabilityCategory.UNAVAILABLE, (
            f"Variable '{var}' should be UNAVAILABLE but is {status.category.name}"
        )
        assert status.value is None, (
            f"Variable '{var}' is UNAVAILABLE but has fabricated value: {status.value}"
        )


def test_all_3_policy_variables_unavailable(valid_record, clf_engine):
    clf = clf_engine.classify(valid_record)
    for var in POLICY_PARAMETER_ORDER:
        assert clf.variables[var].category == AvailabilityCategory.UNAVAILABLE


def test_dt_fields_directly_observed_in_classification(valid_record, clf_engine):
    clf = clf_engine.classify(valid_record)
    for field_name in DIGITAL_TWIN_FIELDS[1:]:
        dt_key = f"DT:{field_name}"
        assert clf.variables[dt_key].category == AvailabilityCategory.DIRECTLY_OBSERVED


def test_classification_unavailable_list_matches_epinn_variables(valid_record, clf_engine):
    clf = clf_engine.classify(valid_record)
    unavailable = clf.unavailable_names()
    # All 26 physical + 3 policy should be in unavailable list
    for var in CORE_PHYSICAL_INPUT_ORDER + POLICY_PARAMETER_ORDER:
        assert var in unavailable


# ---------------------------------------------------------------------------
# Temporal / ordering semantics
# ---------------------------------------------------------------------------

def test_captured_co2_g_per_timestep_not_cumulative(parser):
    """The parser must NOT accumulate Captured_CO2_g across records."""
    lines = [
        "2026-09-16T16:30:01.250,421.3,29.4,61.2,487.5,0.18,1500",
        "2026-09-16T16:30:02.250,422.0,29.5,61.3,488.0,0.20,1510",
    ]
    records, _ = parser.parse_batch(lines)
    assert records[0].Captured_CO2_g == pytest.approx(0.18)
    assert records[1].Captured_CO2_g == pytest.approx(0.20)
    # No accumulation: record[1] should NOT be 0.38
    assert records[1].Captured_CO2_g != pytest.approx(0.38)


def test_timestamps_are_parsed_as_datetime_objects(parser):
    records, _ = parser.parse_batch([VALID_LINE])
    assert isinstance(records[0].timestamp, datetime)


def test_out_of_order_timestamps_detected_externally(parser):
    """The parser does not enforce ordering; callers must detect it.
    This test documents that out-of-order timestamps are accessible from records."""
    lines = [
        "2026-09-16T16:30:05.000,421.3,29.4,61.2,487.5,0.18,1500",
        "2026-09-16T16:30:01.000,400.0,28.0,60.0,480.0,0.15,1480",  # earlier timestamp
    ]
    records, _ = parser.parse_batch(lines)
    # Caller can detect out-of-order by comparing consecutive timestamps
    assert records[0].timestamp > records[1].timestamp


def test_duplicate_timestamps_detectable(parser):
    ts = "2026-09-16T16:30:01.250"
    lines = [f"{ts},421.3,29.4,61.2,487.5,0.18,1500", f"{ts},422.0,29.5,61.3,488.0,0.20,1510"]
    records, _ = parser.parse_batch(lines)
    assert records[0].timestamp == records[1].timestamp


# ---------------------------------------------------------------------------
# Serial tests (mocked)
# ---------------------------------------------------------------------------

def test_serial_source_connect_calls_serial(monkeypatch):
    """connect() must call serial.Serial with correct args."""
    mock_serial = MagicMock()
    mock_serial.is_open = True
    with patch("carbon_capture.input.source.serial_source.serial.Serial", return_value=mock_serial) as mock_cls:
        src = SerialSource(port="COM99", baud_rate=9600, timeout=0.5)
        src.connect()
        mock_cls.assert_called_once_with(port="COM99", baudrate=9600, timeout=0.5)


def test_serial_source_disconnect_closes_port(monkeypatch):
    mock_serial = MagicMock()
    mock_serial.is_open = True
    with patch("carbon_capture.input.source.serial_source.serial.Serial", return_value=mock_serial):
        src = SerialSource(port="COM99")
        src.connect()
        src.disconnect()
        mock_serial.close.assert_called_once()


def test_serial_source_connection_failure_raises(monkeypatch):
    import serial as pyserial
    with patch("carbon_capture.input.source.serial_source.serial.Serial", side_effect=pyserial.SerialException("No port")):
        src = SerialSource(port="COMNONEXISTENT")
        with pytest.raises(pyserial.SerialException):
            src.connect()


def test_serial_source_stream_yields_decoded_lines(monkeypatch):
    """read_stream() must yield stripped UTF-8 lines from readline()."""
    mock_serial = MagicMock()
    mock_serial.is_open = True
    lines_encoded = [
        b"2026-09-16T16:30:01.250,421.3,29.4,61.2,487.5,0.18,1500\r\n",
        b"2026-09-16T16:30:02.250,422.0,29.5,61.3,488.0,0.20,1510\r\n",
        b"",  # Simulate timeout / no data → triggers end
    ]
    call_count = [0]
    def fake_readline():
        i = call_count[0]
        call_count[0] += 1
        if i < len(lines_encoded):
            return lines_encoded[i]
        # Stop the stream cleanly
        src._running = False
        return b""
    mock_serial.readline = fake_readline

    with patch("carbon_capture.input.source.serial_source.serial.Serial", return_value=mock_serial):
        src = SerialSource(port="COM99")
        src.connect()
        yielded = list(src.read_stream())

    assert len(yielded) == 2
    assert "421.3" in yielded[0]
    assert "422.0" in yielded[1]


def test_serial_source_skips_empty_lines(monkeypatch):
    mock_serial = MagicMock()
    mock_serial.is_open = True
    call_count = [0]
    lines_encoded = [
        b"\r\n",   # empty
        b"   \n",  # whitespace only
        b"2026-09-16T16:30:01.250,421.3,29.4,61.2,487.5,0.18,1500\r\n",
        b"",
    ]
    def fake_readline():
        i = call_count[0]; call_count[0] += 1
        if i < len(lines_encoded):
            return lines_encoded[i]
        src._running = False
        return b""
    mock_serial.readline = fake_readline
    with patch("carbon_capture.input.source.serial_source.serial.Serial", return_value=mock_serial):
        src = SerialSource(port="COM99")
        src.connect()
        yielded = list(src.read_stream())
    assert len(yielded) == 1  # only the valid data line


def test_serial_read_stream_raises_if_not_connected():
    src = SerialSource(port="COM99")
    with pytest.raises(RuntimeError):
        list(src.read_stream())


# ---------------------------------------------------------------------------
# End-to-end: serial stream → parser → classification
# ---------------------------------------------------------------------------

def test_e2e_stream_to_classification(monkeypatch):
    """Simulate a full Digital Twin stream: mock serial → parser → classification."""
    mock_serial = MagicMock()
    mock_serial.is_open = True
    raw_lines = [
        b"2026-09-16T16:30:01.250,421.3,29.4,61.2,487.5,0.18,1500\r\n",
        b"2026-09-16T16:30:02.250,-1.0,29.5,61.3,488.0,0.20,1510\r\n",  # CO2 dropout
        b"",
    ]
    call_count = [0]
    def fake_readline():
        i = call_count[0]; call_count[0] += 1
        if i < len(raw_lines):
            return raw_lines[i]
        src._running = False
        return b""
    mock_serial.readline = fake_readline

    with patch("carbon_capture.input.source.serial_source.serial.Serial", return_value=mock_serial):
        src = SerialSource(port="COM99")
        src.connect()
        raw_text_lines = list(src.read_stream())

    parser = DigitalTwinParser()
    clf_engine = TelemetryClassificationEngine()

    records = []
    for raw in raw_text_lines:
        result = parser.parse_line(raw)
        assert result.ok
        records.append(result.record)

    assert len(records) == 2

    # First record: all observed
    clf1 = clf_engine.classify(records[0])
    assert clf1.variables["DT:CO2_ppm"].category == AvailabilityCategory.DIRECTLY_OBSERVED

    # Second record: CO2 dropout
    clf2 = clf_engine.classify(records[1])
    assert clf2.variables["DT:CO2_ppm"].category == AvailabilityCategory.UNAVAILABLE
    assert clf2.variables["DT:CO2_ppm"].value is None

    # All 26 physical E-PINN variables remain UNAVAILABLE in both records
    for var in CORE_PHYSICAL_INPUT_ORDER:
        assert clf1.variables[var].category == AvailabilityCategory.UNAVAILABLE
        assert clf2.variables[var].category == AvailabilityCategory.UNAVAILABLE

    # True process state does NOT cross the COM boundary:
    # No key resembling Digital Twin internal state exists in the canonical classification
    for key in clf1.variables:
        assert "true_" not in key.lower()
        assert "internal_" not in key.lower()
        assert "simulated_" not in key.lower()
