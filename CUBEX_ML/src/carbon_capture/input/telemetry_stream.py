"""Telemetry streaming ingestion layer.

Handles continuous parsing of the CUBEX_COM 7-field telemetry string.
(timestamp,CO2_ppm,Temperature_C,Humidity_percent,Gas_Flow_L_min,Captured_CO2_g,Fan_Speed_RPM)
Gracefully ignores a trailing 8th 'scenario' field if inadvertently present, ensuring
scenario is never used as an input feature for anomaly detection.
"""

import sys
import logging
from typing import Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)

class TelemetryParser:
    EXPECTED_FIELDS = [
        "timestamp",
        "CO2_ppm",
        "Temperature_C",
        "Humidity_percent",
        "Gas_Flow_L_min",
        "Captured_CO2_g",
        "Fan_Speed_RPM"
    ]
    
    def __init__(self):
        pass
        
    def parse_line(self, line: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str], Optional[str]]:
        """
        Parses a single telemetry line.
        Returns:
            is_valid (bool): True if parsing succeeded
            features (dict or None): Extracted continuous features
            timestamp (str or None): Extracted timestamp string
            error (str or None): Error description if parsing failed
        """
        line = line.strip()
        if not line:
            return False, None, None, None, "Empty line"
            
        parts = line.split(",")
        
        # We expect at least 7 fields. If 8 are present, the 8th is the scenario label (training data format).
        if len(parts) < 7:
            return False, None, None, None, f"Insufficient fields. Expected at least 7, got {len(parts)}."
            
        timestamp = parts[0]
        
        features = {}
        for i, field_name in enumerate(self.EXPECTED_FIELDS[1:]): # skip timestamp
            idx = i + 1
            raw_val = parts[idx]
            try:
                val = float(raw_val)
            except ValueError:
                return False, None, None, None, f"Non-numeric value for {field_name}: '{raw_val}'"
                
            # Handle CUBEX_COM missing sensor semantics
            if val == -1.0:
                features[field_name] = None
            else:
                features[field_name] = val
                
        # We extract parts[7] if it exists strictly for labeling during training.
        scenario_label = parts[7] if len(parts) > 7 else None
        
        return True, features, timestamp, scenario_label, None

class TelemetryStream:
    """An abstraction for reading telemetry from a stream (e.g. sys.stdin)."""
    def __init__(self, file_stream=None):
        self.stream = file_stream or sys.stdin
        self.parser = TelemetryParser()
        
    def __iter__(self):
        for line in self.stream:
            yield self.parser.parse_line(line)
