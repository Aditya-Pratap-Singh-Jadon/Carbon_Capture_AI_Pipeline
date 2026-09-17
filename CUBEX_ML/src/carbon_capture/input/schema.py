"""Input schema definition and validation rules (Frozen Reconciliation)."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from carbon_capture.input.canonical_order import (
    CORE_PHYSICAL_INPUT_ORDER,
    POLICY_PARAMETER_ORDER,
    CANONICAL_TABULAR_ORDER,
    BOUNDS,
)

@dataclass
class FieldDefinition:
    name: str
    dtype: str = "float32"
    required: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    unit: Optional[str] = None
    description: str = ""

@dataclass
class InputSchema:
    fields: Dict[str, FieldDefinition] = field(default_factory=dict)
    core_physical_order: List[str] = field(default_factory=lambda: list(CORE_PHYSICAL_INPUT_ORDER))
    policy_order: List[str] = field(default_factory=lambda: list(POLICY_PARAMETER_ORDER))
    canonical_order: List[str] = field(default_factory=lambda: list(CANONICAL_TABULAR_ORDER))

    def __post_init__(self):
        if not self.fields:
            for name in self.canonical_order:
                b_min, b_max = BOUNDS.get(name, (None, None))
                self.fields[name] = FieldDefinition(
                    name=name,
                    dtype="float32",
                    required=True,
                    min_value=b_min,
                    max_value=b_max,
                )

    def get_required_columns(self) -> List[str]:
        """Return strictly required fields."""
        return [k for k, v in self.fields.items() if v.required]

    def get_bounds(self, column: str) -> Tuple[Optional[float], Optional[float]]:
        """Return (min_val, max_val) for a column."""
        if column in self.fields:
            return self.fields[column].min_value, self.fields[column].max_value
        return None, None
