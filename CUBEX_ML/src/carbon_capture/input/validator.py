"""Input validation and structured data hygiene reporting."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from carbon_capture.input.canonical_order import CANONICAL_INPUT_ORDER
from carbon_capture.input.schema import InputSchema
from carbon_capture.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class ValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    corrections: List[str] = field(default_factory=list)
    original_columns: List[str] = field(default_factory=list)
    canonical_columns: List[str] = field(default_factory=list)
    record_count: int = 0
    clean_dataframe: Optional[pd.DataFrame] = None
    rejected_rows: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "corrections": self.corrections,
            "original_columns": self.original_columns,
            "canonical_columns": self.canonical_columns,
            "record_count": self.record_count,
            "rejected_count": len(self.rejected_rows),
        }

class InputValidator:
    def __init__(self, schema: Optional[InputSchema] = None):
        self.schema = schema or InputSchema()

    def validate(self, df: pd.DataFrame, original_cols: Optional[List[str]] = None) -> ValidationResult:
        """Validate dataframe against the canonical schema and domain constraints."""
        errors: List[str] = []
        warnings: List[str] = []
        corrections: List[str] = []
        rejected_indices: List[int] = []
        
        orig_cols = original_cols if original_cols is not None else list(df.columns)
        
        # Check required columns
        missing_cols = [col for col in self.schema.get_required_columns() if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required canonical columns: {missing_cols}")
            return ValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
                corrections=corrections,
                original_columns=orig_cols,
                canonical_columns=list(df.columns),
                record_count=len(df),
                clean_dataframe=None
            )

        working_df = df[self.schema.canonical_order].copy()

        # Check duplicates
        duplicate_count = working_df.duplicated().sum()
        if duplicate_count > 0:
            warnings.append(f"Detected {duplicate_count} exact duplicate rows; deduplicating.")
            working_df = working_df.drop_duplicates()
            corrections.append("Deduplicated exact identical rows.")

        # Check NaN and Infinite values
        nan_counts = working_df.isna().sum()
        cols_with_nan = nan_counts[nan_counts > 0].to_dict()
        if cols_with_nan:
            errors.append(f"Contains NaN values in columns: {cols_with_nan}")

        inf_counts = np.isinf(working_df.select_dtypes(include=[np.number])).sum()
        cols_with_inf = inf_counts[inf_counts > 0].to_dict()
        if cols_with_inf:
            errors.append(f"Contains Infinite values in columns: {cols_with_inf}")

        # Check physical domain bounds
        for col in self.schema.canonical_order:
            b_min, b_max = self.schema.get_bounds(col)
            if b_min is not None:
                viol_min = (working_df[col] < b_min).sum()
                if viol_min > 0:
                    errors.append(f"Column '{col}' has {viol_min} records below physical minimum {b_min}")
            if b_max is not None:
                viol_max = (working_df[col] > b_max).sum()
                if viol_max > 0:
                    errors.append(f"Column '{col}' has {viol_max} records above physical maximum {b_max}")

        is_valid = len(errors) == 0
        if is_valid:
            logger.info(f"Input validation successful for {len(working_df)} records.")
        else:
            logger.warning(f"Input validation failed with {len(errors)} errors.")

        return ValidationResult(
            valid=is_valid,
            errors=errors,
            warnings=warnings,
            corrections=corrections,
            original_columns=orig_cols,
            canonical_columns=self.schema.canonical_order,
            record_count=len(working_df),
            clean_dataframe=working_df if is_valid else None,
            rejected_rows=rejected_indices
        )
