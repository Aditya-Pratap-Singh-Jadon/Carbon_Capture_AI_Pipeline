"""Universal Data Loader with Automatic Canonical Column Reordering (Frozen Schema)."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from carbon_capture.input.canonical_order import (
    CORE_PHYSICAL_INPUT_ORDER,
    POLICY_PARAMETER_ORDER,
    CANONICAL_TABULAR_ORDER,
    COLUMN_ALIASES,
)
from carbon_capture.input.schema import InputSchema
from carbon_capture.input.validator import InputValidator, ValidationResult
from carbon_capture.utils.exceptions import SchemaValidationError
from carbon_capture.utils.logging import get_logger

logger = get_logger(__name__)

class DataLoader:
    """Loads and canonicalizes tabular carbon capture input data."""

    def __init__(self, schema: Optional[InputSchema] = None):
        self.schema = schema or InputSchema()
        self.validator = InputValidator(self.schema)

    def canonicalize_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """Map raw columns via aliases, reorder into CANONICAL_TABULAR_ORDER, and preserve extra metadata."""
        original_cols = list(df.columns)
        rename_map = {}
        for col in df.columns:
            cleaned = col.strip().lower()
            if col in CANONICAL_TABULAR_ORDER:
                continue
            elif cleaned in COLUMN_ALIASES:
                rename_map[col] = COLUMN_ALIASES[cleaned]

        mapped_df = df.rename(columns=rename_map)

        # Check for missing core physical columns
        missing_core = [col for col in CORE_PHYSICAL_INPUT_ORDER if col not in mapped_df.columns]
        if missing_core:
            raise SchemaValidationError(f"Missing required core physical columns: {missing_core}")

        # If policy columns are missing in a raw operational sensor stream, supply standard defaults
        defaults = {"F_SME": 1.05, "F_perm": 0.98, "alpha": 0.90}
        for p_col in POLICY_PARAMETER_ORDER:
            if p_col not in mapped_df.columns:
                mapped_df[p_col] = defaults[p_col]
                logger.info(f"Policy column '{p_col}' not found in raw input; populated default: {defaults[p_col]}")

        extra_cols = [col for col in mapped_df.columns if col not in CANONICAL_TABULAR_ORDER]

        # Enforce canonical ordering
        ordered_cols = CANONICAL_TABULAR_ORDER + extra_cols
        reordered_df = mapped_df[ordered_cols].copy()

        logger.info(
            f"Successfully canonicalized columns. Original order: {original_cols[:5]}... "
            f"Canonical order restored with {len(CANONICAL_TABULAR_ORDER)} columns (26 physical + 3 policy)."
        )
        return reordered_df, original_cols, extra_cols

    def load_from_csv(self, file_path: Union[str, Path]) -> Tuple[pd.DataFrame, ValidationResult]:
        """Load from CSV file, reorder columns canonically, and validate."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input data file not found: {path}")

        raw_df = pd.read_csv(path)
        return self.load_from_dataframe(raw_df)

    def load_from_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationResult]:
        """Canonicalize and validate a pandas DataFrame."""
        reordered_df, orig_cols, _ = self.canonicalize_columns(df)
        val_result = self.validator.validate(reordered_df, original_cols=orig_cols)
        return reordered_df, val_result

    def load_from_records(self, records: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, ValidationResult]:
        """Load from a list of dict records."""
        df = pd.DataFrame.from_records(records)
        return self.load_from_dataframe(df)
