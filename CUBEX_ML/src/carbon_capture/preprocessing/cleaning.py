"""Data Hygiene & Cleaning Layer (strictly outside the E-PINN)."""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from carbon_capture.input.canonical_order import CANONICAL_INPUT_ORDER
from carbon_capture.utils.logging import get_logger

logger = get_logger(__name__)

class DataCleaner:
    """Performs hygiene checks, deduplication, and safe imputations."""

    def __init__(self, fill_strategy: str = "median"):
        self.fill_strategy = fill_strategy
        self.impute_values: dict = {}

    def fit(self, df: pd.DataFrame) -> "DataCleaner":
        """Compute training-set statistics for imputation."""
        for col in CANONICAL_INPUT_ORDER:
            if col in df.columns:
                if self.fill_strategy == "median":
                    self.impute_values[col] = float(df[col].median())
                else:
                    self.impute_values[col] = float(df[col].mean())
        return self

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Apply hygiene cleaning without modifying physical relationships."""
        cleaned_df = df.copy()
        corrections: List[str] = []

        # Deduplicate
        dup_count = cleaned_df.duplicated().sum()
        if dup_count > 0:
            cleaned_df = cleaned_df.drop_duplicates()
            corrections.append(f"Removed {dup_count} duplicate records.")

        # Impute missing values with fitted statistics if available
        for col in CANONICAL_INPUT_ORDER:
            if col in cleaned_df.columns and cleaned_df[col].isna().sum() > 0:
                n_missing = int(cleaned_df[col].isna().sum())
                val = self.impute_values.get(col, 0.0)
                cleaned_df[col] = cleaned_df[col].fillna(val)
                corrections.append(f"Imputed {n_missing} missing values in '{col}' with {val:.4f}")

        return cleaned_df, corrections
