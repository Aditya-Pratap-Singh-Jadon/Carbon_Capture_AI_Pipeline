"""Temporal Sequence Construction and Leakage Prevention."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from carbon_capture.utils.logging import get_logger
from carbon_capture.input.canonical_order import CANONICAL_INPUT_ORDER, CORE_PHYSICAL_INPUT_ORDER

logger = get_logger(__name__)

# Dynamic variables expected to fluctuate during a single continuous industrial run.
# Static variables (emission factors, constants) will not be embedded in the temporal sequence diffs,
# but can just be passed statically or repeated. For the LSTM, we can pass all features,
# but it's important conceptually to know what drives dynamics.
DYNAMIC_VARIABLES = [
    "FC_actual",
    "EC_actual",
    "M_captured",
    "P_CO2",
    "eta_purity",
    "M_byproduct",
    "H_recovered",
    "P_fan",
    "P_pump",
    "gamma_slip",
]

class SequenceBuilder:
    """Builds temporal sequences for iteration-aware E-PINN processing.
    
    Ensures chronology and strictly prevents future-to-past information leakage.
    """
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        
    def build_sequences(
        self, 
        df: pd.DataFrame, 
        timestamp_col: Optional[str] = None
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Convert a chronological DataFrame into sliding windows of features.
        
        Args:
            df: Input DataFrame (must contain all canonical columns).
            timestamp_col: The column used to sort. If None, looks for 'timestamp', 
                           'iteration', 'time', or uses raw index if none exist.
                           
        Returns:
            X_seq: 3D numpy array (batch_size, window_size, num_features).
                   batch_size will be len(df). The first (window_size-1) elements 
                   will be padded by replicating the first row, so that X_seq[i] 
                   always ends exactly at df.iloc[i].
            df_sorted: The chronologically sorted dataframe, for downstream use.
        """
        # 1. Identify timestamp and sort
        df_sorted = df.copy()
        
        if timestamp_col and timestamp_col in df_sorted.columns:
            df_sorted = df_sorted.sort_values(by=timestamp_col).reset_index(drop=True)
        else:
            # Auto-detect
            for col in ["timestamp", "iteration", "time", "date"]:
                if col in df_sorted.columns:
                    df_sorted = df_sorted.sort_values(by=col).reset_index(drop=True)
                    logger.info(f"Auto-detected temporal column '{col}', sorted chronologically.")
                    break
            else:
                logger.info("No temporal column detected. Preserving existing row order as chronology.")
                df_sorted = df_sorted.reset_index(drop=True)
                
        # We need all canonical columns. If some are missing, downstream fails anyway, 
        # but let's just extract what's canonical that exists.
        available_cols = [c for c in CANONICAL_INPUT_ORDER if c in df_sorted.columns]
        
        data_matrix = df_sorted[available_cols].to_numpy(dtype=np.float32)
        n_samples, n_features = data_matrix.shape
        
        # 2. Build sliding windows with causal padding
        # To ensure output[i] corresponds to df_sorted.iloc[i], the window covers [i - window_size + 1 : i].
        # If i < window_size - 1, we pad by repeating the first row (steady state assumption for initial condition).
        
        X_seq = np.zeros((n_samples, self.window_size, n_features), dtype=np.float32)
        
        for i in range(n_samples):
            start_idx = i - self.window_size + 1
            if start_idx < 0:
                # Pad with the first observed state
                pad_size = abs(start_idx)
                pad_matrix = np.tile(data_matrix[0], (pad_size, 1))
                valid_matrix = data_matrix[0 : i + 1]
                window = np.vstack([pad_matrix, valid_matrix])
            else:
                window = data_matrix[start_idx : i + 1]
                
            X_seq[i] = window
            
        return X_seq, df_sorted
