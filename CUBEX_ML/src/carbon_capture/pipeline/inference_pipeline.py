"""E-PINN Inference and Verification Pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd
import numpy as np
import torch
from carbon_capture.input.loader import DataLoader
from carbon_capture.input.validator import ValidationResult
from carbon_capture.preprocessing.cleaning import DataCleaner
from carbon_capture.preprocessing.scaling import CanonicalScaler
from carbon_capture.epinn.model import EPINNModel
from carbon_capture.epinn.inference import EPINNInferenceEngine, EPINNInferenceOutput
from carbon_capture.carbon_credit.calculator import DeterministicCarbonCreditCalculator, CarbonCreditSummary
from carbon_capture.carbon_credit.verification import VerificationEngine, VerificationDecision
from carbon_capture.preprocessing.sequence_builder import SequenceBuilder
from carbon_capture.utils.serialization import load_artifact
from carbon_capture.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class PipelineInferenceResult:
    validation_result: ValidationResult
    epinn_output: EPINNInferenceOutput
    credit_summary: CarbonCreditSummary
    verification_decision: VerificationDecision
    dataframe_clean: pd.DataFrame

class InferencePipeline:
    """Production verification pipeline orchestrating data validation, physics checks, and credit accounting."""

    def __init__(
        self,
        model_path: Union[str, Path],
        scaler_path: Union[str, Path],
        pe_aux_mode: str = "verbatim",
        device: Optional[torch.device] = None,
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.pe_aux_mode = pe_aux_mode
        self.loader = DataLoader()
        self.cleaner = DataCleaner()
        
        # Determine temporal config from model_config.json if available
        self.temporal_window_size = 1
        model_cfg_path = Path(model_path).parent / "model_config.json"
        if model_cfg_path.exists():
            model_cfg = load_artifact(model_cfg_path, artifact_type="json")
            self.temporal_window_size = model_cfg.get("temporal_window_size", 1)

        if self.temporal_window_size > 1:
            self.sequence_builder = SequenceBuilder(window_size=self.temporal_window_size)
        else:
            self.sequence_builder = None

        # Load scaler
        scaler_state = load_artifact(scaler_path, artifact_type="json")
        self.scaler = CanonicalScaler.from_dict(scaler_state)

        # Inspect if model_config.json exists
        model_cfg_path = Path(model_path).parent / "model_config.json"
        if model_cfg_path.exists():
            model_cfg = load_artifact(model_cfg_path, artifact_type="json")
            hidden_dims = model_cfg.get("hidden_dims", [128, 128, 64])
            activation = model_cfg.get("activation", "silu")
            dropout = model_cfg.get("dropout", 0.05)
            pe_aux_mode = model_cfg.get("pe_aux_mode", pe_aux_mode)
        else:
            hidden_dims = [128, 128, 64]
            activation = "silu"
            dropout = 0.05

        # Load model
        self.model = EPINNModel(
            input_dim=len(self.scaler.feature_names),
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            characteristic_scales=self.scaler.get_characteristic_scales(),
            pe_aux_mode=pe_aux_mode,
        )
        state_dict = load_artifact(model_path, artifact_type="torch")
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        self.epinn_engine = EPINNInferenceEngine(self.model, self.scaler, device=self.device)
        self.credit_calculator = DeterministicCarbonCreditCalculator(pe_aux_mode=pe_aux_mode)
        self.verification_engine = VerificationEngine()

    def run(self, input_source: Union[str, Path, pd.DataFrame]) -> PipelineInferenceResult:
        """Run full verification on input CSV file or DataFrame."""
        if isinstance(input_source, (str, Path)):
            df, val_res = self.loader.load_from_csv(input_source)
        else:
            df, val_res = self.loader.load_from_dataframe(input_source)

        if not val_res.valid:
            logger.warning("Input failed schema validation.")
            # Produce placeholder outputs for failed validation
            empty_preds = {col: 0.0 for col in ["E_reduced", "CC_T"]}
            credit_sum = self.credit_calculator.calculate_from_dataframe(df.fillna(0.0))
            epinn_out = self.epinn_engine.evaluate(df.fillna(0.0))
            decision = self.verification_engine.decide(val_res, epinn_out, credit_sum)
            return PipelineInferenceResult(
                validation_result=val_res,
                epinn_output=epinn_out,
                credit_summary=credit_sum,
                verification_decision=decision,
                dataframe_clean=df,
            )

        # Clean
        df_clean, _ = self.cleaner.clean(df)

        # Build Sequences if Temporal Mode
        if self.temporal_window_size > 1 and self.sequence_builder:
            logger.info(f"Temporal mode active: building sequences (window_size={self.temporal_window_size})")
            X_seq, df_sorted = self.sequence_builder.build_sequences(df_clean)
            # The model accepts the scaled tensor directly in 3D
            # We scale the 2D matrix first to apply CanonicalScaler uniformly
            x_scaled = self.scaler.transform(df_sorted)
            
            # Reconstruct the 3D window manually from the scaled 2D array to ensure correct shape
            # (or we can just scale X_seq. X_seq is (batch, window, features))
            # Actually, let's just scale the 3D tensor manually:
            # (X_seq - mean) / scale is broadcastable if mean/scale are (features,)
            mean, scale = self.scaler.mean_, self.scaler.scale_
            x_seq_scaled = (X_seq - mean) / scale
            
            x_tensor = torch.tensor(x_seq_scaled, dtype=torch.float32, device=self.device)
            self.model.eval()
            with torch.no_grad():
                out_dict = self.model(x_tensor)
                
            # Convert out_dict back to CPU numpy arrays
            epinn_out_raw = {k: v.cpu().numpy() for k, v in out_dict.items()}
            
            # Repackage into EPINNInferenceOutput using the sorted DataFrame
            from carbon_capture.epinn.inference import EPINNInferenceOutput
            
            # For validation and residuals, it compares the LAST timestep of each window
            # against df_sorted. 
            # We compute residuals based on df_sorted (which corresponds to the last timestep)
            # using a temporary EPINNInferenceEngine instance to handle the generic logic
            df_clean = df_sorted
            
            # The standard EPINNInferenceEngine.evaluate takes df and does scaling/prediction.
            # We can mock its output since we already have the predictions.
            # Calculate physical residuals for the current step (df_clean) vs the model output
            temp_engine = self.epinn_engine
            # We must use temp_engine's residual calculation logic.
            # We'll just instantiate the dataclass manually with the required arrays.
            # Wait, epinn_engine.evaluate already handles physical vs predicted, but it calls model(x_tensor).
            # To preserve exact logic, we should probably patch EPINNInferenceEngine to support 3D directly.
            pass
        
        # E-PINN Inference & Residuals
        # (If temporal, self.epinn_engine must handle 3D sequence building internally or accept prebuilt tensor)
        # Let's adjust EPINNInferenceEngine to handle the temporal_window_size if passed, but
        # for backwards compatibility and to avoid massive refactoring, we handle 3D inference here:
        if self.temporal_window_size > 1:
            epinn_out = self._evaluate_temporal(df_clean, X_seq, df_sorted)
            df_clean = df_sorted
        else:
            epinn_out = self.epinn_engine.evaluate(df_clean)

        # Authoritative Deterministic Carbon Credit Calculation
        credit_sum = self.credit_calculator.calculate_from_dataframe(df_clean)

        # Verification Decision
        decision = self.verification_engine.decide(val_res, epinn_out, credit_sum)

        logger.info(
            f"Inference complete. Decision: {decision.decision} | "
            f"Verified Credits: {decision.verified_credits_tco2e:.2f} tCO2e"
        )
        return PipelineInferenceResult(
            validation_result=val_res,
            epinn_output=epinn_out,
            credit_summary=credit_sum,
            verification_decision=decision,
            dataframe_clean=df_clean,
        )
        
    def _evaluate_temporal(self, df: pd.DataFrame, X_seq: np.ndarray, df_sorted: pd.DataFrame):
        """Helper to evaluate temporal sequences and compute residuals correctly."""
        mean, scale = self.scaler.mean_, self.scaler.scale_
        x_seq_scaled = (X_seq - mean) / scale
        x_tensor = torch.tensor(x_seq_scaled, dtype=torch.float32, device=self.device)
        
        self.model.eval()
        with torch.no_grad():
            out_dict = self.model(x_tensor)
            
        preds = {k: v.cpu().numpy() for k, v in out_dict.items()}
        
        # Now we need to compute residuals. 
        # Since we have the model's predictions, we can use the main EPINNInferenceEngine's
        # evaluate_from_preds which correctly normalizes and aggregates them against the 
        # corresponding true DataFrame (df_sorted, representing the current/last step).
        return self.epinn_engine.evaluate_from_preds(preds, df_sorted)
