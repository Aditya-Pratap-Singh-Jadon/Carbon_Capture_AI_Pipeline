"""End-to-End Execution Pipeline connecting input ingestion to report publishing."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd
from carbon_capture.pipeline.inference_pipeline import InferencePipeline, PipelineInferenceResult
from carbon_capture.reporting.report_generator import ReportGenerator
from carbon_capture.utils.logging import get_logger

logger = get_logger(__name__)

class EndToEndPipeline:
    """Executes full carbon capture verification lifecycle: raw input to published reports."""

    def __init__(
        self,
        model_path: Union[str, Path],
        scaler_path: Union[str, Path],
        reports_dir: Union[str, Path] = "reports/generated",
        pe_aux_mode: str = "verbatim",
    ):
        self.inference_pipeline = InferencePipeline(
            model_path=model_path,
            scaler_path=scaler_path,
            pe_aux_mode=pe_aux_mode,
        )
        self.report_generator = ReportGenerator(output_dir=Path(reports_dir))
        self.pe_aux_mode = pe_aux_mode

    def verify_and_report(
        self,
        input_source: Union[str, Path, pd.DataFrame],
        run_name: str = "industrial_batch",
    ) -> Dict[str, Any]:
        """Execute complete verification and generate all deliverables."""
        logger.info(f"Executing end-to-end verification for run: {run_name}")

        # 1. Run inference & verification
        res: PipelineInferenceResult = self.inference_pipeline.run(input_source)

        # 2. Generate JSON, CSV, HTML, and PDF reports
        report_paths = self.report_generator.generate_all(
            val_result=res.validation_result,
            epinn_output=res.epinn_output,
            credit_summary=res.credit_summary,
            decision=res.verification_decision,
            input_name=run_name,
            pe_aux_mode=self.pe_aux_mode,
        )

        return {
            "result": res,
            "decision": res.verification_decision.decision,
            "verified_credits": res.verification_decision.verified_credits_tco2e,
            "reports": report_paths,
        }
