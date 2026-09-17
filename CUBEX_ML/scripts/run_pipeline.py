"""CLI script for full end-to-end execution: Ingestion -> E-PINN -> Credit -> Reports."""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from carbon_capture.pipeline.end_to_end import EndToEndPipeline
from carbon_capture.utils.logging import setup_logging, get_logger

setup_logging("INFO")
logger = get_logger("run_pipeline")

def main():
    parser = argparse.ArgumentParser(description="Execute full verification and reporting pipeline.")
    parser.add_argument("--input", type=str, default="data/synthetic/test.csv", help="Input CSV path")
    parser.add_argument("--run-name", type=str, default="industrial_verification_run", help="Report identifier")
    parser.add_argument("--model", type=str, default="models/epinn/epinn_final.pt", help="Model checkpoint path")
    parser.add_argument("--scaler", type=str, default="models/scalers/canonical_scaler.json", help="Scaler path")
    parser.add_argument("--reports-dir", type=str, default="reports/generated", help="Reports output directory")
    args = parser.parse_args()

    pipeline = EndToEndPipeline(
        model_path=args.model,
        scaler_path=args.scaler,
        reports_dir=args.reports_dir,
    )

    output = pipeline.verify_and_report(input_source=args.input, run_name=args.run_name)

    print("\n" + "="*70)
    print("END-TO-END PIPELINE EXECUTION SUMMARY")
    print("="*70)
    print(f"Run Name:           {args.run_name}")
    print(f"Decision:           {output['decision']}")
    print(f"Verified Credits:   {output['verified_credits']:,.2f} tCO2e")
    print("\nGenerated Deliverables:")
    for fmt, p in output["reports"].items():
        print(f" - [{fmt.upper()}]: {p}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
