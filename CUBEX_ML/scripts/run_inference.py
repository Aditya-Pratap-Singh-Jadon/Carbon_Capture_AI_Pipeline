"""CLI script to run inference and verification on an input file."""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from carbon_capture.pipeline.inference_pipeline import InferencePipeline
from carbon_capture.utils.logging import setup_logging, get_logger

setup_logging("INFO")
logger = get_logger("run_inference")

def main():
    parser = argparse.ArgumentParser(description="Run carbon capture E-PINN inference.")
    parser.add_argument("--input", type=str, default="data/synthetic/test.csv", help="Input CSV path")
    parser.add_argument("--model", type=str, default="models/epinn/epinn_final.pt", help="Model checkpoint path")
    parser.add_argument("--scaler", type=str, default="models/scalers/canonical_scaler.json", help="Scaler path")
    args = parser.parse_args()

    pipeline = InferencePipeline(
        model_path=args.model,
        scaler_path=args.scaler,
    )
    result = pipeline.run(args.input)

    print("\n" + "="*60)
    print("VERIFICATION RESULT SUMMARY")
    print("="*60)
    print(f"Status:             {result.verification_decision.decision}")
    print(f"Verified Credits:   {result.verification_decision.verified_credits_tco2e:,.2f} tCO2e")
    print(f"Provisional:        {result.verification_decision.provisional_credits_tco2e:,.2f} tCO2e")
    print(f"Confidence Score:   {result.verification_decision.confidence_score * 100:.1f}%")
    print(f"Physics Consistent: {result.verification_decision.physics_consistent}")
    print(f"Data Quality Valid: {result.verification_decision.data_quality_passed}")
    if result.verification_decision.rejection_reasons:
        print("\nRejection Reasons:")
        for r in result.verification_decision.rejection_reasons:
            print(f" - {r}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
