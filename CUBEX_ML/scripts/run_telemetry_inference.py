"""CLI script to run live anomaly detection on streaming telemetry."""

import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from carbon_capture.input.telemetry_stream import TelemetryStream
from carbon_capture.anomaly.gaussian import MultivariateGaussianModel
from carbon_capture.utils.logging import setup_logging, get_logger

setup_logging("WARNING") # suppress info logs so stdout can just be json
logger = get_logger("run_telemetry_inference")

def main():
    parser = argparse.ArgumentParser(description="Run Live Anomaly Detection")
    parser.add_argument("--model", type=str, default="models/anomaly/gaussian_model.pkl", help="Path to trained anomaly model")
    parser.add_argument("--threshold", type=float, default=0.01, help="Statistical score threshold for anomaly")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}. Please train first.")
        sys.exit(1)

    model = MultivariateGaussianModel()
    model.load(model_path)

    # Read from standard input (streaming from CUBEX_COM stdout)
    stream = TelemetryStream(sys.stdin)
    
    for is_valid, features, ts, scenario_label, err in stream:
        if not is_valid:
            print(json.dumps({
                "timestamp": ts if ts else "UNKNOWN",
                "status": "DATA_QUALITY_FAILURE",
                "error": err
            }))
            continue
            
        result = model.evaluate_single(features)
        
        if not result.is_valid_input:
            print(json.dumps({
                "timestamp": ts,
                "status": "DATA_QUALITY_FAILURE",
                "error": result.diagnostics
            }))
            continue
            
        status = "NORMAL" if result.statistical_score >= args.threshold else "ANOMALOUS"
        
        out = {
            "timestamp": ts,
            "status": status,
            "anomaly_score": result.mahalanobis_squared,
            "confidence": result.statistical_score,
            "detector": "MultivariateGaussianModel",
            "diagnostics": result.diagnostics,
            "data_quality": "PASSED"
        }
        print(json.dumps(out))

if __name__ == "__main__":
    main()
