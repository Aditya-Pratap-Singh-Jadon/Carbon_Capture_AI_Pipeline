"""CLI script to train MultivariateGaussianModel for telemetry anomaly detection."""

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from carbon_capture.input.telemetry_stream import TelemetryStream
from carbon_capture.anomaly.gaussian import MultivariateGaussianModel, GAUSSIAN_FEATURES
from carbon_capture.utils.logging import setup_logging, get_logger

setup_logging("INFO")
logger = get_logger("train_anomaly_detector")

def main():
    parser = argparse.ArgumentParser(description="Train Anomaly Detector")
    parser.add_argument("--input", type=str, required=True, help="Input labeled CSV path")
    parser.add_argument("--output", type=str, default="models/anomaly/gaussian_model.pkl", help="Output model path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    X_list = []
    with open(input_path, "r", encoding="utf-8") as f:
        stream = TelemetryStream(f)
        for is_valid, features, ts, scenario_label, err in stream:
            if not is_valid:
                logger.warning(f"Skipping invalid line: {err}")
                continue
                
            # Make sure there are no missing values in the training set
            if any(v is None for v in features.values()):
                logger.warning(f"Skipping row with missing sensor values at {ts}")
                continue
                
            # Only calibrate on normal data
            if scenario_label not in ["normal", "startup", None]:
                continue
                
            vec = [features[k] for k in GAUSSIAN_FEATURES]
            X_list.append(vec)

    if not X_list:
        logger.error("No valid data found for training.")
        sys.exit(1)

    X = np.array(X_list, dtype=np.float64)
    logger.info(f"Loaded {len(X)} samples for calibration.")

    model = MultivariateGaussianModel(regularization=1e-4)
    model.calibrate(X)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)

if __name__ == "__main__":
    main()
