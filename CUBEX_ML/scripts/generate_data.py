"""CLI Script to generate synthetic industrial datasets for training and verification testing."""

import sys
from pathlib import Path
# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from carbon_capture.input.synthetic import SyntheticDataGenerator
from carbon_capture.input.canonical_order import CANONICAL_INPUT_ORDER
from carbon_capture.utils.logging import setup_logging, get_logger
import numpy as np

setup_logging("INFO")
logger = get_logger("generate_data")

def main():
    output_dir = Path("data/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)

    gen = SyntheticDataGenerator(seed=42)

    logger.info("Generating training synthetic dataset (steady_state + peak_load + turndown)...")
    df_steady = gen.generate_normal_regime(n_samples=800, regime="steady_state")
    df_peak = gen.generate_normal_regime(n_samples=400, regime="peak_load")
    df_turndown = gen.generate_normal_regime(n_samples=300, regime="turndown")

    import pandas as pd
    train_df = pd.concat([df_steady, df_peak, df_turndown], ignore_index=True)
    val_df = gen.generate_normal_regime(n_samples=300, regime="steady_state")
    test_df = gen.generate_normal_regime(n_samples=300, regime="peak_load")

    # Save clean datasets
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info(f"Saved clean datasets to {train_path}, {val_path}, {test_path}")

    # Generate targets for reference
    train_targets = gen.compute_ground_truth_targets(train_df)
    train_targets.to_csv(output_dir / "train_targets.csv", index=False)

    # 1. Shuffled columns dataset (to test canonical order restoration)
    shuffled_cols = list(np.random.default_rng(123).permutation(CANONICAL_INPUT_ORDER))
    shuffled_df = test_df[shuffled_cols].copy()
    shuffled_path = output_dir / "test_shuffled_columns.csv"
    shuffled_df.to_csv(shuffled_path, index=False)
    logger.info(f"Saved shuffled-column dataset to {shuffled_path}")

    # 2. Corrupted purity sensor dataset
    corrupt_purity, desc1 = gen.generate_corrupted_dataset(test_df, "sensor_bias")
    purity_path = output_dir / "test_corrupted_purity.csv"
    corrupt_purity.to_csv(purity_path, index=False)
    logger.info(f"Saved corrupted purity dataset to {purity_path} ({desc1})")

    # 3. Corrupted negative power dataset
    corrupt_power, desc2 = gen.generate_corrupted_dataset(test_df, "negative_physical_quantity")
    power_path = output_dir / "test_corrupted_negative_power.csv"
    corrupt_power.to_csv(power_path, index=False)
    logger.info(f"Saved corrupted power dataset to {power_path} ({desc2})")

    # 4. Corrupted mass balance dataset
    corrupt_mass, desc3 = gen.generate_corrupted_dataset(test_df, "mass_balance_violation")
    mass_path = output_dir / "test_corrupted_mass_balance.csv"
    corrupt_mass.to_csv(mass_path, index=False)
    logger.info(f"Saved corrupted mass balance dataset to {mass_path} ({desc3})")

    logger.info("Synthetic data generation completed successfully.")

if __name__ == "__main__":
    main()
