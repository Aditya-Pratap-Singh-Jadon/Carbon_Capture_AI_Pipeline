"""End-to-End Integration and Verification System Tests."""

import pytest
from pathlib import Path
import pandas as pd
from carbon_capture.pipeline.training_pipeline import TrainingPipeline
from carbon_capture.pipeline.end_to_end import EndToEndPipeline
from carbon_capture.input.synthetic import SyntheticDataGenerator

@pytest.fixture(scope="module")
def trained_artifacts(tmp_path_factory):
    """Train a fast toy E-PINN and return model and scaler paths."""
    out_dir = tmp_path_factory.mktemp("models")
    model_dir = out_dir / "epinn"
    scaler_dir = out_dir / "scalers"

    gen = SyntheticDataGenerator(seed=42)
    train_df = gen.generate_normal_regime(n_samples=60)
    val_df = gen.generate_normal_regime(n_samples=20)

    data_dir = tmp_path_factory.mktemp("data")
    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    cfg = {
        "random_seed": 42,
        "model": {
            "architecture": {"input_dim": 26, "hidden_dims": [32, 16], "activation": "silu"},
            "training": {"epochs": 3, "batch_size": 16, "learning_rate": 0.01, "early_stopping_patience": 5},
        },
        "physics": {"pe_aux_mode": "verbatim"},
    }

    trainer_pipe = TrainingPipeline(cfg)
    trainer_pipe.run(
        train_path=train_path,
        val_path=val_path,
        model_save_dir=model_dir,
        scaler_save_dir=scaler_dir,
    )

    return model_dir / "epinn_final.pt", scaler_dir / "canonical_scaler.json"

def test_full_pipeline_clean(trained_artifacts, tmp_path):
    model_path, scaler_path = trained_artifacts
    reports_dir = tmp_path / "reports"

    pipeline = EndToEndPipeline(
        model_path=model_path,
        scaler_path=scaler_path,
        reports_dir=reports_dir,
    )

    gen = SyntheticDataGenerator(seed=99)
    test_df = gen.generate_normal_regime(n_samples=25)

    res = pipeline.verify_and_report(test_df, run_name="test_clean_run")

    assert res["decision"] in ["VERIFIED", "NOT VERIFIED"]
    assert res["reports"]["json"].exists()
    assert res["reports"]["csv"].exists()
    assert res["reports"]["html"].exists()
    assert res["reports"]["pdf"].exists()

def test_pipeline_shuffled_column_resilience(trained_artifacts, tmp_path):
    model_path, scaler_path = trained_artifacts
    reports_dir = tmp_path / "reports_shuffled"

    pipeline = EndToEndPipeline(
        model_path=model_path,
        scaler_path=scaler_path,
        reports_dir=reports_dir,
    )

    gen = SyntheticDataGenerator(seed=123)
    clean_df = gen.generate_normal_regime(n_samples=10)
    shuffled_cols = list(clean_df.columns)
    import random
    random.shuffle(shuffled_cols)
    shuffled_df = clean_df[shuffled_cols]

    res = pipeline.verify_and_report(shuffled_df, run_name="test_shuffled_run")

    # Column ordering must be safely canonicalized
    assert res["result"].validation_result.canonical_columns[:29] == list(clean_df.columns[:29])

def test_pipeline_corrupted_data_rejection(trained_artifacts, tmp_path):
    model_path, scaler_path = trained_artifacts
    reports_dir = tmp_path / "reports_corrupt"

    pipeline = EndToEndPipeline(
        model_path=model_path,
        scaler_path=scaler_path,
        reports_dir=reports_dir,
    )

    gen = SyntheticDataGenerator(seed=456)
    clean_df = gen.generate_normal_regime(n_samples=15)
    corrupted_df, _ = gen.generate_corrupted_dataset(clean_df, "sensor_bias")

    res = pipeline.verify_and_report(corrupted_df, run_name="test_corrupt_run")

    # Corrupted purity (> 1.0) must be rejected
    assert res["decision"] == "NOT VERIFIED"
    assert res["verified_credits"] == 0.0
