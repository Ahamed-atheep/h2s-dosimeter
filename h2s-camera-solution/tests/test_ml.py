"""
Unit and Integration Tests for Module 3 (ML Regression & Calibration Engine).
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.config import physics_config, features_config, ml_config
from src.ml.synthetic import (
    temperature_factor, humidity_factor, calculate_darkening_progress, generate_synthetic_dataset
)
from src.ml.dataset import (
    validate_and_engineer_features, split_grouped_dataset, get_feature_matrices
)
from src.ml.train import train_model_pipeline, build_model
from src.ml.evaluate import evaluate_model_performance
from src.ml.predict import DosimeterPredictor, DosePredictionResult


def test_temperature_factor():
    """Test Arrhenius temperature acceleration properties."""
    # At reference temperature (25°C), factor must be 1.0
    factor_ref = temperature_factor(np.array([25.0]))
    assert np.isclose(factor_ref[0], 1.0, atol=1e-3)
    
    # Higher temperature should accelerate reaction
    factor_hot = temperature_factor(np.array([40.0]))
    assert factor_hot[0] > 1.0
    
    # Lower temperature should decelerate reaction
    factor_cold = temperature_factor(np.array([15.0]))
    assert factor_cold[0] < 1.0


def test_humidity_factor():
    """Test humectant humidity scaling."""
    factor_ref = humidity_factor(np.array([50.0]))
    factor_dry = humidity_factor(np.array([20.0]))
    factor_humid = humidity_factor(np.array([80.0]))
    
    assert 0.6 <= factor_dry[0] <= 1.2
    assert factor_dry[0] < factor_ref[0] < factor_humid[0]


def test_synthetic_data_generation(tmp_path):
    """Test generating a small synthetic dataset."""
    csv_path = tmp_path / "test_synthetic.csv"
    df, meta = generate_synthetic_dataset(
        n_samples=200,
        n_strips=20,
        random_seed=42,
        output_path=csv_path
    )
    
    assert len(df) == 200
    assert csv_path.exists()
    assert meta["is_synthetic"] is True
    assert "cumulative_dose_ppm_h" in df.columns
    assert df["cumulative_dose_ppm_h"].isnull().sum() == 0
    assert df["strip_id"].nunique() == 20
    
    # Check physical ranges
    assert df["L_mean"].min() >= 15.0
    assert df["L_mean"].max() <= 100.0


def test_leakage_free_group_split(tmp_path):
    """Verify GroupShuffleSplit completely isolates strip_ids."""
    df, _ = generate_synthetic_dataset(
        n_samples=300,
        n_strips=30,
        random_seed=123,
        output_path=tmp_path / "split_test.csv"
    )
    
    train_df, test_df = split_grouped_dataset(df, test_size=0.2, random_seed=42)
    
    train_strips = set(train_df["strip_id"].unique())
    test_strips = set(test_df["strip_id"].unique())
    
    # CRITICAL: zero overlap
    overlap = train_strips.intersection(test_strips)
    assert len(overlap) == 0, f"Data leakage detected! Strips present in both sets: {overlap}"


def test_feature_engineering():
    """Test automated feature derivations."""
    raw_dict = {
        "strip_id": ["STRIP_001"],
        "h2s_concentration_ppm": [5.0],
        "exposure_time_minutes": [120.0],
        "temperature_c": [25.0],
        "humidity_percent": [50.0],
        "R_mean": [180.0], "G_mean": [170.0], "B_mean": [150.0],
        "R_std": [2.0], "G_std": [2.0], "B_std": [2.0],
        "H_mean": [30.0], "S_mean": [15.0], "V_mean": [70.0],
        "H_std": [1.0], "S_std": [1.0], "V_std": [1.0],
        "L_mean": [72.0], "a_mean": [1.5], "b_mean": [5.5],
        "L_std": [1.5], "a_std": [0.8], "b_std": [1.0]
    }
    df = pd.DataFrame(raw_dict)
    proc_df = validate_and_engineer_features(df)
    
    assert "cumulative_dose_ppm_h" in proc_df.columns
    assert np.isclose(proc_df["cumulative_dose_ppm_h"].iloc[0], 10.0) # 5 ppm * 2h = 10 ppm*h
    assert "delta_E" in proc_df.columns
    assert "chroma" in proc_df.columns
    assert "hue_angle" in proc_df.columns
    assert "ratio_RG" in proc_df.columns


def test_train_and_predict_pipeline(tmp_path):
    """End-to-end test of training a model and running inference."""
    data_path = tmp_path / "pipeline_data.csv"
    model_path = tmp_path / "test_model_xgboost.joblib"
    generate_synthetic_dataset(n_samples=400, n_strips=40, random_seed=42, output_path=data_path)
    
    # Train XGBoost
    artifact = train_model_pipeline(data_path=data_path, model_type="xgboost", save_artifact=True, output_model_path=model_path)
    assert "model" in artifact
    assert artifact["test_metrics"]["test_r2"] > 0.85
    
    # Load predictor
    predictor = DosimeterPredictor(model_path)
    sample_input = {
        "R_mean": 120.0, "G_mean": 110.0, "B_mean": 95.0,
        "R_std": 2.5, "G_std": 2.5, "B_std": 2.0,
        "H_mean": 32.0, "S_mean": 20.0, "V_mean": 45.0,
        "H_std": 1.5, "S_std": 1.5, "V_std": 1.5,
        "L_mean": 48.0, "a_mean": 3.0, "b_mean": 6.5,
        "L_std": 2.0, "a_std": 1.0, "b_std": 1.2,
        "temperature_c": 28.0,
        "humidity_percent": 60.0,
        "exposure_time_minutes": 240.0
    }
    
    result = predictor.predict_single(sample_input)
    assert isinstance(result, DosePredictionResult)
    assert result.cumulative_dose_ppm_h > 0.0
    assert result.confidence_interval_low <= result.cumulative_dose_ppm_h <= result.confidence_interval_high
    assert result.risk_level in ["SAFE / NORMAL", "WARNING / ELEVATED", "DANGER / ACTION REQUIRED", "CRITICAL / THRESHOLD EXCEEDED"]
