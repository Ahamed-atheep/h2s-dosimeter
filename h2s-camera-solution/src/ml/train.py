"""
ML Model Training and Benchmarking Pipeline for H2S Passive Dosimeter.
Trains and compares XGBoost, Random Forest, and Ridge Regressors using GroupKFold CV.
Prevents data leakage across physical strip/experiment groups.
"""

import time
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb

from src.utils.config import (
    ml_config, features_config, MODELS_DIR, PROCESSED_DATA_DIR, ensure_directories
)
from src.utils.logger import get_logger, console
from src.ml.dataset import (
    load_dataset, split_grouped_dataset, get_feature_matrices, validate_and_engineer_features
)
from rich.table import Table

logger = get_logger("ML_Train")


def build_model(model_type: str = "xgboost") -> Any:
    """
    Instantiate specified model architecture with configured hyperparameters.
    """
    model_type = model_type.lower()
    
    if model_type == "xgboost":
        return xgb.XGBRegressor(**ml_config.XGB_PARAMS)
    elif model_type in ["random_forest", "rf"]:
        return RandomForestRegressor(**ml_config.RF_PARAMS)
    elif model_type in ["ridge", "linear"]:
        return Ridge(**ml_config.RIDGE_PARAMS)
    else:
        raise ValueError(f"Unsupported model type '{model_type}'. Choose from 'xgboost', 'random_forest', 'ridge'.")


def evaluate_cross_validation(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    n_splits: int = 5
) -> Dict[str, Any]:
    """
    Perform GroupKFold cross-validation on training data to ensure zero group leakage.
    """
    gkf = GroupKFold(n_splits=n_splits)
    
    mae_list = []
    rmse_list = []
    r2_list = []
    oof_predictions = np.zeros_like(y_train)
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups_train)):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_va, y_va = X_train[val_idx], y_train[val_idx]
        
        # Fit scaler on fold training set only
        scaler = RobustScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)
        
        # Fit model
        model = build_model(model_type)
        model.fit(X_tr_scaled, y_tr)
        
        # Predict on validation fold
        y_pred = model.predict(X_va_scaled)
        oof_predictions[val_idx] = y_pred
        
        mae = mean_absolute_error(y_va, y_pred)
        rmse = root_mean_squared_error(y_va, y_pred)
        r2 = r2_score(y_va, y_pred)
        
        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)
        
    cv_summary = {
        "model_type": model_type,
        "n_splits": n_splits,
        "cv_mae_mean": float(np.mean(mae_list)),
        "cv_mae_std": float(np.std(mae_list)),
        "cv_rmse_mean": float(np.mean(rmse_list)),
        "cv_rmse_std": float(np.std(rmse_list)),
        "cv_r2_mean": float(np.mean(r2_list)),
        "cv_r2_std": float(np.std(r2_list)),
        "oof_mae": float(mean_absolute_error(y_train, oof_predictions)),
        "oof_rmse": float(root_mean_squared_error(y_train, oof_predictions)),
        "oof_r2": float(r2_score(y_train, oof_predictions))
    }
    
    return cv_summary


def train_model_pipeline(
    data_path: Optional[Path] = None,
    model_type: str = "xgboost",
    save_artifact: bool = True,
    output_model_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Full training pipeline:
    1. Load data
    2. GroupShuffleSplit into train and test sets
    3. Run GroupKFold CV benchmark
    4. Fit final model on full training set
    5. Evaluate on held-out test set
    6. Serialize model artifact
    """
    ensure_directories()
    
    # 1. Load and prepare dataset
    raw_df = load_dataset(data_path)
    train_df, test_df = split_grouped_dataset(
        raw_df, test_size=ml_config.TEST_SIZE, random_seed=ml_config.RANDOM_SEED
    )
    
    # Save processed train and test sets if not using custom path
    if output_model_path is None:
        train_path = PROCESSED_DATA_DIR / "train_features.csv"
        test_path = PROCESSED_DATA_DIR / "test_features.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
    
    X_train, y_train, groups_train, feature_names = get_feature_matrices(train_df)
    X_test, y_test, groups_test, _ = get_feature_matrices(test_df)
    
    # 2. Run GroupKFold Cross-Validation
    logger.info(f"Running {ml_config.N_SPLITS_CV}-fold GroupKFold Cross-Validation for '{model_type}'...")
    cv_metrics = evaluate_cross_validation(
        model_type, X_train, y_train, groups_train, n_splits=ml_config.N_SPLITS_CV
    )
    
    # 3. Fit Scaler and Final Model on entire train set
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    final_model = build_model(model_type)
    start_time = time.time()
    final_model.fit(X_train_scaled, y_train)
    fit_duration = time.time() - start_time
    
    # 4. Evaluate on unseen held-out test set
    y_test_pred = final_model.predict(X_test_scaled)
    test_mae = float(mean_absolute_error(y_test, y_test_pred))
    test_rmse = float(root_mean_squared_error(y_test, y_test_pred))
    test_r2 = float(r2_score(y_test, y_test_pred))
    
    logger.info(
        f"[{model_type.upper()}] Test Results -> MAE: {test_mae:.3f} ppm*h | "
        f"RMSE: {test_rmse:.3f} ppm*h | R²: {test_r2:.4f}"
    )
    
    # Check feature importance if available
    feature_importances = {}
    if hasattr(final_model, "feature_importances_"):
        for name, imp in zip(feature_names, final_model.feature_importances_):
            feature_importances[name] = float(imp)
    elif hasattr(final_model, "coef_"):
        for name, coef in zip(feature_names, final_model.coef_):
            feature_importances[name] = float(coef)
            
    # Assemble package artifact
    is_synthetic = bool(raw_df.get("is_synthetic", pd.Series([False])).iloc[0])
    artifact = {
        "model": final_model,
        "scaler": scaler,
        "model_type": model_type,
        "feature_names": feature_names,
        "target_column": features_config.TARGET_COLUMN,
        "is_synthetic": is_synthetic,
        "cv_metrics": cv_metrics,
        "test_metrics": {
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2
        },
        "feature_importances": feature_importances,
        "training_metadata": {
            "trained_at": datetime.utcnow().isoformat(),
            "train_samples": len(train_df),
            "train_strips": int(len(np.unique(groups_train))),
            "test_samples": len(test_df),
            "test_strips": int(len(np.unique(groups_test))),
            "fit_duration_sec": round(fit_duration, 4)
        }
    }
    
    if save_artifact:
        if output_model_path is not None:
            artifact_path = Path(output_model_path)
        else:
            model_filename = f"model_{model_type}.joblib"
            artifact_path = MODELS_DIR / model_filename
        joblib.dump(artifact, artifact_path)
        logger.info(f"Saved model artifact to: {artifact_path}")
        
    return artifact


def train_and_compare_all_models(data_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Train and benchmark all candidate models (XGBoost, Random Forest, Ridge),
    print a side-by-side comparison table, and save the best performing model as 'best_model.joblib'.
    """
    models_to_test = ["xgboost", "random_forest", "ridge"]
    results = {}
    
    table = Table(title="H2S Cumulative Dose Regression - Model Comparison")
    table.add_column("Model Architecture", style="cyan", no_wrap=True)
    table.add_column("CV MAE (ppm*h)", justify="right", style="magenta")
    table.add_column("CV R2", justify="right", style="green")
    table.add_column("Test MAE (ppm*h)", justify="right", style="magenta")
    table.add_column("Test RMSE (ppm*h)", justify="right", style="red")
    table.add_column("Test R2", justify="right", style="green")
    
    best_model_type = None
    best_test_r2 = -float("inf")
    best_artifact = None
    
    for m_type in models_to_test:
        logger.info(f"\n--- Training {m_type.upper()} ---")
        artifact = train_model_pipeline(data_path=data_path, model_type=m_type, save_artifact=True)
        results[m_type] = artifact
        
        cv = artifact["cv_metrics"]
        test = artifact["test_metrics"]
        
        table.add_row(
            m_type.upper(),
            f"{cv['cv_mae_mean']:.3f} +/- {cv['cv_mae_std']:.3f}",
            f"{cv['cv_r2_mean']:.4f}",
            f"{test['test_mae']:.3f}",
            f"{test['test_rmse']:.3f}",
            f"{test['test_r2']:.4f}"
        )
        
        if test["test_r2"] > best_test_r2:
            best_test_r2 = test["test_r2"]
            best_model_type = m_type
            best_artifact = artifact
            
    console.print(table)
    
    # Save best model copy
    best_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_artifact, best_path)
    logger.info(f"\n[BEST MODEL SELECTED]: {best_model_type.upper()} (Test R2 = {best_test_r2:.4f}) -> Saved to '{best_path}'")
    
    return results


if __name__ == "__main__":
    train_and_compare_all_models()
