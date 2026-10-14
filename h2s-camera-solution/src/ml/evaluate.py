"""
Evaluation and Diagnostic Engine for H2S Passive Dosimeter ML Models.
Generates comprehensive regression metrics, error breakdowns by dose tier,
and publication-quality diagnostic plots.
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional

from sklearn.metrics import (
    mean_absolute_error, root_mean_squared_error, r2_score, median_absolute_error
)

from src.utils.config import (
    MODELS_DIR, RESULTS_DIR, PROCESSED_DATA_DIR, ensure_directories
)
from src.utils.logger import get_logger
from src.ml.dataset import load_dataset, validate_and_engineer_features, get_feature_matrices

logger = get_logger("ML_Evaluate")

# Set matplotlib aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def evaluate_model_performance(
    model_artifact_path: Optional[Path] = None,
    test_data_path: Optional[Path] = None,
    save_plots: bool = True
) -> Dict[str, Any]:
    """
    Evaluate a serialized model artifact against a test dataset and produce diagnostic artifacts.
    """
    ensure_directories()
    
    # 1. Load model artifact
    if model_artifact_path is None:
        model_artifact_path = MODELS_DIR / "best_model.joblib"
        if not model_artifact_path.exists():
            model_artifact_path = MODELS_DIR / "model_xgboost.joblib"
            
    model_artifact_path = Path(model_artifact_path)
    if not model_artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found at '{model_artifact_path}'. Train a model first.")
        
    artifact = joblib.load(model_artifact_path)
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_names = artifact["feature_names"]
    target_col = artifact.get("target_column", "cumulative_dose_ppm_h")
    model_type = artifact.get("model_type", "unknown")
    
    # 2. Load test data
    if test_data_path is None:
        test_data_path = PROCESSED_DATA_DIR / "test_features.csv"
        
    test_data_path = Path(test_data_path)
    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data file not found at '{test_data_path}'.")
        
    df_test = pd.read_csv(test_data_path)
    df_test = validate_and_engineer_features(df_test)
    
    X_test = df_test[feature_names].values.astype(np.float32)
    y_true = df_test[target_col].values.astype(np.float32)
    
    # 3. Model Inference
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    residuals = y_pred - y_true
    abs_errors = np.abs(residuals)
    
    # 4. Overall Metrics
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    med_ae = float(median_absolute_error(y_true, y_pred))
    max_err = float(np.max(abs_errors))
    
    # 5. Dose-tier performance breakdown
    # Low dose (< 5 ppm*h), Medium dose (5 - 20 ppm*h), High dose (> 20 ppm*h)
    tier_low_mask = y_true < 5.0
    tier_med_mask = (y_true >= 5.0) & (y_true <= 20.0)
    tier_high_mask = y_true > 20.0
    
    def get_tier_stats(mask: np.ndarray, name: str) -> Dict[str, Any]:
        if not np.any(mask):
            return {"name": name, "count": 0, "mae": None, "rmse": None}
        return {
            "name": name,
            "count": int(np.sum(mask)),
            "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
            "rmse": float(root_mean_squared_error(y_true[mask], y_pred[mask])),
            "mean_relative_error_pct": float(np.mean(abs_errors[mask] / np.maximum(y_true[mask], 0.1)) * 100.0)
        }
        
    dose_tier_breakdown = {
        "low_dose_under_5_ppm_h": get_tier_stats(tier_low_mask, "Low Dose (<5 ppm*h)"),
        "medium_dose_5_to_20_ppm_h": get_tier_stats(tier_med_mask, "Medium Dose (5-20 ppm*h)"),
        "high_dose_over_20_ppm_h": get_tier_stats(tier_high_mask, "High Dose (>20 ppm*h)")
    }
    
    metrics_summary = {
        "model_type": model_type,
        "model_file": model_artifact_path.name,
        "test_dataset": test_data_path.name,
        "n_test_samples": len(df_test),
        "overall_metrics": {
            "MAE_ppm_h": round(mae, 4),
            "RMSE_ppm_h": round(rmse, 4),
            "R2_score": round(r2, 4),
            "Median_Absolute_Error_ppm_h": round(med_ae, 4),
            "Max_Absolute_Error_ppm_h": round(max_err, 4)
        },
        "dose_tier_breakdown": dose_tier_breakdown
    }
    
    # Save metrics JSON
    metrics_json_path = RESULTS_DIR / "metrics_summary.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
        
    logger.info(f"Evaluation Metrics Summary:")
    logger.info(f"  MAE: {mae:.3f} ppm*h | RMSE: {rmse:.3f} ppm*h | R²: {r2:.4f} | MedAE: {med_ae:.3f} ppm*h")
    logger.info(f"Saved metrics summary to: {metrics_json_path}")
    
    # 6. Generate Diagnostic Plots
    if save_plots:
        _generate_diagnostic_plots(
            y_true=y_true,
            y_pred=y_pred,
            residuals=residuals,
            feature_names=feature_names,
            feature_importances=artifact.get("feature_importances", {}),
            model_type=model_type,
            r2=r2,
            mae=mae
        )
        
    return metrics_summary


def _generate_diagnostic_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residuals: np.ndarray,
    feature_names: list,
    feature_importances: dict,
    model_type: str,
    r2: float,
    mae: float
):
    """
    Render and save diagnostic figures to the results directory.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- Plot 1: Actual vs Predicted Scatter ---
    plt.figure(figsize=(7, 6), dpi=150)
    plt.scatter(y_true, y_pred, alpha=0.5, color="#1f77b4", edgecolors="none", s=25, label="Test Samples")
    
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    line_pts = np.linspace(min_val, max_val, 100)
    
    plt.plot(line_pts, line_pts, color="black", linestyle="--", linewidth=1.5, label="1:1 Perfect Prediction")
    plt.plot(line_pts, line_pts * 1.15, color="gray", linestyle=":", linewidth=1, label="+15% Bounds")
    plt.plot(line_pts, line_pts * 0.85, color="gray", linestyle=":", linewidth=1, label="-15% Bounds")
    
    plt.title(f"Actual vs. Predicted Cumulative H₂S Dose\nModel: {model_type.upper()} ($R^2 = {r2:.3f}$, MAE = {mae:.2f} ppm·h)", fontsize=11, fontweight="bold")
    plt.xlabel("Actual Cumulative Dose (ppm·h)", fontsize=10)
    plt.ylabel("Predicted Cumulative Dose (ppm·h)", fontsize=10)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plot1_path = RESULTS_DIR / "actual_vs_predicted.png"
    plt.savefig(plot1_path)
    plt.close()
    
    # --- Plot 2: Residual Analysis (Residuals vs Predicted & Distribution) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    
    axes[0].scatter(y_pred, residuals, alpha=0.5, color="#2ca02c", edgecolors="none", s=25)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1.2)
    axes[0].set_title("Residuals vs. Predicted Dose", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Predicted Cumulative Dose (ppm·h)", fontsize=10)
    axes[0].set_ylabel("Residual (Predicted - Actual) (ppm·h)", fontsize=10)
    
    sns.histplot(residuals, kde=True, ax=axes[1], color="#2ca02c", bins=30)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1.2)
    axes[1].set_title("Residual Distribution (Error Normality)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Residual (ppm·h)", fontsize=10)
    axes[1].set_ylabel("Sample Count", fontsize=10)
    
    plt.tight_layout()
    plot2_path = RESULTS_DIR / "residual_analysis.png"
    plt.savefig(plot2_path)
    plt.close()
    
    # --- Plot 3: Absolute Error vs True Dose Spectrum ---
    plt.figure(figsize=(7, 5), dpi=150)
    abs_err = np.abs(residuals)
    plt.scatter(y_true, abs_err, alpha=0.5, color="#d62728", edgecolors="none", s=25)
    
    # Add rolling median curve
    df_err = pd.DataFrame({"y_true": y_true, "abs_err": abs_err}).sort_values("y_true")
    df_err["rolling_mae"] = df_err["abs_err"].rolling(window=max(10, len(df_err)//20), min_periods=5).mean()
    plt.plot(df_err["y_true"], df_err["rolling_mae"], color="black", linewidth=2, label="Moving Mean MAE")
    
    plt.title("Prediction Error Magnitude across Dose Spectrum", fontsize=11, fontweight="bold")
    plt.xlabel("True Cumulative Dose (ppm·h)", fontsize=10)
    plt.ylabel("Absolute Error |Predicted - True| (ppm·h)", fontsize=10)
    plt.legend(frameon=True)
    plt.tight_layout()
    plot3_path = RESULTS_DIR / "error_vs_dose.png"
    plt.savefig(plot3_path)
    plt.close()
    
    # --- Plot 4: Feature Importance Ranking (if available) ---
    if feature_importances:
        plt.figure(figsize=(8, 6), dpi=150)
        sorted_feats = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:12]
        feat_labels = [k for k, v in sorted_feats][::-1]
        feat_vals = [v for k, v in sorted_feats][::-1]
        
        plt.barh(range(len(feat_labels)), feat_vals, color="#9467bd", align="center")
        plt.yticks(range(len(feat_labels)), feat_labels)
        plt.title(f"Top Predictive Features — {model_type.upper()}", fontsize=11, fontweight="bold")
        plt.xlabel("Relative Feature Importance / Coefficient Weight", fontsize=10)
        plt.tight_layout()
        plot4_path = RESULTS_DIR / "feature_importance.png"
        plt.savefig(plot4_path)
        plt.close()
        
    logger.info(f"Saved diagnostic plots to: {RESULTS_DIR}")


if __name__ == "__main__":
    evaluate_model_performance()
