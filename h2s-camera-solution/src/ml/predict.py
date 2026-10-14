"""
Inference and Uncertainty Estimation Engine for H2S Passive Dosimeter.
Translates extracted color features and ambient parameters into estimated cumulative dose
with confidence intervals and occupational safety risk banding.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict

from src.utils.config import (
    physics_config, features_config, MODELS_DIR
)
from src.utils.logger import get_logger

logger = get_logger("ML_Predict")


@dataclass
class DosePredictionResult:
    """
    Structured outcome of an H2S dose estimation inference.
    """
    cumulative_dose_ppm_h: float
    confidence_interval_low: float
    confidence_interval_high: float
    uncertainty_margin: float
    twa_8h_equivalent_ppm: float
    risk_level: str
    risk_color: str
    action_recommendation: str
    model_type: str
    is_synthetic_model: bool
    temperature_c: float
    humidity_percent: float
    expiry_status: Optional[str] = "VALID"
    is_badge_valid: bool = True
    lighting_profile: Optional[str] = "BALANCED"
    cv_diagnostics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
        
    def to_json(self, indent: int = 2) -> str:
        d = self.to_dict()
        # Remove raw image arrays if present in diagnostics
        if "cv_diagnostics" in d and d["cv_diagnostics"]:
            d["cv_diagnostics"].pop("debug_overlay", None)
            d["cv_diagnostics"].pop("aligned_badge", None)
            d["cv_diagnostics"].pop("normalized_sensing_patch", None)
        return json.dumps(d, indent=indent)


class DosimeterPredictor:
    """
    Loaded ML Model Predictor for H2S Dosimeter.
    """
    def __init__(self, model_artifact_path: Optional[Path] = None):
        if model_artifact_path is None:
            model_artifact_path = MODELS_DIR / "best_model.joblib"
            if not model_artifact_path.exists():
                model_artifact_path = MODELS_DIR / "model_xgboost.joblib"
                
        self.model_path = Path(model_artifact_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at '{self.model_path}'. Please train a model first using 'python app.py train'."
            )
            
        self.artifact = joblib.load(self.model_path)
        self.model = self.artifact["model"]
        self.scaler = self.artifact["scaler"]
        self.feature_names = self.artifact["feature_names"]
        self.model_type = self.artifact.get("model_type", "xgboost")
        self.is_synthetic = self.artifact.get("is_synthetic", True)
        
        # Determine uncertainty margin from CV RMSE or test RMSE
        test_metrics = self.artifact.get("test_metrics", {})
        cv_metrics = self.artifact.get("cv_metrics", {})
        self.uncertainty_sigma = test_metrics.get("test_rmse", cv_metrics.get("cv_rmse_mean", 1.2))

    def _classify_risk(self, dose_ppm_h: float) -> Tuple[str, str, str]:
        """
        Classify occupational risk based on DGMS / OISD standards.
        """
        if dose_ppm_h < physics_config.DOSE_ALERT_WARNING:
            return (
                "SAFE / NORMAL",
                "green",
                "Cumulative exposure is within permissible workplace limits. Badge is normal."
            )
        elif dose_ppm_h < physics_config.DOSE_ALERT_DANGER:
            return (
                "WARNING / ELEVATED",
                "yellow",
                "Elevated chronic H2S exposure detected. Inspect ventilation and monitor shift duration."
            )
        elif dose_ppm_h < physics_config.DOSE_ALERT_CRITICAL:
            return (
                "DANGER / ACTION REQUIRED",
                "red",
                "High cumulative dose approaching permissible 8-hour shift ceiling. Rotate worker to fresh air."
            )
        else:
            return (
                "CRITICAL / THRESHOLD EXCEEDED",
                "bright_red",
                "Shift maximum permissible dose exceeded! Evacuate area and report to safety officer immediately."
            )

    def predict_single(
        self,
        features_dict: Dict[str, Any]
    ) -> DosePredictionResult:
        """
        Run inference on a single feature dictionary.
        """
        # 1. Fill derived features if missing
        feat_copy = features_dict.copy()
        
        if "delta_E" not in feat_copy and {"L_mean", "a_mean", "b_mean"}.issubset(feat_copy):
            feat_copy["delta_E"] = float(np.sqrt(
                (feat_copy["L_mean"] - physics_config.BASELINE_L) ** 2 +
                (feat_copy["a_mean"] - physics_config.BASELINE_A) ** 2 +
                (feat_copy["b_mean"] - physics_config.BASELINE_B) ** 2
            ))
            
        if "chroma" not in feat_copy and {"a_mean", "b_mean"}.issubset(feat_copy):
            feat_copy["chroma"] = float(np.sqrt(feat_copy["a_mean"] ** 2 + feat_copy["b_mean"] ** 2))
            
        if "hue_angle" not in feat_copy and {"a_mean", "b_mean"}.issubset(feat_copy):
            feat_copy["hue_angle"] = float(np.degrees(np.arctan2(feat_copy["b_mean"], feat_copy["a_mean"])) % 360.0)

        if "ratio_RG" not in feat_copy and {"R_mean", "G_mean"}.issubset(feat_copy):
            feat_copy["ratio_RG"] = float(feat_copy["R_mean"] / max(feat_copy["G_mean"], 1e-4))

        if "ratio_GB" not in feat_copy and {"G_mean", "B_mean"}.issubset(feat_copy):
            feat_copy["ratio_GB"] = float(feat_copy["G_mean"] / max(feat_copy["B_mean"], 1e-4))

        if "ratio_RB" not in feat_copy and {"R_mean", "B_mean"}.issubset(feat_copy):
            feat_copy["ratio_RB"] = float(feat_copy["R_mean"] / max(feat_copy["B_mean"], 1e-4))

        # Clamp exposure_time_minutes to training distribution range
        if "exposure_time_minutes" in feat_copy:
            feat_copy["exposure_time_minutes"] = float(np.clip(feat_copy["exposure_time_minutes"], 30.0, 480.0))

        # Check required features
        vector = []
        for f in self.feature_names:
            if f not in feat_copy:
                raise ValueError(f"Feature '{f}' missing from inference input dictionary.")
            vector.append(float(feat_copy[f]))
            
        X = np.array([vector], dtype=np.float32)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        raw_pred = float(self.model.predict(X_scaled)[0])
        pred_dose = max(0.0, raw_pred)
        
        # Confidence interval (+/- 1.96 sigma for ~95% confidence bounds)
        ci_margin = 1.96 * self.uncertainty_sigma
        ci_low = max(0.0, pred_dose - ci_margin)
        ci_high = pred_dose + ci_margin
        
        # Derived 8-hour equivalent TWA concentration
        twa_8h = pred_dose / 8.0
        
        # Classify risk
        risk_level, risk_color, recommendation = self._classify_risk(pred_dose)
        
        return DosePredictionResult(
            cumulative_dose_ppm_h=round(pred_dose, 3),
            confidence_interval_low=round(ci_low, 3),
            confidence_interval_high=round(ci_high, 3),
            uncertainty_margin=round(ci_margin, 3),
            twa_8h_equivalent_ppm=round(twa_8h, 3),
            risk_level=risk_level,
            risk_color=risk_color,
            action_recommendation=recommendation,
            model_type=self.model_type,
            is_synthetic_model=self.is_synthetic,
            temperature_c=float(feat_copy.get("temperature_c", physics_config.T_REF_CELSIUS)),
            humidity_percent=float(feat_copy.get("humidity_percent", physics_config.RH_REF_PERCENT)),
            is_badge_valid=True,
            expiry_status="VALID"
        )

    def predict_from_image(
        self,
        image_input: Union[str, Path, np.ndarray],
        temp_c: float = 25.0,
        humidity_pct: float = 50.0,
        exposure_time_min: float = 240.0,
        qr_payload: Optional[Union[str, Dict[str, Any]]] = None
    ) -> Tuple[DosePredictionResult, Dict[str, Any]]:
        """
        End-to-End inference directly from a badge photograph.
        Runs CV ROI detection, illumination normalization, expiry check, and ML prediction.
        """
        import cv2
        from src.preprocessing.color_features import process_badge_image_pipeline
        
        if isinstance(image_input, (str, Path)):
            image_path = Path(image_input)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found at: {image_path}")
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                raise ValueError(f"Could not decode image at: {image_path}")
        else:
            image_bgr = image_input

        # 1. Run Complete Computer Vision Pipeline
        pipeline_result = process_badge_image_pipeline(
            image_bgr,
            temp_c=temp_c,
            humidity_pct=humidity_pct,
            exposure_time_min=exposure_time_min,
            qr_payload=qr_payload
        )
        
        features = pipeline_result["color_features"]
        expiry = pipeline_result["expiry_report"]
        lighting_qa = pipeline_result["lighting_qa"]
        capture_quality = pipeline_result["capture_quality"]

        if not capture_quality["is_acceptable"]:
            raise ValueError(f"Retake required: {capture_quality['retake_reason']}")

        # 2. Run ML Prediction
        res = self.predict_single(features)
        
        # 3. Update with CV specifics
        res.is_badge_valid = expiry["is_valid"]
        res.expiry_status = expiry["expiry_status"]
        res.lighting_profile = lighting_qa["lighting_profile"]
        
        if not expiry["is_valid"]:
            res.risk_level = "INVALID / BADGE EXPIRED"
            res.risk_color = "red"
            res.action_recommendation = (
                f"BADGE INVALID: {expiry['recommendation']} "
                f"Estimated dose ({res.cumulative_dose_ppm_h:.1f} ppm*h) cannot be certified."
            )
            
        res.cv_diagnostics = {
            "is_auto_aligned": pipeline_result["is_auto_aligned"],
            "alignment_confidence": pipeline_result["alignment_confidence"],
            "capture_quality": capture_quality,
            "lighting_qa": lighting_qa,
            "normalization": pipeline_result["normalization_diagnostics"],
            "expiry_report": expiry,
            "debug_overlay": pipeline_result["debug_overlay"]
        }

        return res, pipeline_result
