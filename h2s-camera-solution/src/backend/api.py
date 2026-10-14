"""
FastAPI REST API Service for Digital H2S Passive Dosimeter.
Serves mobile and web clients with badge image scanning, OpenCV processing,
ML cumulative dose prediction, shift exposure logging, and compliance audits.
"""

import os
import cv2
import base64
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from src.utils.config import (
    MODELS_DIR, IMAGES_DIR, RESULTS_DIR, physics_config, ensure_directories
)
from src.utils.logger import get_logger
from src.utils.badge_generator import generate_test_badge_image
from src.backend.database import init_db, get_db, Worker, ExposureRecord
import json
import urllib.request
import urllib.error
from src.backend.schemas import (
    WorkerResponse, ExposureRecordResponse, ScanResponse,
    WorkerSummaryResponse, ComplianceReportResponse, GenerateBadgeRequest,
    ValidateQRRequest, ValidateQRResponse,
    CameraGuideRequest, CameraGuideResponse, DetectZonesResponse,
    HardwareReadingRequest, HardwareReadingResponse,
    HardwareStationStatus, HardwareLedCommandRequest
)
from src.backend.compliance import generate_oisd_compliance_report
from src.ml.predict import DosimeterPredictor
from src.ml.as7343_model import AS7343Predictor
from src.utils.qr_validator import process_badge_qr, decode_qr_from_image
from src.preprocessing.camera_guide import analyze_live_frame
from src.preprocessing.zone_detector import BadgeZoneDetector, draw_zone_detections

logger = get_logger("FastAPI_Server")
ensure_directories()

# Initialize FastAPI App
app = FastAPI(
    title="Digital Wearable H2S Passive Dosimeter API",
    description="REST API for Camera-based Optical Dosimetry, ML Cumulative Exposure Prediction & DGMS/OISD Auditing",
    version="1.0.0"
)

# Enable CORS for universal access (Web browsers, React Native mobile, Expo Go)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount debug images directory for overlay visualization
debug_dir = IMAGES_DIR / "debug"
debug_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/debug", StaticFiles(directory=str(debug_dir)), name="debug_images")

# Global predictor instance
predictor_instance: Optional[DosimeterPredictor] = None


def get_predictor() -> DosimeterPredictor:
    """Get or lazily instantiate ML model predictor."""
    global predictor_instance
    if predictor_instance is None:
        init_db()
        predictor_instance = DosimeterPredictor()
    return predictor_instance


@app.get("/", tags=["Health"])
def root_endpoint():
    return {
        "system": "Digital Wearable H2S Passive Dosimeter API",
        "status": "ONLINE",
        "version": "1.0.0",
        "docs_url": "/docs",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/health", tags=["Health"])
def health_endpoint():
    try:
        pred = get_predictor()
        is_model_ready = pred is not None
        model_type = pred.model_type
        is_synth = pred.is_synthetic
        sigma = round(pred.uncertainty_sigma, 3)
    except Exception:
        is_model_ready = False
        model_type = "None"
        is_synth = True
        sigma = None

    return {
        "status": "HEALTHY",
        "model_loaded": is_model_ready,
        "model_type": model_type,
        "is_synthetic_model": is_synth,
        "uncertainty_sigma": sigma,
        "server_time": datetime.utcnow().isoformat(),
        "calibration_standards": {
            "twa_pel_limit_ppm": 10.0,
            "warning_threshold_ppm_h": physics_config.DOSE_ALERT_WARNING,
            "danger_threshold_ppm_h": physics_config.DOSE_ALERT_DANGER
        }
    }


@app.post("/api/validate-qr", response_model=ValidateQRResponse, tags=["Badge Identity"])
def validate_qr_endpoint(req: ValidateQRRequest):
    """
    Validate a QR payload string decoded from a physical badge.
    Returns badge identity, expiry status, and calibration version.
    Call this BEFORE /api/scan to gate the workflow on badge validity.
    """
    result = process_badge_qr(req.qr_payload, reference_date=req.reference_date)
    return ValidateQRResponse(**result)


@app.post("/api/scan", response_model=ScanResponse, tags=["Dosimetry & Scanning"])
async def scan_badge_endpoint(
    file: Optional[UploadFile] = File(None, description="Badge photograph (JPEG/PNG)"),
    image_base64: Optional[str] = Form(None, description="Raw base64 image string (mobile fallback)"),
    qr_payload: Optional[str] = Form(None, description="Raw QR code payload string or JSON"),
    worker_id: str = Form(default="WRK-1001", description="Worker ID"),
    badge_id: Optional[str] = Form(default=None, description="Badge Serial/Batch ID"),
    batch_id: Optional[str] = Form(default=None, description="Badge batch from QR"),
    calibration_version: Optional[str] = Form(default="CAL-03", description="Calibration version from QR"),
    badge_expiry_date: Optional[str] = Form(default=None, description="Expiry date from QR (YYYY-MM-DD)"),
    reference_date: Optional[str] = Form(default=None, description="Optional reference date for expiry evaluation"),
    temperature_c: float = Form(default=25.0, description="Ambient Temperature in °C"),
    humidity_percent: float = Form(default=50.0, description="Relative Humidity in %"),
    exposure_time_minutes: float = Form(default=240.0, description="Shift duration in minutes"),
    bypass_expiry: bool = Form(default=False, description="Override expiry check (for xerox/demo badges)"),
    db: Session = Depends(get_db)
):
    """
    Main Mobile Scanning Endpoint:
    1. Validates digital QR badge identity and declared expiry date.
    2. If expired, halts and safely rejects dose estimation.
    3. Runs Computer Vision perspective alignment and reference color balancing.
    4. Feeds extracted multi-space features to the XGBoost model.
    5. Computes estimated cumulative dose, 95% CI, 8-hr equivalent TWA, and DGMS/OISD risk status.
    6. Saves immutable exposure record with badge ID and calibration version in database.
    """
    global predictor_instance
    if predictor_instance is None:
        try:
            predictor_instance = DosimeterPredictor()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"ML Model is not loaded. Train a model first. ({e})"
            )

    # 1. Decode image — from file upload or base64 string (mobile fallback)
    if image_base64:
        # Handle data URL prefix if present
        clean_b64 = image_base64.split(",")[-1] if "," in image_base64 else image_base64
        img_bytes = base64.b64decode(clean_b64)
    elif file:
        img_bytes = await file.read()
    else:
        raise HTTPException(
             status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either a file upload or image_base64 field."
        )
    nparr = np.frombuffer(img_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format. Please upload a valid JPEG or PNG photograph."
        )

    # 2. Extract and Validate QR Badge Identity / Expiry
    # If QR payload was not explicitly supplied as form field, try decoding from the image
    qr_data_to_validate = qr_payload
    if not qr_data_to_validate:
        decoded_from_img = decode_qr_from_image(image_bgr)
        if decoded_from_img:
            qr_data_to_validate = decoded_from_img

    eff_badge_id = badge_id or "H2S-B17-00482"
    eff_batch_id = batch_id or "B17"
    eff_cal_ver = calibration_version or "CAL-03"
    eff_exp_date = badge_expiry_date
    qr_validation_result = None

    if qr_data_to_validate:
        qr_val = process_badge_qr(qr_data_to_validate, reference_date=reference_date)
        qr_validation_result = qr_val
        if qr_val["badge_id"]:
            eff_badge_id = qr_val["badge_id"]
        if qr_val["batch_id"]:
            eff_batch_id = qr_val["batch_id"]
        if qr_val["calibration_version"]:
            eff_cal_ver = qr_val["calibration_version"]
        if qr_val["expiry_date"]:
            eff_exp_date = qr_val["expiry_date"]
            
        # Reject malformed QR format if not bypassed
        if qr_val["badge_status"] == "INVALID_FORMAT" and not bypass_expiry:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=qr_val["message"]
            )
            
        # Halt and block expired badge if not bypassed
        if not qr_val["is_valid"] and qr_val["badge_status"] == "EXPIRED" and not bypass_expiry:
            logger.warning(f"Scan rejected: Badge {eff_badge_id} is expired ({eff_exp_date}).")
            
            # Ensure worker exists in database
            worker = db.query(Worker).filter(Worker.id == worker_id).first()
            if not worker:
                worker = Worker(id=worker_id, name=f"Worker {worker_id}", department="Operations", designation="Field Operator")
                db.add(worker)
                db.commit()

            # Record attempted scan as rejected/invalid
            record = ExposureRecord(
                worker_id=worker_id,
                badge_id=eff_badge_id,
                batch_id=eff_batch_id,
                calibration_version=eff_cal_ver,
                image_filename=None,
                cumulative_dose_ppm_h=0.0,
                confidence_interval_low=0.0,
                confidence_interval_high=0.0,
                uncertainty_margin=0.0,
                twa_8h_ppm=0.0,
                risk_level="INVALID / BADGE EXPIRED",
                action_recommendation="BADGE INVALID: Badge expired. Please use a new badge.",
                temperature_c=temperature_c,
                humidity_percent=humidity_percent,
                exposure_duration_minutes=exposure_time_minutes,
                is_badge_valid=False,
                expiry_status="EXPIRED",
                badge_status="EXPIRED",
                reliability_score=0.0,
                lighting_profile="UNKNOWN",
                model_type="NONE",
                model_version="v1.0-xgboost"
            )
            db.add(record)
            db.commit()
            db.refresh(record)

            return ScanResponse(
                success=False,
                record_id=record.id,
                worker_id=worker_id,
                badge_id=eff_badge_id,
                batch_id=eff_batch_id,
                calibration_version=eff_cal_ver,
                badge_status="EXPIRED",
                expiry_date=eff_exp_date,
                cumulative_dose_ppm_h=0.0,
                dose_ppm_h=0.0,
                confidence_interval_low=0.0,
                confidence_interval_high=0.0,
                uncertainty_margin=0.0,
                twa_8h_equivalent_ppm=0.0,
                twa_ppm=0.0,
                risk_level="INVALID / BADGE EXPIRED",
                risk="EXPIRED",
                risk_color="red",
                action_recommendation="BADGE INVALID: Badge expired. Please use a new badge.",
                model_type="NONE",
                model_version="v1.0-xgboost",
                is_synthetic_model=True,
                temperature_c=temperature_c,
                humidity_percent=humidity_percent,
                exposure_duration_minutes=exposure_time_minutes,
                is_badge_valid=False,
                expiry_status="EXPIRED",
                reliability_score=0.0,
                image_quality="REJECTED_EXPIRED",
                lighting_profile="UNKNOWN",
                overlay_image_url=None,
                color_features=None
            )
    elif badge_expiry_date and not bypass_expiry:
        # Validate passed expiry date
        val_res = process_badge_qr({
            "badge_id": eff_badge_id,
            "batch_id": eff_batch_id,
            "expiry_date": badge_expiry_date,
            "calibration_version": eff_cal_ver
        }, reference_date=reference_date)
        if not val_res["is_valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=val_res["message"]
            )

    # 3. Run End-to-End Prediction via OpenCV & XGBoost
    try:
        result, pipe_res = predictor_instance.predict_from_image(
            image_bgr,
            temp_c=temperature_c,
            humidity_pct=humidity_percent,
            exposure_time_min=exposure_time_minutes,
            qr_payload=qr_data_to_validate
        )
        # Override expiry flag for xerox/demo/printout badges
        if bypass_expiry and not result.is_badge_valid:
            result.is_badge_valid = True
            result.expiry_status = "VALID (OVERRIDE)"
            result.risk_level, result.risk_color, result.action_recommendation = (
                predictor_instance._classify_risk(result.cumulative_dose_ppm_h)
            )
    except ValueError as e:
        logger.warning(f"Rejected badge image: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process badge image: {str(e)}"
        )

    # 4. Save Annotated Debug Overlay Image
    timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    overlay_filename = f"scan_{worker_id}_{timestamp_str}.png"
    overlay_path = debug_dir / overlay_filename
    cv2.imwrite(str(overlay_path), pipe_res["debug_overlay"])
    overlay_url = f"/static/debug/{overlay_filename}"

    # 5. Save Record in Database
    worker = db.query(Worker).filter(Worker.id == worker_id).first()
    if not worker:
        worker = Worker(id=worker_id, name=f"Worker {worker_id}", department="Operations", designation="Field Operator")
        db.add(worker)
        db.commit()

    rel_score = 0.96 if pipe_res.get("capture_quality", {}).get("is_acceptable", True) else 0.70

    record = ExposureRecord(
        worker_id=worker_id,
        badge_id=eff_badge_id,
        batch_id=eff_batch_id,
        calibration_version=eff_cal_ver,
        image_filename=overlay_filename,
        cumulative_dose_ppm_h=result.cumulative_dose_ppm_h,
        confidence_interval_low=result.confidence_interval_low,
        confidence_interval_high=result.confidence_interval_high,
        uncertainty_margin=result.uncertainty_margin,
        twa_8h_ppm=result.twa_8h_equivalent_ppm,
        risk_level=result.risk_level,
        action_recommendation=result.action_recommendation,
        temperature_c=temperature_c,
        humidity_percent=humidity_percent,
        exposure_duration_minutes=exposure_time_minutes,
        is_badge_valid=result.is_badge_valid,
        expiry_status=result.expiry_status,
        badge_status=result.expiry_status,
        reliability_score=rel_score,
        lighting_profile=result.lighting_profile,
        model_type=result.model_type,
        model_version="v1.0-xgboost"
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return ScanResponse(
        success=True,
        record_id=record.id,
        worker_id=worker_id,
        badge_id=eff_badge_id,
        batch_id=eff_batch_id,
        calibration_version=eff_cal_ver,
        badge_status=result.expiry_status,
        expiry_date=eff_exp_date,
        cumulative_dose_ppm_h=result.cumulative_dose_ppm_h,
        dose_ppm_h=result.cumulative_dose_ppm_h,
        confidence_interval_low=result.confidence_interval_low,
        confidence_interval_high=result.confidence_interval_high,
        uncertainty_margin=result.uncertainty_margin,
        twa_8h_equivalent_ppm=result.twa_8h_equivalent_ppm,
        twa_ppm=result.twa_8h_equivalent_ppm,
        risk_level=result.risk_level,
        risk=result.risk_level.split("/")[0].strip(),
        risk_color=result.risk_color,
        action_recommendation=result.action_recommendation,
        model_type=result.model_type,
        model_version="v1.0-xgboost",
        is_synthetic_model=result.is_synthetic_model,
        temperature_c=result.temperature_c,
        humidity_percent=result.humidity_percent,
        exposure_duration_minutes=exposure_time_minutes,
        is_badge_valid=result.is_badge_valid,
        expiry_status=result.expiry_status,
        reliability_score=rel_score,
        image_quality="HIGH" if pipe_res.get("is_auto_aligned") else "MEDIUM",
        lighting_profile=result.lighting_profile,
        overlay_image_url=overlay_url,
        color_features={k: float(v) for k, v in pipe_res["color_features"].items() if isinstance(v, (int, float))}
    )


@app.get("/api/records", response_model=List[ExposureRecordResponse], tags=["Records & History"])
def get_records_endpoint(
    worker_id: Optional[str] = Query(None, description="Filter by Worker ID"),
    limit: int = Query(50, ge=1, le=500),
    skip: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """Retrieve historical shift exposure records with optional worker filter."""
    query = db.query(ExposureRecord)
    if worker_id:
        query = query.filter(ExposureRecord.worker_id == worker_id)
    records = query.order_by(ExposureRecord.timestamp.desc()).offset(skip).limit(limit).all()
    return records


@app.get("/api/workers", response_model=List[WorkerResponse], tags=["Workers"])
def get_workers_endpoint(db: Session = Depends(get_db)):
    """List all registered industrial workers."""
    return db.query(Worker).filter(Worker.active == True).all()


@app.get("/api/workers/{worker_id}", response_model=WorkerSummaryResponse, tags=["Workers"])
def get_worker_summary_endpoint(worker_id: str, db: Session = Depends(get_db)):
    """Get single worker profile, cumulative 7-day exposure dose, and recent shift logs."""
    worker = db.query(Worker).filter(Worker.id == worker_id).first()
    if not worker:
        raise HTTPException(status_code=404, detail=f"Worker '{worker_id}' not found.")

    recent_records = (
        db.query(ExposureRecord)
        .filter(ExposureRecord.worker_id == worker_id)
        .order_by(ExposureRecord.timestamp.desc())
        .limit(20)
        .all()
    )

    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    records_7d = [r for r in recent_records if r.timestamp >= seven_days_ago]

    cum_7d = sum(r.cumulative_dose_ppm_h for r in records_7d)
    recent_dose = recent_records[0].cumulative_dose_ppm_h if recent_records else 0.0
    avg_twa = (sum(r.twa_8h_ppm for r in records_7d) / len(records_7d)) if records_7d else 0.0

    if not recent_records or recent_records[0].risk_level == "SAFE / NORMAL":
        status_text = "NORMAL / SAFE"
        status_color = "green"
    elif "WARNING" in recent_records[0].risk_level:
        status_text = "ELEVATED EXPOSURE"
        status_color = "yellow"
    else:
        status_text = "ACTION REQUIRED"
        status_color = "red"

    return WorkerSummaryResponse(
        worker=worker,
        total_scans=len(recent_records),
        recent_dose_ppm_h=round(recent_dose, 2),
        cumulative_7day_dose_ppm_h=round(cum_7d, 2),
        average_twa_ppm=round(avg_twa, 2),
        overall_status=status_text,
        status_color=status_color,
        recent_records=recent_records
    )


@app.get("/api/report/compliance", response_model=ComplianceReportResponse, tags=["Compliance"])
def get_compliance_report_endpoint(db: Session = Depends(get_db)):
    """Generate DGMS / OISD standardized occupational health compliance report."""
    return generate_oisd_compliance_report(db)


@app.post("/api/generate-badge", tags=["Simulation & Testing"])
def generate_badge_endpoint(req: GenerateBadgeRequest):
    """
    On-the-fly Badge Generator API.
    Renders a realistic badge image with a QR code and returns base64 string for immediate mobile testing.
    """
    img, saved_path = generate_test_badge_image(
        dose_ppm_h=req.exposure,
        temp_c=req.temp,
        humidity_pct=req.humidity,
        lighting=req.lighting,
        tilt_angle=req.tilt,
        is_expired=req.expired,
        badge_id=req.badge_id or "H2S-B17-00482",
        batch_id=req.batch_id or "B17",
        manufactured_date=req.manufactured_date,
        expiry_date=req.expiry_date,
        calibration_version=req.calibration_version or "CAL-03"
    )
    
    # Encode to base64
    _, buffer = cv2.imencode(".png", img)
    b64_str = base64.b64encode(buffer).decode("utf-8")
    
    # Decode or build QR payload for easy mobile sandbox consumption
    from src.utils.badge_generator import build_badge_qr_payload
    qr_payload = build_badge_qr_payload(
        badge_id=req.badge_id or "H2S-B17-00482",
        batch_id=req.batch_id or "B17",
        manufactured_date=req.manufactured_date,
        expiry_date=req.expiry_date,
        calibration_version=req.calibration_version or "CAL-03",
        is_expired=req.expired
    )
    
    return {
        "success": True,
        "exposure_ppm_h": req.exposure,
        "lighting": req.lighting,
        "is_expired": req.expired,
        "badge_id": qr_payload["badge_id"],
        "batch_id": qr_payload["batch_id"],
        "manufactured_date": qr_payload["manufactured_date"],
        "expiry_date": qr_payload["expiry_date"],
        "calibration_version": qr_payload["calibration_version"],
        "qr_payload": qr_payload,
        "image_base64": f"data:image/png;base64,{b64_str}",
        "width": img.shape[1],
        "height": img.shape[0]
    }


# Dedicated real-time zone detector instance for API
api_zone_detector = BadgeZoneDetector(use_temporal_smoothing=True, smoothing_factor=0.45)


@app.post("/api/camera/analyze-frame", response_model=CameraGuideResponse, tags=["Live Camera Guide"])
def analyze_camera_frame_endpoint(req: CameraGuideRequest):
    """
    Real-time Live Camera Guidance & Dynamic Zone Detection API.
    Analyzes live camera stream frame for dynamic corner tracking, 6-zone projection,
    positioning quality, and auto-capture readiness.
    """
    try:
        raw_b64 = req.image_base64
        if "base64," in raw_b64:
            raw_b64 = raw_b64.split("base64,")[1]
        img_bytes = base64.b64decode(raw_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image frame")
        
        result = analyze_live_frame(img)
        return CameraGuideResponse(**result)
    except Exception as e:
        logger.warning(f"Error analyzing live camera frame: {e}")
        return CameraGuideResponse(
            status="SEARCHING",
            message="Looking for dosimeter badge...",
            color="#38bdf8",
            can_autocapture=False,
            badge_detected=False,
            corners=None,
            coverage_pct=0.0,
            center_offset_pct=0.0,
            blur_score=0.0,
            brightness=0.0,
            rotation_deg=0.0,
            zones=[]
        )


@app.post("/api/camera/detect-zones", response_model=DetectZonesResponse, tags=["Live Camera Guide"])
def detect_zones_endpoint(req: CameraGuideRequest):
    """
    Lightweight, ultra-low-latency endpoint for live streaming zone detection.
    Returns dynamic bounding boxes and polygons for:
    - 🟩 Sensing Zone
    - 🔵 4 Reference Calibration Zones
    - 🟧 QR Badge Zone
    """
    try:
        raw_b64 = req.image_base64
        if "base64," in raw_b64:
            raw_b64 = raw_b64.split("base64,")[1]
        img_bytes = base64.b64decode(raw_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image frame")

        result = api_zone_detector.detect_zones(img)
        zones_data = [z.to_dict() if hasattr(z, "to_dict") else z for z in result.get("zones", [])]

        return DetectZonesResponse(
            badge_detected=result.get("badge_detected", False),
            confidence=result.get("confidence", 0.0),
            corners=result.get("corners", None),
            rotation_deg=result.get("rotation_deg", 0.0),
            zones=zones_data
        )
    except Exception as e:
        logger.warning(f"Error detecting badge zones: {e}")
        return DetectZonesResponse(
            badge_detected=False,
            confidence=0.0,
            corners=None,
            rotation_deg=0.0,
            zones=[]
        )


# =====================================================================
#             AS7343 HARDWARE READER INTEGRATION ENDPOINTS
# =====================================================================

@app.post("/api/hardware/reading", response_model=HardwareReadingResponse, tags=["Hardware Reader Station"])
def process_hardware_reading_endpoint(
    req: HardwareReadingRequest,
    db: Session = Depends(get_db)
):
    """
    Ingest and process multispectral reading from the AS7343 + BME688 Hardware Reader.
    1. Computes 12 channel/Clear ratios and evaluates the 50-tree Random Forest model.
    2. Determines 8-h TWA, risk tier, and action recommendation.
    3. Persists complete spectral signature, environmental data, and saturation flag to database.
    """
    try:
        # Run AS7343 Ratio Model Prediction
        pred = AS7343Predictor.predict(
            channels=req.channels,
            clear=req.clear_count,
            exposure_duration_hours=req.exposure_duration_minutes / 60.0,
            is_saturated=req.is_saturated
        )

        # Check if worker exists
        worker = db.query(Worker).filter(Worker.id == req.worker_id).first()
        if not worker:
            worker = Worker(
                id=req.worker_id,
                name=f"Worker {req.worker_id}",
                department="Operations",
                designation="Field Operator"
            )
            db.add(worker)
            db.commit()

        # Build spectral data JSON payload for auditing
        spectral_json = json.dumps({
            "raw_channels": pred.raw_channels,
            "ratios": pred.ratios,
            "clear_count": pred.clear_count,
            "is_saturated": req.is_saturated,
            "led_current_ma": req.led_current_ma
        })

        # Save to database
        record = ExposureRecord(
            worker_id=req.worker_id,
            badge_id=req.badge_id or "H2S-HW-001",
            batch_id=req.batch_id or "HW-01",
            calibration_version=req.calibration_version or "AS7343-CAL-01",
            image_filename=None,
            cumulative_dose_ppm_h=pred.cumulative_dose_ppm_h,
            confidence_interval_low=max(0.0, pred.cumulative_dose_ppm_h - 2.3),
            confidence_interval_high=pred.cumulative_dose_ppm_h + 2.3,
            uncertainty_margin=2.3,  # Leave-one-out validation MAE from model header
            twa_8h_ppm=pred.twa_8h_equivalent_ppm,
            risk_level=pred.risk_level,
            action_recommendation=pred.action_recommendation,
            temperature_c=req.temperature_c,
            humidity_percent=req.humidity_percent,
            exposure_duration_minutes=req.exposure_duration_minutes,
            is_badge_valid=not req.is_saturated,
            expiry_status="VALID" if not req.is_saturated else "SATURATED_REDUCE_LED",
            badge_status="VALID" if not req.is_saturated else "SATURATED",
            reliability_score=0.98 if not req.is_saturated else 0.50,
            lighting_profile="OPTICAL_AS7343_CHAMBER",
            model_type=pred.model_type,
            model_version=pred.model_version,
            measurement_source="SPECTRAL_HARDWARE_AS7343",
            spectral_data=spectral_json,
            gas_resistance_ohms=req.gas_resistance_ohms,
            pressure_hpa=req.pressure_hpa,
            is_saturated=req.is_saturated,
            led_current_ma=req.led_current_ma
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        return HardwareReadingResponse(
            success=True,
            record_id=record.id,
            worker_id=req.worker_id,
            badge_id=req.badge_id or "H2S-HW-001",
            measurement_source="SPECTRAL_HARDWARE_AS7343",
            cumulative_dose_ppm_h=pred.cumulative_dose_ppm_h,
            dose_ppm_h=pred.cumulative_dose_ppm_h,
            twa_8h_equivalent_ppm=pred.twa_8h_equivalent_ppm,
            twa_ppm=pred.twa_8h_equivalent_ppm,
            risk_level=pred.risk_level,
            risk_tier=pred.risk_tier,
            risk_color=pred.risk_color,
            action_recommendation=pred.action_recommendation,
            temperature_c=req.temperature_c,
            humidity_percent=req.humidity_percent,
            pressure_hpa=req.pressure_hpa,
            gas_resistance_ohms=req.gas_resistance_ohms,
            exposure_duration_minutes=req.exposure_duration_minutes,
            is_saturated=req.is_saturated,
            led_current_ma=req.led_current_ma,
            clear_count=pred.clear_count,
            ratios=pred.ratios,
            raw_channels=pred.raw_channels,
            model_type=pred.model_type,
            model_version=pred.model_version,
            timestamp=record.timestamp
        )
    except ValueError as e:
        logger.warning(f"Invalid hardware reading payload: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error processing hardware reading: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process hardware reading: {str(e)}"
        )


@app.get("/api/hardware/proxy/read", tags=["Hardware Reader Station"])
def proxy_hardware_read(station_ip: str = Query(..., description="IP Address of ESP32 Hardware Station")):
    """
    Proxy request to ESP32 Station /api/read to bypass CORS/NAT restrictions in web browser or mobile client.
    """
    clean_ip = station_ip.replace("http://", "").replace("https://", "").strip("/")
    target_url = f"http://{clean_ip}/api/read"
    try:
        req = urllib.request.Request(target_url, headers={"User-Agent": "H2S-Backend-Proxy/1.0"})
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return {
                "station_ip": clean_ip,
                "connected": True,
                **data
            }
    except Exception as e:
        return {
            "station_ip": clean_ip,
            "connected": False,
            "error": str(e)
        }


@app.get("/api/hardware/proxy/scan", tags=["Hardware Reader Station"])
def proxy_hardware_scan(station_ip: str = Query(..., description="IP Address of ESP32 Hardware Station")):
    """
    Trigger scan on ESP32 Station (/api/scan) via proxy.
    """
    clean_ip = station_ip.replace("http://", "").replace("https://", "").strip("/")
    target_url = f"http://{clean_ip}/api/scan"
    try:
        req = urllib.request.Request(target_url, headers={"User-Agent": "H2S-Backend-Proxy/1.0"})
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return {
                "station_ip": clean_ip,
                "success": True,
                **data
            }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not reach hardware station at {clean_ip}: {str(e)}"
        )


@app.post("/api/hardware/proxy/command", tags=["Hardware Reader Station"])
def proxy_hardware_command(
    command: str = Query(..., description="activate | led_toggle | set_current"),
    station_ip: str = Query(..., description="IP Address of ESP32 Hardware Station"),
    state: Optional[int] = Query(None, description="1 for LED ON, 0 for LED OFF"),
    ma: Optional[int] = Query(None, description="Drive current in mA (4 to 258)")
):
    """
    Send hardware controls (Badge activate, LED on/off, LED current adjustment) to ESP32 station.
    """
    clean_ip = station_ip.replace("http://", "").replace("https://", "").strip("/")
    
    if command == "activate":
        target_url = f"http://{clean_ip}/api/activate"
    elif command == "led_toggle":
        st = state if state is not None else 1
        target_url = f"http://{clean_ip}/api/led?state={st}"
    elif command == "set_current":
        current_val = ma if ma is not None else 25
        target_url = f"http://{clean_ip}/api/led/current?ma={current_val}"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown command '{command}'")

    try:
        req = urllib.request.Request(target_url, headers={"User-Agent": "H2S-Backend-Proxy/1.0"})
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return {
                "station_ip": clean_ip,
                "success": True,
                **data
            }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to communicate with station at {clean_ip}: {str(e)}"
        )



