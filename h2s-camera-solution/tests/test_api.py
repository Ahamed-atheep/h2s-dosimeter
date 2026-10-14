"""
Automated Unit and Integration Tests for FastAPI REST API Backend.
Tests all endpoints, image scanning, database persistence, and compliance reports.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import cv2

from src.backend.api import app
from src.backend.database import init_db, SessionLocal, Worker, ExposureRecord
from src.utils.badge_generator import generate_test_badge_image

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    """Ensure database tables and demo workers are initialized."""
    init_db()


def test_root_endpoint():
    """Test health and root status endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ONLINE"
    assert "version" in data


def test_health_endpoint():
    """Test system health and ML model metadata."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["model_loaded"] is True
    assert data["model_type"] in ["xgboost", "random_forest", "ridge"]


def test_generate_badge_api():
    """Test on-the-fly badge generation endpoint."""
    payload = {
        "exposure": 15.0,
        "temp": 30.0,
        "humidity": 60.0,
        "lighting": "daylight",
        "tilt": 0.0,
        "expired": False
    }
    response = client.post("/api/generate-badge", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "image_base64" in data
    assert data["image_base64"].startswith("data:image/png;base64,")


def test_scan_badge_upload_valid(tmp_path):
    """Test uploading a valid badge photograph to /api/scan."""
    badge_path = tmp_path / "scan_test_valid.png"
    generate_test_badge_image(dose_ppm_h=12.0, temp_c=25.0, humidity_pct=50.0, is_expired=False, output_path=badge_path)
    
    with open(badge_path, "rb") as f:
        response = client.post(
            "/api/scan",
            files={"file": ("badge.png", f, "image/png")},
            data={
                "worker_id": "WRK-1001",
                "badge_id": "BADGE-TEST-01",
                "temperature_c": 25.0,
                "humidity_percent": 50.0,
                "exposure_time_minutes": 240.0
            }
        )
        
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["worker_id"] == "WRK-1001"
    assert data["is_badge_valid"] is True
    assert data["expiry_status"] == "VALID"
    assert data["cumulative_dose_ppm_h"] > 0.0
    assert data["confidence_interval_low"] <= data["cumulative_dose_ppm_h"] <= data["confidence_interval_high"]
    assert data["overlay_image_url"] is not None
    assert data["record_id"] is not None


def test_scan_badge_upload_expired(tmp_path):
    """Test uploading an expired badge to /api/scan and assert safety block."""
    badge_path = tmp_path / "scan_test_expired.png"
    generate_test_badge_image(dose_ppm_h=8.0, is_expired=True, output_path=badge_path)
    
    with open(badge_path, "rb") as f:
        response = client.post(
            "/api/scan",
            files={"file": ("expired.png", f, "image/png")},
            data={
                "worker_id": "WRK-1002",
                "temperature_c": 25.0,
                "humidity_percent": 50.0
            }
        )
        
    assert response.status_code == 200
    data = response.json()
    assert data["is_badge_valid"] is False
    assert data["expiry_status"] == "EXPIRED"
    assert "BADGE INVALID" in data["action_recommendation"]


def test_get_records_and_worker_summary():
    """Test retrieving shift history and worker summaries."""
    # 1. Get records
    resp_records = client.get("/api/records?limit=10")
    assert resp_records.status_code == 200
    records = resp_records.json()
    assert isinstance(records, list)
    assert len(records) > 0

    # 2. Get worker summary
    resp_worker = client.get("/api/workers/WRK-1001")
    assert resp_worker.status_code == 200
    worker_summary = resp_worker.json()
    assert worker_summary["worker"]["id"] == "WRK-1001"
    assert worker_summary["total_scans"] > 0


def test_compliance_report_endpoint():
    """Test DGMS / OISD occupational health compliance report generation."""
    response = client.get("/api/report/compliance")
    assert response.status_code == 200
    data = response.json()
    assert "OISD-STD-155" in data["standard_reference"]
    assert data["total_shifts_logged"] > 0
    assert 0.0 <= data["compliance_rate_percent"] <= 100.0
    assert len(data["records"]) > 0


# ==============================================================================
# QR Identity & Expiry Test Suite (SIH Requirements)
# ==============================================================================

def test_validate_qr_valid():
    """1. Valid QR payload -> validation succeeds with VALID status and metadata."""
    import json
    payload = {
        "badge_id": "H2S-B17-00482",
        "batch_id": "B17",
        "manufactured_date": "2026-09-01",
        "expiry_date": "2026-12-06",
        "calibration_version": "CAL-03"
    }
    resp = client.post("/api/validate-qr", json={"qr_payload": json.dumps(payload)})
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_valid"] is True
    assert data["badge_status"] == "VALID"
    assert data["badge_id"] == "H2S-B17-00482"
    assert data["batch_id"] == "B17"
    assert data["calibration_version"] == "CAL-03"
    assert "verified" in data["message"].lower()


def test_validate_qr_expired():
    """2. Expired QR payload -> validation rejected with EXPIRED status."""
    import json
    payload = {
        "badge_id": "H2S-B17-00482",
        "batch_id": "B17",
        "manufactured_date": "2026-05-01",
        "expiry_date": "2026-08-01",
        "calibration_version": "CAL-03"
    }
    resp = client.post("/api/validate-qr", json={"qr_payload": json.dumps(payload)})
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_valid"] is False
    assert data["badge_status"] == "EXPIRED"
    assert "expired" in data["message"].lower()
    assert "new badge" in data["message"].lower()


def test_validate_qr_malformed():
    """3. Malformed non-JSON QR payload -> rejected with INVALID_FORMAT."""
    resp = client.post("/api/validate-qr", json={"qr_payload": "NOT_A_VALID_JSON_STRING_12345"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_valid"] is False
    assert data["badge_status"] == "INVALID_FORMAT"
    assert "invalid json" in data["message"].lower()


def test_validate_qr_missing_expiry():
    """4. Missing expiry date in QR payload -> rejected."""
    import json
    payload = {
        "badge_id": "H2S-B17-00482",
        "batch_id": "B17"
        # missing expiry_date
    }
    resp = client.post("/api/validate-qr", json={"qr_payload": json.dumps(payload)})
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_valid"] is False
    assert data["badge_status"] == "INVALID_FORMAT"
    assert "expiry_date" in data["message"]


def test_scan_with_valid_qr_metadata(tmp_path):
    """5. Valid QR metadata + badge image -> dose prediction works and returns all traceability fields."""
    import json
    badge_path = tmp_path / "scan_qr_valid.png"
    generate_test_badge_image(dose_ppm_h=14.0, temp_c=25.0, humidity_pct=50.0, is_expired=False, output_path=badge_path)
    
    qr_payload = {
        "badge_id": "H2S-B17-00482",
        "batch_id": "B17",
        "manufactured_date": "2026-09-01",
        "expiry_date": "2026-12-01",
        "calibration_version": "CAL-03"
    }

    with open(badge_path, "rb") as f:
        resp = client.post(
            "/api/scan",
            files={"file": ("badge.png", f, "image/png")},
            data={
                "worker_id": "WRK-1001",
                "qr_payload": json.dumps(qr_payload),
                "temperature_c": 25.0,
                "humidity_percent": 50.0,
                "exposure_time_minutes": 240.0
            }
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["badge_status"] == "VALID"
    assert data["badge_id"] == "H2S-B17-00482"
    assert data["batch_id"] == "B17"
    assert data["calibration_version"] == "CAL-03"
    assert data["cumulative_dose_ppm_h"] > 0.0
    assert data["twa_ppm"] is not None
    assert data["reliability_score"] is not None
    assert data["model_version"] == "v1.0-xgboost"


def test_scan_with_expired_qr_blocked(tmp_path):
    """6. Expired QR metadata -> scan rejected without dose calculation."""
    import json
    badge_path = tmp_path / "scan_qr_expired.png"
    generate_test_badge_image(dose_ppm_h=10.0, is_expired=False, output_path=badge_path)
    
    expired_qr = {
        "badge_id": "H2S-B12-00088",
        "batch_id": "B12",
        "manufactured_date": "2026-04-01",
        "expiry_date": "2026-07-01",
        "calibration_version": "CAL-03"
    }

    with open(badge_path, "rb") as f:
        resp = client.post(
            "/api/scan",
            files={"file": ("badge.png", f, "image/png")},
            data={
                "worker_id": "WRK-1003",
                "qr_payload": json.dumps(expired_qr),
                "temperature_c": 25.0,
                "humidity_percent": 50.0
            }
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert data["badge_status"] == "EXPIRED"
    assert data["badge_id"] == "H2S-B12-00088"
    assert data["cumulative_dose_ppm_h"] == 0.0
    assert "badge expired" in data["action_recommendation"].lower()
    assert "new badge" in data["action_recommendation"].lower()


def test_badge_id_and_calibration_stored_in_db(tmp_path):
    """7. Traceability: Badge ID, Batch ID, and Calibration Version persisted in database."""
    import json
    badge_path = tmp_path / "scan_db_test.png"
    generate_test_badge_image(dose_ppm_h=16.0, output_path=badge_path)
    
    unique_badge_id = "H2S-B22-99123"
    unique_batch_id = "B22"
    unique_cal_ver = "CAL-03"
    qr_data = {
        "badge_id": unique_badge_id,
        "batch_id": unique_batch_id,
        "manufactured_date": "2026-09-01",
        "expiry_date": "2026-12-01",
        "calibration_version": unique_cal_ver
    }

    with open(badge_path, "rb") as f:
        resp = client.post(
            "/api/scan",
            files={"file": ("badge.png", f, "image/png")},
            data={
                "worker_id": "WRK-1004",
                "qr_payload": json.dumps(qr_data),
                "temperature_c": 28.0,
                "humidity_percent": 55.0
            }
        )

    assert resp.status_code == 200
    rec_id = resp.json()["record_id"]

    # Verify directly from Database session
    db = SessionLocal()
    try:
        saved_rec = db.query(ExposureRecord).filter(ExposureRecord.id == rec_id).first()
        assert saved_rec is not None
        assert saved_rec.badge_id == unique_badge_id
        assert saved_rec.batch_id == unique_batch_id
        assert saved_rec.calibration_version == unique_cal_ver
        assert saved_rec.badge_status == "VALID"
        assert saved_rec.model_version == "v1.0-xgboost"
        assert saved_rec.reliability_score is not None
    finally:
        db.close()


def test_synthetic_valid_badge_generation():
    """8. Synthetic valid badge generation endpoint returns QR payload and valid expiry."""
    payload = {
        "exposure": 12.0,
        "temp": 28.0,
        "humidity": 55.0,
        "lighting": "daylight",
        "expired": False,
        "badge_id": "H2S-B17-00482",
        "batch_id": "B17",
        "calibration_version": "CAL-03"
    }
    resp = client.post("/api/generate-badge", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["is_expired"] is False
    assert data["badge_id"] == "H2S-B17-00482"
    assert data["batch_id"] == "B17"
    assert data["calibration_version"] == "CAL-03"
    assert "qr_payload" in data
    assert data["qr_payload"]["badge_id"] == "H2S-B17-00482"


def test_synthetic_expired_badge_generation():
    """9. Synthetic expired badge generation endpoint returns expired date metadata."""
    payload = {
        "exposure": 8.0,
        "expired": True,
        "badge_id": "H2S-B12-00019",
        "batch_id": "B12",
        "calibration_version": "CAL-03"
    }
    resp = client.post("/api/generate-badge", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["is_expired"] is True
    assert data["badge_id"] == "H2S-B12-00019"
    assert "qr_payload" in data
    assert data["qr_payload"]["expiry_date"] < "2026-09-06"

