import pytest
from fastapi.testclient import TestClient
from src.backend.api import app
from src.ml.as7343_model import AS7343Predictor, SPECTRAL_CHANNELS

client = TestClient(app)

def test_as7343_predictor_ratios():
    # 14 channels (Clear is index 12)
    sample_channels = [
        172, 891, 4365, 4078, 4773, 1639, 6588, 5866, 4402, 2007, 233, 162, 7750, 0
    ]
    res = AS7343Predictor.predict(sample_channels, exposure_duration_hours=4.0)
    assert res.cumulative_dose_ppm_h >= 0.0
    assert res.twa_8h_equivalent_ppm >= 0.0
    assert res.clear_count == 7750.0
    assert len(res.ratios) == 12
    assert "F1" in res.ratios
    assert res.ratios["F1"] == round(172 / 7750.0, 6)
    assert res.risk_tier in ["SAFE", "WARNING", "DANGER", "CRITICAL"]

def test_as7343_predictor_zero_clear_rejection():
    with pytest.raises(ValueError):
        AS7343Predictor.predict([100]*12, clear=0)

def test_hardware_reading_api_endpoint():
    payload = {
        "worker_id": "WRK-1001",
        "badge_id": "H2S-HW-TEST-01",
        "batch_id": "HW-01",
        "calibration_version": "AS7343-CAL-01",
        "channels": [172, 891, 4365, 4078, 4773, 1639, 6588, 5866, 4402, 2007, 233, 162, 7750, 0],
        "temperature_c": 29.5,
        "humidity_percent": 58.2,
        "pressure_hpa": 1011.8,
        "gas_resistance_ohms": 48200.0,
        "exposure_duration_minutes": 240.0,
        "is_saturated": False,
        "led_current_ma": 25
    }
    response = client.post("/api/hardware/reading", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["worker_id"] == "WRK-1001"
    assert data["measurement_source"] == "SPECTRAL_HARDWARE_AS7343"
    assert data["cumulative_dose_ppm_h"] > 0
    assert data["twa_8h_equivalent_ppm"] > 0
    assert "ratios" in data
    assert "raw_channels" in data
    assert data["record_id"] is not None
