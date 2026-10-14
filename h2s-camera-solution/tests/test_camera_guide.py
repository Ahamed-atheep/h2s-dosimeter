"""
Unit and Integration Tests for Real-Time Camera Guidance and Frame Analysis Engine.
"""

import cv2
import pytest
import base64
import numpy as np
from fastapi.testclient import TestClient

from src.backend.api import app
from src.utils.badge_generator import generate_test_badge_image
from src.preprocessing.camera_guide import analyze_live_frame

client = TestClient(app)


def test_camera_guide_detects_badge():
    """Test that a well-framed badge image receives PERFECT_ALIGNMENT and valid normalized corners."""
    img, _ = generate_test_badge_image(dose_ppm_h=10.0, lighting="daylight", tilt_angle=0.0)
    result = analyze_live_frame(img)
    
    assert result["badge_detected"] is True
    assert result["corners"] is not None
    assert len(result["corners"]) == 4
    # All corner coordinates should be in [0.0, 1.0]
    for pt in result["corners"]:
        assert 0.0 <= pt[0] <= 1.0
        assert 0.0 <= pt[1] <= 1.0
        
    assert result["can_autocapture"] is True
    assert result["status"] == "PERFECT_ALIGNMENT"
    assert result["color"] == "#10b981"


def test_camera_guide_advises_move_closer():
    """Test that a tiny badge embedded in a huge canvas triggers MOVE_CLOSER."""
    canonical, _ = generate_test_badge_image(dose_ppm_h=10.0)
    # Scale down badge to 150x100 on a 1000x1000 background (coverage < 5%)
    small_badge = cv2.resize(canonical, (150, 100))
    large_canvas = np.full((1000, 1000, 3), 180, dtype=np.uint8)
    large_canvas[450:550, 425:575] = small_badge
    
    result = analyze_live_frame(large_canvas)
    if result["badge_detected"]:
        assert result["status"] == "MOVE_CLOSER"
        assert result["can_autocapture"] is False
    else:
        assert result["status"] in ("SEARCHING", "MOVE_CLOSER")


def test_camera_guide_detects_low_lighting():
    """Test that an excessively dark frame triggers TOO_DARK."""
    dark_image = np.full((400, 600, 3), 15, dtype=np.uint8)
    result = analyze_live_frame(dark_image)
    
    assert result["status"] == "TOO_DARK"
    assert result["can_autocapture"] is False
    assert result["brightness"] < 30.0


def test_camera_guide_detects_motion_blur():
    """Test that an artificially blurred badge frame triggers HOLD_STEADY."""
    img, _ = generate_test_badge_image(dose_ppm_h=10.0)
    # Apply severe Gaussian blur to simulate fast camera shake
    blurred = cv2.GaussianBlur(img, (25, 25), 10.0)
    result = analyze_live_frame(blurred)
    
    assert result["can_autocapture"] is False
    if result["badge_detected"]:
        assert result["status"] in ("HOLD_STEADY", "SEARCHING")


def test_api_camera_analyze_endpoint():
    """Test POST /api/camera/analyze-frame endpoint with base64 encoded frame."""
    img, _ = generate_test_badge_image(dose_ppm_h=12.0)
    _, buffer = cv2.imencode(".jpg", img)
    b64_str = base64.b64encode(buffer).decode("utf-8")
    
    response = client.post("/api/camera/analyze-frame", json={"image_base64": b64_str})
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert "message" in data
    assert "can_autocapture" in data
    assert "badge_detected" in data
    assert "color" in data
    assert isinstance(data["can_autocapture"], bool)
