"""
Unit and Integration Tests for Dynamic Live Camera-Based Zone Detection.
Verifies real-time detection, homography projection, 6-zone tracking,
rotation estimation, confidence scoring, and API endpoints.
"""

import cv2
import pytest
import base64
import numpy as np
from fastapi.testclient import TestClient

from src.backend.api import app
from src.utils.badge_generator import generate_test_badge_image
from src.preprocessing.zone_detector import (
    BadgeZoneDetector, BadgeZoneTracker, draw_zone_detections,
    CANONICAL_ZONE_DEFS, CANONICAL_WIDTH, CANONICAL_HEIGHT
)
from src.preprocessing.camera_guide import analyze_live_frame

client = TestClient(app)


def test_canonical_zone_definitions():
    """Verify that all 6 required zones are defined with valid bounding geometry and colors."""
    assert "sensing_zone" in CANONICAL_ZONE_DEFS
    assert "ref_white" in CANONICAL_ZONE_DEFS
    assert "ref_light_gray" in CANONICAL_ZONE_DEFS
    assert "ref_med_gray" in CANONICAL_ZONE_DEFS
    assert "ref_dark" in CANONICAL_ZONE_DEFS
    assert "qr_badge" in CANONICAL_ZONE_DEFS
    assert len(CANONICAL_ZONE_DEFS) == 6

    for zid, zdef in CANONICAL_ZONE_DEFS.items():
        x1, y1, x2, y2 = zdef["box"]
        assert 0 <= x1 < x2 <= CANONICAL_WIDTH
        assert 0 <= y1 < y2 <= CANONICAL_HEIGHT
        assert "color_hex" in zdef
        assert "color_bgr" in zdef
        assert "category" in zdef


def test_detector_finds_all_six_zones_upright():
    """Test dynamic zone detection on a standard upright badge image."""
    img, _ = generate_test_badge_image(dose_ppm_h=10.0, lighting="daylight", tilt_angle=0.0)
    detector = BadgeZoneDetector(use_temporal_smoothing=False)
    
    result = detector.detect_zones(img)
    
    assert result["badge_detected"] is True
    assert result["confidence"] > 0.80
    assert result["corners"] is not None
    assert len(result["corners"]) == 4
    
    zones = result["zones"]
    assert len(zones) == 6
    
    zone_ids = [z.id for z in zones]
    assert "sensing_zone" in zone_ids
    assert "ref_white" in zone_ids
    assert "ref_light_gray" in zone_ids
    assert "ref_med_gray" in zone_ids
    assert "ref_dark" in zone_ids
    assert "qr_badge" in zone_ids

    # Validate normalized polygon coordinates are within [0.0, 1.0]
    for z in zones:
        assert len(z.polygon_norm) == 4
        for pt in z.polygon_norm:
            assert 0.0 <= pt[0] <= 1.0
            assert 0.0 <= pt[1] <= 1.0
        assert 0.0 <= z.bbox_norm["x"] <= 1.0
        assert 0.0 <= z.bbox_norm["y"] <= 1.0
        assert z.bbox_norm["width"] > 0.0
        assert z.bbox_norm["height"] > 0.0


def test_detector_tracks_tilted_badge():
    """Test that dynamic zone detection accurately projects and follows a tilted badge."""
    tilt_degrees = 15.0
    img, _ = generate_test_badge_image(dose_ppm_h=20.0, lighting="warm", tilt_angle=tilt_degrees)
    detector = BadgeZoneDetector(use_temporal_smoothing=False)
    
    result = detector.detect_zones(img)
    
    assert result["badge_detected"] is True
    assert len(result["zones"]) == 6
    
    # Check that sensing zone is on the left and QR is on the right
    sensing = next(z for z in result["zones"] if z.id == "sensing_zone")
    qr = next(z for z in result["zones"] if z.id == "qr_badge")
    
    assert sensing.bbox_norm["x"] < qr.bbox_norm["x"]


def test_temporal_smoothing_tracker():
    """Verify that BadgeZoneTracker smoothes corner coordinates across successive frames."""
    tracker = BadgeZoneTracker(smoothing_factor=0.5)
    
    corners_f1 = np.array([[100, 100], [500, 100], [500, 400], [100, 400]], dtype=np.float32)
    smoothed_1, conf_1 = tracker.update(corners_f1, 0.95)
    assert np.allclose(smoothed_1, corners_f1)
    
    # Second frame with sudden shift
    corners_f2 = np.array([[120, 100], [520, 100], [520, 400], [120, 400]], dtype=np.float32)
    smoothed_2, conf_2 = tracker.update(corners_f2, 0.95)
    # EMA smoothed position should be midpoint between f1 and f2
    expected_x = 110.0
    assert np.isclose(smoothed_2[0, 0], expected_x, atol=2.0)


def test_draw_zone_detections_rendering():
    """Test that draw_zone_detections renders without error on detected frame and empty frame."""
    img, _ = generate_test_badge_image(dose_ppm_h=8.0)
    detector = BadgeZoneDetector(use_temporal_smoothing=False)
    result = detector.detect_zones(img)
    
    annotated = draw_zone_detections(img, result, show_hud=True, show_fps=45.2)
    assert annotated.shape == img.shape
    assert annotated.dtype == np.uint8

    # Empty detection case
    empty_result = {"badge_detected": False, "zones": []}
    annotated_empty = draw_zone_detections(img, empty_result, show_hud=True)
    assert annotated_empty.shape == img.shape


def test_camera_guide_includes_dynamic_zones():
    """Test that analyze_live_frame output includes the dynamic zones and rotation metadata."""
    img, _ = generate_test_badge_image(dose_ppm_h=12.0)
    result = analyze_live_frame(img)
    
    assert result["badge_detected"] is True
    assert "zones" in result
    assert isinstance(result["zones"], list)
    assert len(result["zones"]) == 6
    assert "rotation_deg" in result
    assert isinstance(result["rotation_deg"], (int, float))


def test_api_detect_zones_endpoint():
    """Test POST /api/camera/detect-zones endpoint returns 6-zone payload."""
    img, _ = generate_test_badge_image(dose_ppm_h=15.0)
    _, buffer = cv2.imencode(".jpg", img)
    b64_str = base64.b64encode(buffer).decode("utf-8")
    
    response = client.post("/api/camera/detect-zones", json={"image_base64": b64_str})
    assert response.status_code == 200
    data = response.json()
    
    assert data["badge_detected"] is True
    assert data["confidence"] > 0.70
    assert len(data["zones"]) == 6
    for z in data["zones"]:
        assert "id" in z
        assert "name" in z
        assert "color" in z
        assert "confidence" in z
        assert "polygon" in z
        assert "bbox" in z
