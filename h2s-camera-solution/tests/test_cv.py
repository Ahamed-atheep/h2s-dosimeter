"""
Unit and Integration Tests for Module 2 (Computer Vision & Preprocessing Pipeline).
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

from src.utils.config import physics_config
from src.utils.badge_generator import (
    render_canonical_badge, apply_photorealistic_effects, generate_test_badge_image
)
from src.preprocessing.roi import (
    detect_badge_boundary, warp_to_canonical_badge, extract_badge_rois, BoundingBox
)
from src.preprocessing.illumination import (
    compute_channel_transfer_gains, normalize_patch_illumination, assess_lighting_quality,
    assess_capture_quality
)
from src.preprocessing.color_features import (
    bgr_to_standard_cielab, extract_color_features_from_patch, check_badge_expiry, process_badge_image_pipeline
)
from src.ml.predict import DosimeterPredictor


def test_render_canonical_badge():
    """Verify canonical badge rendering dimensions and chemical darkening."""
    badge_fresh = render_canonical_badge(dose_ppm_h=0.5, temp_c=25.0, humidity_pct=50.0)
    badge_dark = render_canonical_badge(dose_ppm_h=40.0, temp_c=25.0, humidity_pct=50.0)
    
    assert badge_fresh.shape == (400, 600, 3)
    assert badge_dark.shape == (400, 600, 3)
    
    # Sensing patch region in fresh badge should be significantly brighter than in exposed badge
    sensing_fresh = badge_fresh[120:300, 80:250]
    sensing_dark = badge_dark[120:300, 80:250]
    
    assert np.mean(sensing_fresh) > np.mean(sensing_dark) + 20.0


def test_badge_detection_and_warp(tmp_path):
    """Test perspective detection and rectification on a tilted badge image."""
    img_tilted, _ = generate_test_badge_image(
        dose_ppm_h=10.0,
        tilt_angle=8.0,
        output_path=tmp_path / "tilted.png"
    )
    
    warped, is_aligned = warp_to_canonical_badge(img_tilted)
    assert warped.shape == (400, 600, 3)
    assert is_aligned is True


def test_roi_extraction():
    """Verify all canonical ROIs are extracted with proper bounding boxes."""
    canonical = render_canonical_badge(dose_ppm_h=8.0)
    rois = extract_badge_rois(canonical, apply_erosion=True)
    
    expected_keys = ["sensing_patch", "ref_white", "ref_light_gray", "ref_med_gray", "ref_dark", "expiry_patch"]  # internal pipeline keys (L1-L4)
    for key in expected_keys:
        assert key in rois
        patch = rois[key]["image"]
        assert patch.ndim == 3
        assert patch.shape[0] > 10 and patch.shape[1] > 10
        assert np.mean(patch) > 0


def test_illumination_normalization():
    """Verify that multi-point color calibration balances warm ambient casts."""
    canonical = render_canonical_badge(dose_ppm_h=15.0)
    # Apply warm illumination cast (high red, low blue)
    warm_img = apply_photorealistic_effects(canonical, lighting="warm", tilt_angle=0.0)
    aligned_warm, _ = warp_to_canonical_badge(warm_img)
    
    rois = extract_badge_rois(aligned_warm, apply_erosion=True)
    ref_patches = {k: rois[k]["image"] for k in ["ref_white", "ref_light_gray", "ref_med_gray", "ref_dark"]}
    
    slopes, offsets, diag = compute_channel_transfer_gains(ref_patches)
    # Under warm lighting (excess red, deficit blue), Blue slope should be > Red slope
    assert slopes[0] > slopes[2]  # Blue gain > Red gain
    
    norm_sensing = normalize_patch_illumination(rois["sensing_patch"]["image"], slopes, offsets)
    assert norm_sensing.shape == rois["sensing_patch"]["image"].shape


def test_capture_quality_rejects_blurry_image():
    """Very low-detail captures should produce a deterministic retake reason."""
    blurry = np.full((400, 600, 3), 128, dtype=np.uint8)
    report = assess_capture_quality(blurry)

    assert report["is_acceptable"] is False
    assert "blurry" in report["retake_reason"]


def test_capture_quality_accepts_generated_badge():
    """Generated badges remain valid inputs after adding quality gates."""
    badge = render_canonical_badge(dose_ppm_h=15.0)
    report = assess_capture_quality(badge)

    assert report["is_acceptable"] is True


def test_expiry_check():
    """Test expiry indicator verification for valid vs expired badges."""
    canonical_valid = render_canonical_badge(dose_ppm_h=5.0, is_expired=False)
    canonical_expired = render_canonical_badge(dose_ppm_h=5.0, is_expired=True)
    
    rois_val = extract_badge_rois(canonical_valid)
    rois_exp = extract_badge_rois(canonical_expired)
    
    res_val = check_badge_expiry(rois_val["expiry_patch"]["image"])
    res_exp = check_badge_expiry(rois_exp["expiry_patch"]["image"])
    
    assert res_val["is_valid"] is True
    assert res_val["expiry_status"] == "VALID"
    
    assert res_exp["is_valid"] is False
    assert res_exp["expiry_status"] == "EXPIRED"


def test_end_to_end_image_pipeline(tmp_path):
    """End-to-end test from simulated badge photo to predicted cumulative dose."""
    known_dose = 18.0
    img_path = tmp_path / "test_badge_18ppm.png"
    img, _ = generate_test_badge_image(
        dose_ppm_h=known_dose,
        temp_c=28.0,
        humidity_pct=60.0,
        lighting="daylight",
        output_path=img_path
    )
    
    predictor = DosimeterPredictor()
    result, pipeline_res = predictor.predict_from_image(
        img_path, temp_c=28.0, humidity_pct=60.0, exposure_time_min=240.0
    )
    
    assert result.is_badge_valid is True
    assert result.cumulative_dose_ppm_h > 0.0
    # Estimated dose should track close to known 18 ppm*h (within ~20% envelope)
    assert 12.0 <= result.cumulative_dose_ppm_h <= 24.0
    assert pipeline_res["debug_overlay"].shape == (400, 600, 3)
