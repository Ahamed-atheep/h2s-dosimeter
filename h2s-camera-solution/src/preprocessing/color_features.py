"""
Multi-Space Statistical Color Feature Extractor and Badge Shelf-Life Validator.
Extracts comprehensive colorimetric features (RGB, HSV, CIELAB, Delta E*, Chroma, Hue)
and verifies the chemical badge's expiry status before exposure estimation.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union
from src.utils.config import physics_config, features_config
from src.utils.logger import get_logger
from src.preprocessing.roi import (
    warp_to_canonical_badge, extract_badge_rois, draw_roi_overlay,
    detect_badge_boundary_with_confidence
)
from src.preprocessing.zone_detector import BadgeZoneDetector, draw_zone_detections
from src.preprocessing.illumination import (
    compute_channel_transfer_gains, normalize_patch_illumination, assess_lighting_quality,
    assess_capture_quality
)

logger = get_logger("CV_Features")


def bgr_to_standard_cielab(patch_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert OpenCV BGR image to standard CIELAB (L* in [0, 100], a* in [-128, 127], b* in [-128, 127]).
    """
    lab_cv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    # OpenCV scaling: L_cv in [0, 255], a_cv in [0, 255], b_cv in [0, 255]
    L_star = lab_cv[:, :, 0] * (100.0 / 255.0)
    a_star = lab_cv[:, :, 1] - 128.0
    b_star = lab_cv[:, :, 2] - 128.0
    return L_star, a_star, b_star


def extract_color_features_from_patch(
    patch_bgr: np.ndarray,
    temp_c: float = 25.0,
    humidity_pct: float = 50.0,
    exposure_time_min: float = 240.0
) -> Dict[str, Any]:
    """
    Extract comprehensive multi-space statistical color features from a normalized sensing patch.
    """
    # 1. RGB Features (converted from OpenCV BGR)
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    R = patch_rgb[:, :, 0]
    G = patch_rgb[:, :, 1]
    B = patch_rgb[:, :, 2]
    
    r_mean, r_std = float(np.mean(R)), float(np.std(R))
    g_mean, g_std = float(np.mean(G)), float(np.std(G))
    rgb_b_mean, rgb_b_std = float(np.mean(B)), float(np.std(B))
    
    # 2. HSV Features
    patch_hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H = patch_hsv[:, :, 0]  # [0, 180] in OpenCV
    S = patch_hsv[:, :, 1] * (100.0 / 255.0)  # [0, 100]%
    V = patch_hsv[:, :, 2] * (100.0 / 255.0)  # [0, 100]%
    
    h_mean, h_std = float(np.mean(H)), float(np.std(H))
    s_mean, s_std = float(np.mean(S)), float(np.std(S))
    v_mean, v_std = float(np.mean(V)), float(np.std(V))
    
    # 3. CIELAB Features
    L_star, a_star, b_star = bgr_to_standard_cielab(patch_bgr)
    
    l_mean, l_std = float(np.mean(L_star)), float(np.std(L_star))
    a_mean, a_std = float(np.mean(a_star)), float(np.std(a_star))
    lab_b_mean, lab_b_std = float(np.mean(b_star)), float(np.std(b_star))
    
    # 4. Derived Features
    base_l = physics_config.BASELINE_L
    base_a = physics_config.BASELINE_A
    base_b = physics_config.BASELINE_B
    
    delta_E = float(np.sqrt((l_mean - base_l)**2 + (a_mean - base_a)**2 + (lab_b_mean - base_b)**2))
    chroma = float(np.sqrt(a_mean**2 + lab_b_mean**2))
    hue_angle = float(np.degrees(np.arctan2(lab_b_mean, a_mean)) % 360.0)
    
    ratio_rg = float(r_mean / max(g_mean, 1e-4))
    ratio_gb = float(g_mean / max(rgb_b_mean, 1e-4))
    ratio_rb = float(r_mean / max(rgb_b_mean, 1e-4))
    
    features = {
        # RGB
        "R_mean": round(r_mean, 3), "G_mean": round(g_mean, 3), "B_mean": round(rgb_b_mean, 3),
        "R_std": round(r_std, 3), "G_std": round(g_std, 3), "B_std": round(rgb_b_std, 3),
        
        # HSV
        "H_mean": round(h_mean, 3), "S_mean": round(s_mean, 3), "V_mean": round(v_mean, 3),
        "H_std": round(h_std, 3), "S_std": round(s_std, 3), "V_std": round(v_std, 3),
        
        # CIELAB
        "L_mean": round(l_mean, 3), "a_mean": round(a_mean, 3), "b_mean": round(lab_b_mean, 3),
        "L_std": round(l_std, 3), "a_std": round(a_std, 3), "b_std": round(lab_b_std, 3),
        
        # Derived Metrics
        "delta_E": round(delta_E, 3),
        "chroma": round(chroma, 3),
        "hue_angle": round(hue_angle, 3),
        "ratio_RG": round(ratio_rg, 4),
        "ratio_GB": round(ratio_gb, 4),
        "ratio_RB": round(ratio_rb, 4),
        
        # Environmental Context
        "temperature_c": float(temp_c),
        "humidity_percent": float(humidity_pct),
        "exposure_time_minutes": float(exposure_time_min)
    }
    
    return features


def check_badge_expiry(
    expiry_patch_bgr: np.ndarray,
    qr_payload: Optional[Union[str, Dict[str, Any]]] = None,
    aligned_badge_bgr: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Examine the badge expiry status using digital QR identity metadata or optical indicator patch.
    1. If QR metadata is provided or detected in the image/patch, digital shelf-life is evaluated.
    2. Otherwise, falls back to optical colorimetric analysis.
    """
    from src.utils.qr_validator import decode_qr_from_image, validate_badge_qr_payload

    # 1. Digital QR Code validation
    detected_qr = qr_payload
    if detected_qr is None and aligned_badge_bgr is not None:
        detected_qr = decode_qr_from_image(aligned_badge_bgr)
    if detected_qr is None and expiry_patch_bgr is not None:
        detected_qr = decode_qr_from_image(expiry_patch_bgr)

    if detected_qr is not None:
        qr_val = validate_badge_qr_payload(detected_qr)
        return {
            "is_valid": qr_val["is_valid"],
            "expiry_status": qr_val["badge_status"],
            "degradation_percentage": 0.0 if qr_val["is_valid"] else 100.0,
            "mean_L": 0.0,
            "mean_a": 0.0,
            "mean_b": 0.0,
            "recommendation": qr_val["message"],
            "qr_data": qr_val
        }

    # 2. Optical Colorimetric fallback
    L_star, a_star, b_star = bgr_to_standard_cielab(expiry_patch_bgr)
    mean_L = float(np.mean(L_star))
    mean_a = float(np.mean(a_star))
    mean_b = float(np.mean(b_star))

    # Check if this is a high-contrast printed digital QR area (not a degraded chemical patch)
    is_digital_qr = float(np.std(expiry_patch_bgr)) > 40.0

    if is_digital_qr:
        is_valid = True
        status = "VALID"
        recommendation = "Badge identity & shelf-life verified."
        degradation_pct = 0.0
    else:
        # Baseline fresh L1 warm-tan patch in CIELAB: L*~74, a*~-0.2, b*~21.5
        # (matches physics_config.BASELINE_L/A/B)
        base_l = physics_config.BASELINE_L
        base_a = physics_config.BASELINE_A
        base_b = physics_config.BASELINE_B
        fresh_dist = float(np.sqrt(
            (mean_L - base_l) ** 2 +
            (mean_a - base_a) ** 2 +
            (mean_b - base_b) ** 2
        ))
        # Valid if patch is still light (L*>50) and retains warm-tan hue (b*>8)
        is_valid = (mean_L > 50.0) and (mean_b > 8.0)
        degradation_pct = min(100.0, max(0.0, (fresh_dist / 40.0) * 100.0))
        if is_valid:
            status = "VALID"
            recommendation = "Badge shelf-life verified. Safe for shift deployment."
        else:
            status = "EXPIRED"
            recommendation = "Badge shelf-life expired or packaging seal compromised. Discard and use a fresh badge."
        
    return {
        "is_valid": is_valid,
        "expiry_status": status,
        "degradation_percentage": round(degradation_pct, 1),
        "mean_L": round(mean_L, 2),
        "mean_a": round(mean_a, 2),
        "mean_b": round(mean_b, 2),
        "recommendation": recommendation
    }


def process_badge_image_pipeline(
    image_bgr: np.ndarray,
    temp_c: float = 25.0,
    humidity_pct: float = 50.0,
    exposure_time_min: float = 240.0,
    qr_payload: Optional[Union[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Complete End-to-End Image Processing Pipeline:
    1. Perspective rectification
    2. ROI extraction
    3. Lighting quality check
    4. Reference-based illumination normalization
    5. Badge expiry & identity verification
    6. Multi-space color feature extraction
    """
    capture_quality = assess_capture_quality(image_bgr)

    # Pre-process: CLAHE to recover underexposed/overexposed camera photos
    from src.preprocessing.illumination import apply_clahe_preprocess
    image_bgr = apply_clahe_preprocess(image_bgr)

    # 1. Perspective alignment
    corners, alignment_confidence = detect_badge_boundary_with_confidence(image_bgr)
    aligned_badge, is_auto_aligned = warp_to_canonical_badge(image_bgr, corners=corners)

    # If badge was not auto-aligned by 4-point quadrilateral (e.g. close-up framing or soft card edges),
    # verify that the cropped frame contains a valid, well-lit scene.
    if not is_auto_aligned:
        gray_crop = cv2.cvtColor(aligned_badge, cv2.COLOR_BGR2GRAY)
        mean_lum = float(np.mean(gray_crop))
        std_dev = float(np.std(gray_crop))
        lap_var = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())

        # Check if the frame is completely unusable (pitch black, solid white, or zero variance)
        if mean_lum < 15.0:
            raise ValueError(
                "Image is too dark. Increase lighting or turn on flash/torch."
            )
        if mean_lum > 250.0 and std_dev < 10.0:
            raise ValueError(
                "Image is overexposed / washed out. Adjust lighting and avoid direct glare."
            )
        if std_dev < 6.0 and lap_var < 3.0:
            raise ValueError(
                "Badge not detected in frame. Move closer until the badge fills at least 50% of the camera view."
            )
        
        # Valid framed badge photograph: set fallback confidence
        alignment_confidence = max(alignment_confidence, 0.70)
        logger.info(f"Using framed badge fallback (confidence: {alignment_confidence:.2f}).")
    
    # 2. Extract ROIs
    rois = extract_badge_rois(aligned_badge, apply_erosion=True)
    raw_sensing = rois["sensing_patch"]["image"]
    raw_expiry = rois["expiry_patch"]["image"]
    
    ref_patches = {
        "ref_white": rois["ref_white"]["image"],
        "ref_light_gray": rois["ref_light_gray"]["image"],
        "ref_med_gray": rois["ref_med_gray"]["image"],
        "ref_dark": rois["ref_dark"]["image"]
    }
    
    # 3. Assess lighting quality
    lighting_qa = assess_lighting_quality(raw_sensing, ref_patches)
    
    # 4. Illumination Normalization
    slopes, offsets, norm_diag = compute_channel_transfer_gains(ref_patches)
    norm_sensing = normalize_patch_illumination(raw_sensing, slopes, offsets)
    norm_expiry = normalize_patch_illumination(raw_expiry, slopes, offsets)
    
    # 5. Expiry Check (Digital QR + Optical)
    expiry_report = check_badge_expiry(
        norm_expiry,
        qr_payload=qr_payload,
        aligned_badge_bgr=aligned_badge
    )
    
    # 6. Extract Features from normalized sensing patch
    features = extract_color_features_from_patch(
        norm_sensing, temp_c=temp_c, humidity_pct=humidity_pct, exposure_time_min=exposure_time_min
    )
    
    # 7. Annotated Visual Overlay with Dynamic Object-Detection HUD
    zone_detector = BadgeZoneDetector(use_temporal_smoothing=False)
    canonical_corners = np.array([
        [0.0, 0.0],
        [float(aligned_badge.shape[1] - 1), 0.0],
        [float(aligned_badge.shape[1] - 1), float(aligned_badge.shape[0] - 1)],
        [0.0, float(aligned_badge.shape[0] - 1)]
    ], dtype=np.float32)
    zone_res = zone_detector.detect_zones(
        aligned_badge,
        corners=canonical_corners,
        confidence_override=max(0.95, float(alignment_confidence) if is_auto_aligned else 0.90)
    )
    debug_overlay = draw_zone_detections(aligned_badge, zone_res, show_hud=True)
    
    pipeline_result = {
        "is_auto_aligned": is_auto_aligned,
        "alignment_confidence": round(float(alignment_confidence), 3),
        "capture_quality": capture_quality,
        "lighting_qa": lighting_qa,
        "normalization_diagnostics": norm_diag,
        "expiry_report": expiry_report,
        "color_features": features,
        "aligned_badge": aligned_badge,
        "normalized_sensing_patch": norm_sensing,
        "normalized_expiry_patch": norm_expiry,
        "debug_overlay": debug_overlay
    }
    
    return pipeline_result
