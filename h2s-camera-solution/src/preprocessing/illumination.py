"""
Reference-Based Illumination Normalization and Color Constancy Engine.
Performs lighting compensation by calibrating measured multi-step reference color swatches
against standard calibrated D65 target values to reduce ambient illumination variability.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
from src.utils.logger import get_logger

logger = get_logger("CV_Illumination")

# Calibrated target RGB values for the 4 reference swatches printed on the badge.
# These MUST match the swatch colors in badge_generator.py and gen_real_color_badges.py exactly.
# Warm-tan lead acetate family: fresh cream -> dark brown (PbAc color scale)
CALIBRATED_TARGETS_RGB = {
    "ref_white":      np.array([200.0, 185.0, 145.0], dtype=np.float32),  # L1 fresh cream
    "ref_light_gray": np.array([161.0, 133.0,  83.0], dtype=np.float32),  # L2 light tan
    "ref_med_gray":   np.array([113.0,  81.0,  39.0], dtype=np.float32),  # L3 medium brown
    "ref_dark":       np.array([ 59.0,  39.0,  16.0], dtype=np.float32),  # L4 dark brown
}


def robust_patch_mean(patch: np.ndarray) -> np.ndarray:
    """Estimate patch color while excluding clipped and near-black pixels."""
    pixels = patch.reshape(-1, 3).astype(np.float32)
    valid = np.all((pixels > 5.0) & (pixels < 250.0), axis=1)
    if int(np.sum(valid)) < max(20, int(len(pixels) * 0.1)):
        return np.median(pixels, axis=0)
    return np.median(pixels[valid], axis=0)


def apply_clahe_preprocess(image_bgr: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) per channel
    to recover detail in underexposed or overexposed camera photos before
    illumination normalization runs.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def assess_capture_quality(image_bgr: np.ndarray) -> Dict[str, Any]:
    """Measure basic capture quality before color features reach the model."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Image is empty.")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    glare_fraction = float(np.mean(gray >= 250))
    reasons = []
    if blur_score < 18.0:
        reasons.append("image is too blurry — hold phone steady and focus on badge")
    if brightness < 25.0:
        reasons.append("image is too dark — move to better lighting or enable torch")
    if brightness > 248.0:
        reasons.append("image is overexposed — reduce lighting or move away from direct light")
    if contrast < 8.0:
        reasons.append("image has insufficient contrast — ensure badge fills the frame")
    if glare_fraction > 0.15:
        reasons.append("image contains excessive glare — tilt badge slightly to reduce reflection")
    return {
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "glare_pixel_percentage": round(glare_fraction * 100.0, 2),
        "is_acceptable": not reasons,
        "retake_reason": "; ".join(reasons) if reasons else None,
    }


def compute_channel_transfer_gains(
    reference_rois: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Compute channel-wise affine color transfer mapping: Target = M * Measured + C
    using least-squares regression across the 4 reference color swatches.
    
    Parameters
    ----------
    reference_rois : Dict[str, np.ndarray]
        Dictionary with keys 'ref_white', 'ref_light_gray', 'ref_med_gray', 'ref_dark',
        containing the cropped BGR image arrays.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Dict[str, Any]]
        Slope multipliers M (shape: (3,)), Offsets C (shape: (3,)), and diagnostics dict.
    """
    measured_means_bgr = []
    target_values_bgr = []
    
    for key, target_rgb in CALIBRATED_TARGETS_RGB.items():
        if key in reference_rois and reference_rois[key] is not None and reference_rois[key].size > 0:
            patch = reference_rois[key]
            # Compute mean in BGR
            mean_bgr = robust_patch_mean(patch)
            measured_means_bgr.append(mean_bgr)
            # Target RGB to BGR: [B, G, R]
            target_bgr = np.array([target_rgb[2], target_rgb[1], target_rgb[0]], dtype=np.float32)
            target_values_bgr.append(target_bgr)

    measured = np.array(measured_means_bgr)  # (N, 3)
    targets = np.array(target_values_bgr)    # (N, 3)
    
    if len(measured) < 2:
        logger.warning("Insufficient reference swatches for multi-point regression. Using unit gains.")
        return np.ones(3, dtype=np.float32), np.zeros(3, dtype=np.float32), {"method": "fallback_unit"}

    slopes = np.zeros(3, dtype=np.float32)
    offsets = np.zeros(3, dtype=np.float32)

    # Check if reference swatches have enough spread for reliable regression
    swatch_spread = float(np.max(measured) - np.min(measured))
    if swatch_spread < 30.0:
        logger.warning(f"Reference swatch spread too low ({swatch_spread:.1f}) — badge too small or poorly lit. Using unit gains.")
        return np.ones(3, dtype=np.float32), np.zeros(3, dtype=np.float32), {"method": "fallback_low_spread", "swatch_spread": swatch_spread}

    for c in range(3):  # B, G, R channels
        x = measured[:, c]
        y = targets[:, c]
        # Guard against degenerate polyfit when all swatches have same value
        if np.ptp(x) < 10.0:
            slopes[c] = 1.0
            offsets[c] = 0.0
            continue
        m, b = np.polyfit(x, y, 1)
        # Wider clamp to handle real camera underexposure (dark rooms, dim offices)
        m_clamped = float(np.clip(m, 0.75, 1.80))
        b_clamped = float(np.clip(b, -45.0, 45.0))
        slopes[c] = m_clamped
        offsets[c] = b_clamped

    diagnostics = {
        "method": "multi_point_affine_regression",
        "n_swatches_used": len(measured),
        "channel_gains_bgr": [round(float(s), 3) for s in slopes],
        "channel_offsets_bgr": [round(float(o), 2) for o in offsets]
    }
    
    return slopes, offsets, diagnostics


def normalize_patch_illumination(
    patch_bgr: np.ndarray,
    slopes: np.ndarray,
    offsets: np.ndarray
) -> np.ndarray:
    """
    Apply computed color transfer function to normalize patch lighting.
    I_norm = clip(slopes * I_raw + offsets, 0, 255)
    """
    patch_float = patch_bgr.astype(np.float32)
    normalized = np.zeros_like(patch_float)
    
    for c in range(3):
        normalized[:, :, c] = patch_float[:, :, c] * slopes[c] + offsets[c]
        
    return np.clip(normalized, 0, 255).astype(np.uint8)


def assess_lighting_quality(
    sensing_patch_bgr: np.ndarray,
    reference_rois: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """
    Inspect image lighting conditions for glare, specular highlights, and severe underexposure.
    """
    sensing_gray = cv2.cvtColor(sensing_patch_bgr, cv2.COLOR_BGR2GRAY)
    
    # Check specular glare (pixels near clipping limit 255)
    glare_fraction = float(np.mean(sensing_gray > 250))
    is_glare_detected = glare_fraction > 0.05
    
    # Check underexposure from lightest reference patch (L1 fresh cream)
    # L1 target RGB(200,185,145) -> mean ~177. Underexposed if camera reads below 45.
    white_mean = float(robust_patch_mean(reference_rois["ref_white"]).mean()) if "ref_white" in reference_rois else float(np.mean(sensing_gray))
    is_underexposed = white_mean < 45.0

    # Color temperature: compare measured L1 ref_white to its known target BGR(145,185,200)
    # Expected B/R ratio for L1 warm-tan swatch: 145/200 = 0.725
    if "ref_white" in reference_rois and reference_rois["ref_white"].size > 0:
        w_bgr = np.mean(reference_rois["ref_white"], axis=(0, 1))  # B,G,R
        br_ratio = max(w_bgr[0], 1.0) / max(w_bgr[2], 1.0)
        if br_ratio > 0.90:
            lighting_profile = "COOL / BLUE_CAST"
        elif br_ratio < 0.55:
            lighting_profile = "WARM / INCANDESCENT"
        else:
            lighting_profile = "NEUTRAL / BALANCED"
    else:
        lighting_profile = "UNKNOWN"

    quality_report = {
        "lighting_profile": lighting_profile,
        "is_glare_detected": is_glare_detected,
        "glare_pixel_percentage": round(glare_fraction * 100.0, 2),
        "is_underexposed": is_underexposed,
        "white_patch_intensity": round(white_mean, 1),
        "is_lighting_suitable": not (is_glare_detected or is_underexposed)
    }
    
    return quality_report
