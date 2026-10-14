"""
Real-time Live Camera Guidance & Positioning Coach Engine.
Analyzes live viewfinder video frames to:
1. Dynamically locate badge quadrilateral boundary and 4 corners (no fixed template).
2. Measure frame coverage, optical centering, sharpness, luminance, and glare.
3. Provide instantaneous coaching instructions:
   - "SEARCHING"
   - "MOVE_CLOSER"
   - "MOVE_BACK"
   - "CENTER_BADGE"
   - "TOO_DARK"
   - "REDUCE_GLARE"
   - "HOLD_STEADY"
   - "PERFECT_ALIGNMENT" (triggers auto-capture countdown)
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from src.preprocessing.roi import detect_badge_boundary_with_confidence, order_points
from src.preprocessing.zone_detector import BadgeZoneDetector, CANONICAL_ZONE_DEFS
from src.utils.logger import get_logger

logger = get_logger("Camera_Guide")

# Global zone detector instance with temporal smoothing
_global_zone_detector = BadgeZoneDetector(use_temporal_smoothing=True, smoothing_factor=0.45)


def analyze_live_frame(
    image_bgr: np.ndarray,
    min_coverage: float = 0.20,
    max_coverage: float = 0.88,
    min_blur_var: float = 28.0,
    min_brightness: float = 40.0,
    max_brightness: float = 245.0,
    max_glare_fraction: float = 0.12
) -> Dict[str, Any]:
    """
    Rapidly analyze a single camera frame for real-time positioning, dynamic zone detection,
    and auto-capture readiness.
    
    Parameters
    ----------
    image_bgr : np.ndarray
        Input frame image in BGR format.
    min_coverage : float
        Minimum badge area fraction of the total frame to avoid being too far.
    max_coverage : float
        Maximum badge area fraction to prevent badge clipping.
    min_blur_var : float
        Minimum Laplacian variance threshold for motion blur detection.
    min_brightness : float
        Minimum mean luminance threshold.
    max_brightness : float
        Maximum mean luminance threshold.
    max_glare_fraction : float
        Maximum allowed percentage of specular highlight pixels.
        
    Returns
    -------
    Dict[str, Any]
        Structured guidance payload with corners, dynamic zones, metrics, coach message,
        and auto-capture flag.
    """
    if image_bgr is None or image_bgr.size == 0:
        return {
            "status": "NO_FRAME",
            "message": "Waiting for camera stream...",
            "color": "#94a3b8",
            "can_autocapture": False,
            "badge_detected": False,
            "corners": None,
            "coverage_pct": 0.0,
            "center_offset_pct": 0.0,
            "blur_score": 0.0,
            "brightness": 0.0,
            "rotation_deg": 0.0,
            "zones": []
        }

    h, w = image_bgr.shape[:2]
    frame_area = float(h * w)

    # 1. Quick Grayscale & Image Statistics
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    glare_fraction = float(np.mean(gray >= 250))

    # 2. Check Lighting First to guide user before boundary search
    if brightness < min_brightness:
        return {
            "status": "TOO_DARK",
            "message": "Too dark. Increase light or turn on torch.",
            "color": "#f59e0b",
            "can_autocapture": False,
            "badge_detected": False,
            "corners": None,
            "coverage_pct": 0.0,
            "center_offset_pct": 0.0,
            "blur_score": round(blur_score, 1),
            "brightness": round(brightness, 1),
            "rotation_deg": 0.0,
            "zones": []
        }

    # 3. Dynamic Quadrilateral & Corner Detection
    raw_corners, conf_score = detect_badge_boundary_with_confidence(image_bgr)

    # If no badge quadrilateral found
    if raw_corners is None or conf_score < 0.25:
        return {
            "status": "SEARCHING",
            "message": "Looking for dosimeter badge...",
            "color": "#38bdf8",
            "can_autocapture": False,
            "badge_detected": False,
            "corners": None,
            "coverage_pct": 0.0,
            "center_offset_pct": 0.0,
            "blur_score": round(blur_score, 1),
            "brightness": round(brightness, 1),
            "rotation_deg": 0.0,
            "zones": []
        }

    # 4. Compute Zone Detections & Projections
    zone_detection = _global_zone_detector.detect_zones(image_bgr, corners=raw_corners, confidence_override=conf_score)
    zones_list = [z.to_dict() if hasattr(z, "to_dict") else z for z in zone_detection.get("zones", [])]
    rotation_deg = zone_detection.get("rotation_deg", 0.0)
    normalized_corners = zone_detection.get("corners", None)

    # 5. Geometric & Spatial Analysis
    ordered_corners = order_points(raw_corners.reshape(4, 2))
    contour_area = float(cv2.contourArea(ordered_corners.astype(np.int32)))
    coverage_pct = (contour_area / max(frame_area, 1.0)) * 100.0

    # Optical center of badge vs center of frame
    badge_center_x = float(np.mean(ordered_corners[:, 0]))
    badge_center_y = float(np.mean(ordered_corners[:, 1]))
    frame_center_x = float(w / 2.0)
    frame_center_y = float(h / 2.0)

    # Normalized center offset (0.0 at exact center, 1.0 at border)
    dx = (badge_center_x - frame_center_x) / frame_center_x
    dy = (badge_center_y - frame_center_y) / frame_center_y
    center_offset_pct = float(np.sqrt(dx * dx + dy * dy) * 50.0)

    base_payload = {
        "badge_detected": True,
        "corners": normalized_corners,
        "coverage_pct": round(coverage_pct, 1),
        "center_offset_pct": round(center_offset_pct, 1),
        "blur_score": round(blur_score, 1),
        "brightness": round(brightness, 1),
        "rotation_deg": round(rotation_deg, 1),
        "zones": zones_list
    }

    # 6. Hierarchical Coaching Logic
    # A. Check Sharpness / Motion Blur
    if blur_score < min_blur_var:
        return {
            **base_payload,
            "status": "HOLD_STEADY",
            "message": "Hold phone steady — motion detected.",
            "color": "#ef4444",
            "can_autocapture": False,
        }

    # B. Check Glare / Overexposure
    if glare_fraction > max_glare_fraction or brightness > max_brightness:
        return {
            **base_payload,
            "status": "REDUCE_GLARE",
            "message": "Glare / overexposure detected. Tilt slightly.",
            "color": "#f59e0b",
            "can_autocapture": False,
        }

    # C. Check Framing Distance
    if coverage_pct < (min_coverage * 100.0):
        return {
            **base_payload,
            "status": "MOVE_CLOSER",
            "message": "Move closer to badge.",
            "color": "#38bdf8",
            "can_autocapture": False,
        }

    if coverage_pct > (max_coverage * 100.0):
        return {
            **base_payload,
            "status": "MOVE_BACK",
            "message": "Too close. Step back slightly.",
            "color": "#38bdf8",
            "can_autocapture": False,
        }

    # D. Check Centering
    if center_offset_pct > 32.0:
        return {
            **base_payload,
            "status": "CENTER_BADGE",
            "message": "Center the badge in the frame.",
            "color": "#f59e0b",
            "can_autocapture": False,
        }

    # E. All Conditions Met -> Perfect Alignment
    return {
        **base_payload,
        "status": "PERFECT_ALIGNMENT",
        "message": "Perfect alignment! Hold still...",
        "color": "#10b981",
        "can_autocapture": True,
    }


