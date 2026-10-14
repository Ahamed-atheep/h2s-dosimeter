"""
Dynamic Live Camera-Based Zone Detection and Tracking Engine.
Detects and dynamically tracks badge regions in real time using OpenCV homography
and perspective projection:
1. 🟩 Chemical Sensing Zone
2. 🔵 Reference / Calibration Zones (4 individual swatches: White, Light Gray, Med Gray, Dark)
3. 🟧 QR Badge Identity Zone

Provides object-detection style bounding boxes, corner polygons, confidence scores,
and temporal smoothing for moving/rotating badges in front of the camera.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from src.preprocessing.roi import detect_badge_boundary_with_confidence, order_points
from src.utils.logger import get_logger

logger = get_logger("Zone_Detector")

# Canonical Badge Dimensions
CANONICAL_WIDTH = 600
CANONICAL_HEIGHT = 400

# Canonical Sub-Region Layouts in (600x400) Pixel Space
# Each zone is defined by (x1, y1, x2, y2), category, display name, and UI color
CANONICAL_ZONE_DEFS = {
    "sensing_zone": {
        "name": "Sensing Zone",
        "short_label": "SENSING",
        "category": "chemical_sensing",
        "box": (55, 90, 255, 330),
        "color_hex": "#10b981",       # Bright Emerald Green
        "color_bgr": (0, 220, 0),      # OpenCV Green
        "icon": "🟩",
        "description": "Lead-acetate chemical reaction substrate"
    },
    "ref_white": {
        "name": "Ref White (L1)",
        "short_label": "REF L1",
        "category": "calibration_reference",
        "box": (295, 90, 375, 140),
        "color_hex": "#3b82f6",
        "color_bgr": (245, 130, 0),
        "icon": "🔵",
        "description": "L1 fresh cream calibration reference"
    },
    "ref_light_gray": {
        "name": "Ref Light Tan (L2)",
        "short_label": "REF L2",
        "category": "calibration_reference",
        "box": (295, 150, 375, 200),
        "color_hex": "#3b82f6",
        "color_bgr": (245, 130, 0),
        "icon": "🔵",
        "description": "L2 light tan calibration reference"
    },
    "ref_med_gray": {
        "name": "Ref Medium Brown (L3)",
        "short_label": "REF L3",
        "category": "calibration_reference",
        "box": (295, 210, 375, 260),
        "color_hex": "#3b82f6",
        "color_bgr": (245, 130, 0),
        "icon": "🔵",
        "description": "L3 medium brown calibration reference"
    },
    "ref_dark": {
        "name": "Ref Dark Brown (L4)",
        "short_label": "REF L4",
        "category": "calibration_reference",
        "box": (295, 270, 375, 320),
        "color_hex": "#3b82f6",
        "color_bgr": (245, 130, 0),
        "icon": "🔵",
        "description": "L4 dark brown calibration reference"
    },
    "qr_badge": {
        "name": "QR Badge Zone",
        "short_label": "QR BADGE",
        "category": "badge_identity_qr",
        "box": (418, 85, 575, 245),
        "color_hex": "#f59e0b",       # Amber / Orange
        "color_bgr": (0, 140, 255),    # OpenCV Orange
        "icon": "🟧",
        "description": "Digital serialization & shelf-life QR code"
    }
}


@dataclass
class DetectedZoneResult:
    """Structure for a single dynamically detected zone in the live frame."""
    id: str
    name: str
    short_label: str
    category: str
    color_hex: str
    color_bgr: Tuple[int, int, int]
    icon: str
    confidence: float
    # 4 polygon points in pixel coordinates [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
    polygon_px: List[List[int]]
    # 4 polygon points normalized to [0.0, 1.0]
    polygon_norm: List[List[float]]
    # Axis-aligned bounding box {x, y, width, height} in pixels
    bbox_px: Dict[str, int]
    # Axis-aligned bounding box normalized to [0.0, 1.0]
    bbox_norm: Dict[str, float]
    # Center coordinate (x, y)
    center_norm: List[float]
    # Mean BGR color sampled from frame inside this zone
    mean_bgr: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "short_label": self.short_label,
            "category": self.category,
            "color": self.color_hex,
            "icon": self.icon,
            "confidence": round(self.confidence, 4),
            "polygon": [[round(pt[0], 4), round(pt[1], 4)] for pt in self.polygon_norm],
            "bbox": {k: round(v, 4) for k, v in self.bbox_norm.items()},
            "center": [round(c, 4) for c in self.center_norm],
            "mean_bgr": [round(v, 1) for v in self.mean_bgr] if self.mean_bgr else None
        }


class BadgeZoneTracker:
    """
    Temporal tracking filter using Exponential Moving Average (EMA) and track persistence.
    Maintains smooth corner coordinates across successive video frames to prevent jitter
    when the badge or camera moves or rotates.
    """
    def __init__(self, smoothing_factor: float = 0.40, max_missing_frames: int = 5):
        self.smoothing_factor = float(np.clip(smoothing_factor, 0.05, 0.95))
        self.max_missing_frames = max_missing_frames
        self.smoothed_corners: Optional[np.ndarray] = None
        self.missing_count = 0
        self.tracked_confidence = 0.0
        
    def update(self, detected_corners: Optional[np.ndarray], confidence: float = 0.0) -> Tuple[Optional[np.ndarray], float]:
        """
        Update the tracker with new frame detection.
        Returns (smoothed_corners, effective_confidence).
        """
        if detected_corners is not None and confidence >= 0.20:
            ordered = order_points(detected_corners.reshape(4, 2))
            if self.smoothed_corners is None or self.missing_count >= self.max_missing_frames:
                self.smoothed_corners = ordered.astype(np.float32)
                self.tracked_confidence = confidence
            else:
                alpha = self.smoothing_factor
                self.smoothed_corners = (alpha * ordered + (1.0 - alpha) * self.smoothed_corners).astype(np.float32)
                self.tracked_confidence = float(alpha * confidence + (1.0 - alpha) * self.tracked_confidence)
            self.missing_count = 0
            return self.smoothed_corners.copy(), self.tracked_confidence
        else:
            self.missing_count += 1
            if self.smoothed_corners is not None and self.missing_count < self.max_missing_frames:
                # Retain previous position with slight decay in confidence
                self.tracked_confidence *= 0.85
                return self.smoothed_corners.copy(), self.tracked_confidence
            else:
                self.smoothed_corners = None
                self.tracked_confidence = 0.0
                return None, 0.0

    def reset(self):
        """Reset tracker state."""
        self.smoothed_corners = None
        self.missing_count = 0
        self.tracked_confidence = 0.0


class BadgeZoneDetector:
    """
    Real-time Dynamic Zone Detection Engine for H2S Dosimeter Badges.
    Uses homography perspective transformation to project canonical badge zones
    into arbitrary live camera frames under full 3D rotation, scaling, and tilt.
    """
    def __init__(self, use_temporal_smoothing: bool = True, smoothing_factor: float = 0.40):
        self.tracker = BadgeZoneTracker(smoothing_factor=smoothing_factor) if use_temporal_smoothing else None
        
        # Define 4 canonical source corners
        self.canonical_corners = np.array([
            [0.0, 0.0],
            [float(CANONICAL_WIDTH - 1), 0.0],
            [float(CANONICAL_WIDTH - 1), float(CANONICAL_HEIGHT - 1)],
            [0.0, float(CANONICAL_HEIGHT - 1)]
        ], dtype=np.float32)

    def detect_zones(
        self,
        image_bgr: np.ndarray,
        corners: Optional[np.ndarray] = None,
        confidence_override: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Detect and project all 6 badge zones in the given camera frame.
        
        Parameters
        ----------
        image_bgr : np.ndarray
            Input BGR image frame from camera or video stream.
        corners : Optional[np.ndarray]
            Pre-detected 4-point quadrilateral corners. If None, auto-detected.
        confidence_override : Optional[float]
            Optional confidence score.
            
        Returns
        -------
        Dict[str, Any]
            Structured detection dictionary containing:
            - badge_detected: bool
            - confidence: float
            - corners: [[x, y], ...] normalized
            - rotation_deg: float
            - zones: List[DetectedZoneResult]
        """
        if image_bgr is None or image_bgr.size == 0:
            return {
                "badge_detected": False,
                "confidence": 0.0,
                "corners": None,
                "rotation_deg": 0.0,
                "zones": []
            }

        h, w = image_bgr.shape[:2]
        
        # 1. Detect Badge Boundary if not provided
        if corners is None:
            raw_corners, detected_conf = detect_badge_boundary_with_confidence(image_bgr)
            # Fallback for cropped badges / close-up photos where outer contour touches image edges
            if raw_corners is None:
                aspect = float(w) / max(float(h), 1.0)
                if 1.2 <= aspect <= 1.8:
                    raw_corners = np.array([
                        [0.0, 0.0],
                        [float(w - 1), 0.0],
                        [float(w - 1), float(h - 1)],
                        [0.0, float(h - 1)]
                    ], dtype=np.float32)
                    detected_conf = 0.85
        else:
            raw_corners = corners
            detected_conf = confidence_override if confidence_override is not None else 0.90

        # 2. Apply Temporal Smoothing if enabled
        if self.tracker is not None:
            active_corners, effective_conf = self.tracker.update(raw_corners, detected_conf)
        else:
            if raw_corners is not None:
                active_corners = order_points(raw_corners.reshape(4, 2))
                effective_conf = detected_conf
            else:
                active_corners = None
                effective_conf = 0.0

        if active_corners is None or effective_conf < 0.20:
            return {
                "badge_detected": False,
                "confidence": 0.0,
                "corners": None,
                "rotation_deg": 0.0,
                "zones": []
            }

        # 3. Compute Homography Matrix: Canonical (600x400) -> Frame (W x H)
        dst_corners = active_corners.astype(np.float32)
        homography_matrix = cv2.getPerspectiveTransform(self.canonical_corners, dst_corners)

        # Compute Orientation / Rotation Angle in degrees
        top_edge = dst_corners[1] - dst_corners[0]
        angle_rad = np.arctan2(top_edge[1], top_edge[0])
        rotation_deg = float(np.degrees(angle_rad))

        # 4. Project Each Canonical Zone into Frame Coordinates
        detected_zones: List[DetectedZoneResult] = []

        for zone_id, zone_def in CANONICAL_ZONE_DEFS.items():
            x1, y1, x2, y2 = zone_def["box"]
            
            # Canonical 4 corner polygon points: [TL, TR, BR, BL]
            canonical_poly = np.array([
                [[float(x1), float(y1)]],
                [[float(x2), float(y1)]],
                [[float(x2), float(y2)]],
                [[float(x1), float(y2)]]
            ], dtype=np.float32)

            # Project to frame using Homography
            frame_poly = cv2.perspectiveTransform(canonical_poly, homography_matrix)
            poly_pts = frame_poly.reshape(-1, 2)

            # Pixel Polygon
            poly_px = [[int(round(pt[0])), int(round(pt[1]))] for pt in poly_pts]

            # Normalized Polygon [0.0, 1.0]
            poly_norm = [[float(pt[0] / max(w, 1)), float(pt[1] / max(h, 1))] for pt in poly_pts]

            # Axis-Aligned Bounding Box in Pixel Space
            xs = [pt[0] for pt in poly_px]
            ys = [pt[1] for pt in poly_px]
            min_x = max(0, min(xs))
            max_x = min(w - 1, max(xs))
            min_y = max(0, min(ys))
            max_y = min(h - 1, max(ys))
            
            bbox_w = max(1, max_x - min_x)
            bbox_h = max(1, max_y - min_y)

            bbox_px = {
                "x": min_x,
                "y": min_y,
                "width": bbox_w,
                "height": bbox_h
            }

            bbox_norm = {
                "x": float(min_x / max(w, 1)),
                "y": float(min_y / max(h, 1)),
                "width": float(bbox_w / max(w, 1)),
                "height": float(bbox_h / max(h, 1))
            }

            center_norm = [
                float((min_x + bbox_w / 2.0) / max(w, 1)),
                float((min_y + bbox_h / 2.0) / max(h, 1))
            ]

            # Sample mean color from zone region inside frame if in bounds
            mean_bgr = None
            if min_x < max_x and min_y < max_y:
                crop = image_bgr[min_y:max_y, min_x:max_x]
                if crop.size > 0:
                    mean_bgr = [float(val) for val in cv2.mean(crop)[:3]]

            # Zone Confidence: combination of badge confidence & valid boundary position
            zone_conf = float(np.clip(effective_conf * 0.98 + 0.01, 0.1, 0.99))

            detected_zones.append(DetectedZoneResult(
                id=zone_id,
                name=zone_def["name"],
                short_label=zone_def["short_label"],
                category=zone_def["category"],
                color_hex=zone_def["color_hex"],
                color_bgr=zone_def["color_bgr"],
                icon=zone_def["icon"],
                confidence=zone_conf,
                polygon_px=poly_px,
                polygon_norm=poly_norm,
                bbox_px=bbox_px,
                bbox_norm=bbox_norm,
                center_norm=center_norm,
                mean_bgr=mean_bgr
            ))

        # Normalized 4 badge corners for UI
        norm_corners = [
            [round(float(pt[0]) / max(w, 1), 4), round(float(pt[1]) / max(h, 1), 4)]
            for pt in dst_corners
        ]

        return {
            "badge_detected": True,
            "confidence": round(float(effective_conf), 4),
            "corners": norm_corners,
            "corners_px": dst_corners.astype(int).tolist(),
            "rotation_deg": round(rotation_deg, 1),
            "zones": detected_zones
        }

    def reset(self):
        """Reset temporal tracker."""
        if self.tracker is not None:
            self.tracker.reset()


def draw_zone_detections(
    image_bgr: np.ndarray,
    detection_result: Dict[str, Any],
    show_hud: bool = True,
    show_fps: Optional[float] = None
) -> np.ndarray:
    """
    Render professional, object-detection style bounding boxes, corner brackets,
    and confidence labels over the live camera frame.
    
    Parameters
    ----------
    image_bgr : np.ndarray
        Input BGR frame.
    detection_result : Dict[str, Any]
        Output dictionary from BadgeZoneDetector.detect_zones().
    show_hud : bool
        Whether to draw top HUD status bar and orientation readout.
    show_fps : Optional[float]
        Optional frame rate to display on the HUD.
        
    Returns
    -------
    np.ndarray
        Annotated BGR frame.
    """
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]

    if not detection_result.get("badge_detected", False):
        if show_hud:
            # Draw Searching HUD banner
            cv2.rectangle(annotated, (15, 15), (320, 55), (20, 20, 20), -1)
            cv2.rectangle(annotated, (15, 15), (320, 55), (0, 180, 255), 1)
            cv2.circle(annotated, (35, 35), 6, (0, 180, 255), -1)
            cv2.putText(annotated, "SEARCHING FOR BADGE...", (52, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        return annotated

    # 1. Draw Outer Badge Quadrilateral Perimeter & Fiducial Crosshairs
    corners_px = detection_result.get("corners_px")
    if corners_px and len(corners_px) == 4:
        pts = np.array(corners_px, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], isClosed=True, color=(60, 60, 60), thickness=2, lineType=cv2.LINE_AA)
        
        # Corner registration crosshairs
        for pt in corners_px:
            cx, cy = int(pt[0]), int(pt[1])
            cv2.circle(annotated, (cx, cy), 5, (0, 255, 255), -1)
            cv2.line(annotated, (cx - 10, cy), (cx + 10, cy), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.line(annotated, (cx, cy - 10), (cx, cy + 10), (0, 255, 255), 1, cv2.LINE_AA)

    # 2. Draw Each Detected Zone with Object Detection Bounding Box & HUD Tag
    zones = detection_result.get("zones", [])
    for zone in zones:
        # Support either DetectedZoneResult dataclass or dict
        if isinstance(zone, DetectedZoneResult):
            poly_px = zone.polygon_px
            color_bgr = zone.color_bgr
            label = f"{zone.short_label} [{int(zone.confidence * 100)}%]"
            bbox = zone.bbox_px
        else:
            poly_px = [[int(p[0] * w), int(p[1] * h)] for p in zone["polygon"]]
            color_bgr = CANONICAL_ZONE_DEFS.get(zone["id"], {}).get("color_bgr", (0, 255, 0))
            label = f"{zone['name'].upper()} [{int(zone['confidence'] * 100)}%]"
            bbox = {
                "x": int(zone["bbox"]["x"] * w),
                "y": int(zone["bbox"]["y"] * h),
                "width": int(zone["bbox"]["width"] * w),
                "height": int(zone["bbox"]["height"] * h)
            }

        poly_np = np.array(poly_px, dtype=np.int32).reshape((-1, 1, 2))

        # A. Semi-transparent polygon fill
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [poly_np], color_bgr)
        cv2.addWeighted(overlay, 0.14, annotated, 0.86, 0, annotated)

        # B. Polygon Bounding Outline
        cv2.polylines(annotated, [poly_np], isClosed=True, color=color_bgr, thickness=2, lineType=cv2.LINE_AA)

        # C. Targeted Corner Brackets at BBox
        bx, by, bw, bh = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        bracket_len = min(14, bw // 3, bh // 3)
        if bracket_len > 3:
            # Top-Left
            cv2.line(annotated, (bx, by), (bx + bracket_len, by), color_bgr, 3, cv2.LINE_AA)
            cv2.line(annotated, (bx, by), (bx, by + bracket_len), color_bgr, 3, cv2.LINE_AA)
            # Top-Right
            cv2.line(annotated, (bx + bw, by), (bx + bw - bracket_len, by), color_bgr, 3, cv2.LINE_AA)
            cv2.line(annotated, (bx + bw, by), (bx + bw, by + bracket_len), color_bgr, 3, cv2.LINE_AA)
            # Bottom-Left
            cv2.line(annotated, (bx, by + bh), (bx + bracket_len, by + bh), color_bgr, 3, cv2.LINE_AA)
            cv2.line(annotated, (bx, by + bh), (bx, by + bh - bracket_len), color_bgr, 3, cv2.LINE_AA)
            # Bottom-Right
            cv2.line(annotated, (bx + bw, by + bh), (bx + bw - bracket_len, by + bh), color_bgr, 3, cv2.LINE_AA)
            cv2.line(annotated, (bx + bw, by + bh), (bx + bw, by + bh - bracket_len), color_bgr, 3, cv2.LINE_AA)

        # D. High-tech Label Pill Tag
        font_scale = 0.38
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        
        tag_x1 = max(4, bx)
        tag_y1 = max(18, by - 6)
        tag_x2 = tag_x1 + text_w + 12
        tag_y2 = tag_y1 - text_h - 6

        # Draw dark tag background pill with colored border
        cv2.rectangle(annotated, (tag_x1, min(tag_y1, tag_y2)), (tag_x2, max(tag_y1, tag_y2)), (18, 18, 18), -1)
        cv2.rectangle(annotated, (tag_x1, min(tag_y1, tag_y2)), (tag_x2, max(tag_y1, tag_y2)), color_bgr, 1)
        
        # Color dot indicator
        cv2.circle(annotated, (tag_x1 + 6, min(tag_y1, tag_y2) + (abs(tag_y1 - tag_y2)) // 2), 3, color_bgr, -1)
        
        # Tag text
        cv2.putText(annotated, label, (tag_x1 + 13, max(tag_y1, tag_y2) - 4),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # 3. Top HUD Status Readout (Pose, Confidence, FPS)
    if show_hud:
        conf_pct = int(detection_result.get("confidence", 0.0) * 100)
        rot_deg = detection_result.get("rotation_deg", 0.0)
        
        hud_text = f"TARGET LOCKED: BADGE ({conf_pct}%) | ROT: {rot_deg:+.1f}deg"
        if show_fps is not None:
            hud_text += f" | {show_fps:.1f} FPS"

        cv2.rectangle(annotated, (12, 12), (min(w - 12, 440), 44), (18, 18, 18), -1)
        cv2.rectangle(annotated, (12, 12), (min(w - 12, 440), 44), (0, 220, 0), 1)
        cv2.circle(annotated, (26, 28), 5, (0, 220, 0), -1)
        cv2.putText(annotated, hud_text, (38, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

    return annotated
