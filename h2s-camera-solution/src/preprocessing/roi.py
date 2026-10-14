"""
Badge Detection, Perspective Correction, and Region-of-Interest (ROI) Segmentation.
Identifies badge boundaries, corrects optical perspective warp, and extracts the
chemical sensing zone, reference calibration swatches, and shelf-life expiry patch.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger("CV_ROI")


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)
        
    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)
        
    def erode(self, margin_fraction: float = 0.12) -> "BoundingBox":
        """Apply inner margin erosion to eliminate edge ink bleed and boundary shadows."""
        mx = int(self.width * margin_fraction)
        my = int(self.height * margin_fraction)
        return BoundingBox(
            x1=self.x1 + mx,
            y1=self.y1 + my,
            x2=self.x2 - mx,
            y2=self.y2 - my
        )


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order coordinates consistently: [top-left, top-right, bottom-right, bottom-left].
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]       # Top-left has smallest sum
    rect[2] = pts[np.argmax(s)]       # Bottom-right has largest sum
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]    # Top-right has smallest difference
    rect[3] = pts[np.argmax(diff)]    # Bottom-left has largest difference
    return rect


def _quadrilateral_score(ordered: np.ndarray, contour_area: float, frame_area: float, is_portrait: bool = False) -> float:
    """Score a candidate using coverage, shape, and corner-angle quality."""
    edge_lengths = [
        np.linalg.norm(ordered[(index + 1) % 4] - ordered[index])
        for index in range(4)
    ]
    width = (edge_lengths[0] + edge_lengths[2]) / 2.0
    height = (edge_lengths[1] + edge_lengths[3]) / 2.0
    aspect = width / max(height, 1.0)
    effective_aspect = (1.0 / aspect) if is_portrait else aspect
    coverage = contour_area / max(frame_area, 1.0)
    aspect_score = 1.0 - min(abs(effective_aspect - 1.5) / 1.5, 1.0)
    return float(np.clip(0.55 * min(coverage / 0.5, 1.0) + 0.45 * aspect_score, 0.0, 1.0))


def _extract_quad_candidate(
    contour: np.ndarray,
    min_area: float,
    max_area: float,
    frame_area: float
) -> Tuple[Optional[np.ndarray], float]:
    """Try multiple polygon approximation epsilons and minAreaRect to extract a 4-point badge quadrilateral."""
    area = cv2.contourArea(contour)
    if area < min_area or area > max_area:
        return None, 0.0

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None, 0.0

    # Try multiple approximation tolerances
    for eps_factor in (0.012, 0.02, 0.03, 0.045, 0.06):
        approx = cv2.approxPolyDP(contour, eps_factor * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2)
            ordered = order_points(pts)
            width = (np.linalg.norm(ordered[1] - ordered[0]) + np.linalg.norm(ordered[2] - ordered[3])) / 2.0
            height = (np.linalg.norm(ordered[3] - ordered[0]) + np.linalg.norm(ordered[2] - ordered[1])) / 2.0
            aspect = width / max(height, 1.0)

            # Landscape orientation (1.0 <= aspect <= 2.3)
            if 1.0 <= aspect <= 2.3:
                score = _quadrilateral_score(ordered, area, frame_area, is_portrait=False)
                return ordered, score

            # Portrait orientation (0.45 <= aspect <= 0.95): Rotate points 90 deg clockwise to normalize to landscape
            elif 0.45 <= aspect <= 0.95:
                # [TL, TR, BR, BL] -> [BL, TL, TR, BR] (rotate to horizontal)
                rotated = np.array([ordered[3], ordered[0], ordered[1], ordered[2]], dtype=np.float32)
                score = _quadrilateral_score(rotated, area, frame_area, is_portrait=True)
                return rotated, score

    # Fallback for rounded corners or notched card holders: minAreaRect
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area >= min_area:
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        box_area = cv2.contourArea(box)
        if box_area > 0 and (hull_area / box_area) >= 0.70:
            ordered = order_points(box.astype(np.float32))
            width = (np.linalg.norm(ordered[1] - ordered[0]) + np.linalg.norm(ordered[2] - ordered[3])) / 2.0
            height = (np.linalg.norm(ordered[3] - ordered[0]) + np.linalg.norm(ordered[2] - ordered[1])) / 2.0
            aspect = width / max(height, 1.0)

            if 1.0 <= aspect <= 2.3:
                score = _quadrilateral_score(ordered, hull_area, frame_area, is_portrait=False) * 0.92
                return ordered, score
            elif 0.45 <= aspect <= 0.95:
                rotated = np.array([ordered[3], ordered[0], ordered[1], ordered[2]], dtype=np.float32)
                score = _quadrilateral_score(rotated, hull_area, frame_area, is_portrait=True) * 0.92
                return rotated, score

    return None, 0.0


def detect_badge_boundary_with_confidence(
    image: np.ndarray
) -> Tuple[Optional[np.ndarray], float]:
    """
    Robust multi-pass detector returning the strongest badge quadrilateral and confidence score.
    Employs adaptive multi-threshold Canny, Otsu thresholding, and morphological filtering.
    """
    h, w = image.shape[:2]
    frame_area = float(h * w)
    min_area = frame_area * 0.04
    max_area = frame_area * 0.98

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    v = float(np.median(blurred))

    # Multiple edge/mask passes for high recall across different lighting & backgrounds
    masks = []
    
    # Pass 1: Auto-Canny with median-derived thresholds
    low_th = int(max(15, 0.60 * v))
    high_th = int(min(240, 1.35 * v))
    edges_auto = cv2.Canny(blurred, low_th, high_th)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_auto = cv2.morphologyEx(edges_auto, cv2.MORPH_CLOSE, kernel3, iterations=2)
    masks.append(cv2.dilate(closed_auto, kernel3, iterations=1))

    # Pass 2: High contrast Canny (30, 150)
    edges_std = cv2.Canny(blurred, 30, 150)
    closed_std = cv2.morphologyEx(edges_std, cv2.MORPH_CLOSE, kernel3, iterations=2)
    masks.append(cv2.dilate(closed_std, kernel3, iterations=1))

    # Pass 3: Otsu binary thresholding (for card foreground segmentation)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    otsu_closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel5, iterations=2)
    masks.append(otsu_closed)

    # Pass 4: Adaptive Gaussian thresholding
    adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 4)
    adaptive_closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel3, iterations=2)
    masks.append(adaptive_closed)

    best_candidate = None
    best_score = 0.0

    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cand, score = _extract_quad_candidate(contour, min_area, max_area, frame_area)
            if cand is not None and score > best_score:
                best_candidate = cand
                best_score = score

    return best_candidate, best_score


def detect_badge_boundary(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect the 4-corner quadrilateral boundary of the badge from a camera photo.
    """
    corners, _ = detect_badge_boundary_with_confidence(image)
    return corners


def warp_to_canonical_badge(
    image: np.ndarray,
    corners: Optional[np.ndarray] = None,
    target_width: int = 600,
    target_height: int = 400
) -> Tuple[np.ndarray, bool]:
    """
    Rectify input image into a standardized 600x400 canonical badge perspective.
    Returns (warped_badge, is_auto_aligned).
    """
    if corners is None:
        corners = detect_badge_boundary(image)
        
    if corners is not None:
        dst_pts = np.array([
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1]
        ], dtype=np.float32)
        
        matrix = cv2.getPerspectiveTransform(corners, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (target_width, target_height))
        logger.info("Badge successfully detected and perspective-aligned.")
        return warped, True
    else:
        # Preserve the badge aspect ratio when the user has already framed it.
        logger.warning("Automatic badge corner detection did not find a 4-point quadrilateral. Using aspect-preserving fallback.")
        source_h, source_w = image.shape[:2]
        target_ratio = target_width / target_height
        source_ratio = source_w / max(source_h, 1)
        if source_ratio > target_ratio:
            crop_width = int(source_h * target_ratio)
            start = max((source_w - crop_width) // 2, 0)
            image = image[:, start:start + crop_width]
        elif source_ratio < target_ratio:
            crop_height = int(source_w / target_ratio)
            start = max((source_h - crop_height) // 2, 0)
            image = image[start:start + crop_height, :]
        resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        return resized, False


def extract_badge_rois(
    aligned_badge: np.ndarray,
    apply_erosion: bool = True
) -> Dict[str, Any]:
    """
    Extract canonical sub-regions (sensing zone, reference swatches, expiry patch)
    from a standardized 600x400 badge image.
    """
    h, w = aligned_badge.shape[:2]
    if (w, h) != (600, 400):
        aligned_badge = cv2.resize(aligned_badge, (600, 400))

    # Standard proportional bounding boxes
    raw_boxes = {
        "sensing_patch": BoundingBox(x1=55, y1=90, x2=255, y2=330),
        "ref_white": BoundingBox(x1=295, y1=90, x2=375, y2=140),
        "ref_light_gray": BoundingBox(x1=295, y1=150, x2=375, y2=200),
        "ref_med_gray": BoundingBox(x1=295, y1=210, x2=375, y2=260),
        "ref_dark": BoundingBox(x1=295, y1=270, x2=375, y2=320),
        "expiry_patch": BoundingBox(x1=418, y1=85, x2=575, y2=245)
    }

    rois = {}
    for name, bbox in raw_boxes.items():
        box_to_use = bbox.erode(0.12) if (apply_erosion and name != "expiry_patch") else bbox
        if box_to_use.x1 < 0 or box_to_use.y1 < 0 or box_to_use.x2 > w or box_to_use.y2 > h:
            raise ValueError(f"ROI '{name}' is outside the aligned badge bounds.")
        crop = aligned_badge[box_to_use.y1:box_to_use.y2, box_to_use.x1:box_to_use.x2].copy()
        if crop.size == 0:
            raise ValueError(f"ROI '{name}' is empty after extraction.")
        
        rois[name] = {
            "image": crop,
            "bbox_raw": bbox,
            "bbox_eroded": box_to_use,
            "shape": crop.shape
        }

    return rois


def draw_roi_overlay(
    aligned_badge: np.ndarray,
    rois: Dict[str, Any]
) -> np.ndarray:
    """
    Draw visual bounding box annotations and labels over the badge for visual verification.
    """
    annotated = aligned_badge.copy()
    
    colors = {
        "sensing_patch": (0, 220, 0),
        "ref_white":      (245, 130, 0),
        "ref_light_gray": (245, 130, 0),
        "ref_med_gray":   (245, 130, 0),
        "ref_dark":       (245, 130, 0),
        "expiry_patch":   (0, 140, 255)
    }

    _labels = {
        "sensing_patch": "SENSING (PbAc)",
        "ref_white":      "CAL L1",
        "ref_light_gray": "CAL L2",
        "ref_med_gray":   "CAL L3",
        "ref_dark":       "CAL L4",
        "expiry_patch":   "QR BADGE"
    }

    for name, data in rois.items():
        box = data["bbox_raw"]
        eroded = data["bbox_eroded"]
        color = colors.get(name, (0, 255, 0))
        cv2.rectangle(annotated, (box.x1, box.y1), (box.x2, box.y2), color, 1)
        cv2.rectangle(annotated, (eroded.x1, eroded.y1), (eroded.x2, eroded.y2), color, 2)
        label = _labels.get(name, name.upper())
        cv2.putText(annotated, label, (box.x1, max(15, box.y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    return annotated
