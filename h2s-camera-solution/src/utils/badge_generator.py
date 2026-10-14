"""
Realistic Synthetic Badge Image Generator for H2S Passive Dosimeter.
Renders photorealistic test badge images featuring:
1. Central chemical sensing strip (darkening based on physical exposure dose)
2. 4-step printed reference color calibration scale
3. Printed QR Code carrying digital badge identity, batch, manufacturing, expiry, and calibration metadata
4. Human-readable badge serial, batch, expiry, and calibration text
5. Outer fiducial border for perspective detection
6. Simulated environmental lighting (daylight, warm 3000K, cool, shadows, camera tilt)
"""

import json
import cv2
import numpy as np
import qrcode
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
from datetime import datetime, timedelta

from src.utils.config import (
    physics_config, TEST_IMAGES_DIR, ensure_directories
)
from src.utils.logger import get_logger

logger = get_logger("Badge_Generator")


def build_badge_qr_payload(
    badge_id: str = "H2S-B17-00482",
    batch_id: str = "B17",
    manufactured_date: Optional[str] = None,
    expiry_date: Optional[str] = None,
    calibration_version: str = "CAL-03",
    is_expired: bool = False
) -> Dict[str, Any]:
    """Build standard JSON payload structure for badge QR code."""
    now = datetime.utcnow().date()
    
    if manufactured_date is None:
        if is_expired:
            manufactured_date = (now - timedelta(days=180)).strftime("%Y-%m-%d")
        else:
            manufactured_date = (now - timedelta(days=5)).strftime("%Y-%m-%d")
            
    if expiry_date is None:
        if is_expired:
            expiry_date = (now - timedelta(days=20)).strftime("%Y-%m-%d")
        else:
            expiry_date = (now + timedelta(days=90)).strftime("%Y-%m-%d")
            
    return {
        "badge_id": badge_id,
        "batch_id": batch_id,
        "manufactured_date": manufactured_date,
        "expiry_date": expiry_date,
        "calibration_version": calibration_version
    }


def render_canonical_badge_with_payload(
    dose_ppm_h: float = 10.0,
    temp_c: float = 25.0,
    humidity_pct: float = 50.0,
    is_expired: bool = False,
    badge_id: str = "H2S-B17-00482",
    batch_id: str = "B17",
    manufactured_date: Optional[str] = None,
    expiry_date: Optional[str] = None,
    calibration_version: str = "CAL-03",
    width: int = 600,
    height: int = 400
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Render an upright canonical badge image and return (badge_bgr_image, qr_payload_dict).
    """
    # Create clean badge background (off-white industrial plastic substrate)
    badge = np.ones((height, width, 3), dtype=np.uint8) * 242
    
    # 1. Outer high-contrast black registration border (for contour detection)
    border_thickness = 12
    cv2.rectangle(badge, (border_thickness//2, border_thickness//2), 
                  (width - border_thickness//2, height - border_thickness//2), (25, 25, 25), border_thickness)
    
    # Corner registration markers (fiducials)
    marker_size = 28
    for pt in [(18, 18), (width - 18 - marker_size, 18), 
               (18, height - 18 - marker_size), (width - 18 - marker_size, height - 18 - marker_size)]:
        cv2.rectangle(badge, pt, (pt[0] + marker_size, pt[1] + marker_size), (20, 20, 20), -1)
        cv2.rectangle(badge, (pt[0] + 6, pt[1] + 6), (pt[0] + marker_size - 6, pt[1] + marker_size - 6), (240, 240, 240), -1)
        cv2.circle(badge, (pt[0] + marker_size//2, pt[1] + marker_size//2), 4, (20, 20, 20), -1)

    # Header text
    cv2.putText(badge, "H2S PASSIVE DOSIMETER", (150, 45), cv2.FONT_HERSHEY_DUPLEX, 0.65, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(badge, "DGMS / OISD COMPLIANT - OPTICAL READOUT", (160, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1, cv2.LINE_AA)

    # 2. Chemical Sensing Zone (Left-Center)
    # Proportions: x: [55, 255], y: [90, 330] -> width: 200, height: 240
    sx1, sy1, sx2, sy2 = 55, 90, 255, 330
    cv2.rectangle(badge, (sx1 - 3, sy1 - 3), (sx2 + 3, sy2 + 3), (120, 120, 120), 2)
    cv2.putText(badge, "SENSING ZONE (PbAc)", (sx1 + 15, sy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (60, 60, 60), 1, cv2.LINE_AA)
    
    # Real lead acetate color via anchor interpolation (matches gen_real_color_badges.py)
    _ANCHORS = [
        (0.0,  210, 195, 155), (3.0,  198, 178, 138), (5.0,  182, 158, 112),
        (7.0,  168, 140,  90), (10.0, 150, 118,  68), (13.0, 132,  98,  50),
        (15.0, 115,  82,  40), (18.0,  95,  65,  30), (20.0,  80,  52,  22),
        (30.0,  58,  38,  15),
    ]
    _PRINT_DARKEN = 20
    sensing_bgr = (max(5, int(_ANCHORS[-1][3]) - _PRINT_DARKEN),
                   max(8, int(_ANCHORS[-1][2]) - _PRINT_DARKEN),
                   max(10, int(_ANCHORS[-1][1]) - _PRINT_DARKEN))
    for _i in range(len(_ANCHORS) - 1):
        _d0, _r0, _g0, _b0 = _ANCHORS[_i]
        _d1, _r1, _g1, _b1 = _ANCHORS[_i + 1]
        if _d0 <= dose_ppm_h <= _d1:
            _t = (dose_ppm_h - _d0) / (_d1 - _d0)
            sensing_bgr = (
                max(5,  int(_b0 + _t * (_b1 - _b0)) - _PRINT_DARKEN),
                max(8,  int(_g0 + _t * (_g1 - _g0)) - _PRINT_DARKEN),
                max(10, int(_r0 + _t * (_r1 - _r0)) - _PRINT_DARKEN),
            )
            break
    
    # Fill sensing patch with realistic porous substrate texture & slight micro-noise
    sensing_patch = np.zeros((sy2 - sy1, sx2 - sx1, 3), dtype=np.uint8)
    for c in range(3):
        noise = np.random.normal(0, 1.8, (sy2 - sy1, sx2 - sx1))
        sensing_patch[:, :, c] = np.clip(sensing_bgr[c] + noise, 0, 255).astype(np.uint8)
    badge[sy1:sy2, sx1:sx2] = sensing_patch

    # 3. Printed Reference Color Scale — warm-tan lead acetate family
    # Matches illumination.py CALIBRATED_TARGETS_RGB and gen_real_color_badges.py CAL_SWATCHES
    # ref_white:      RGB(200,185,145) -> BGR(145,185,200)
    # ref_light_gray: RGB(161,133, 83) -> BGR( 83,133,161)
    # ref_med_gray:   RGB(113, 81, 39) -> BGR( 39, 81,113)
    # ref_dark:       RGB( 59, 39, 16) -> BGR( 16, 39, 59)
    ref_x1, ref_x2 = 295, 375
    ref_swatches = [
        ("L1", 90,  140, (145, 185, 200)),
        ("L2", 150, 200, ( 83, 133, 161)),
        ("L3", 210, 260, ( 39,  81, 113)),
        ("L4", 270, 320, ( 16,  39,  59)),
    ]

    cv2.putText(badge, "CAL SCALE", (ref_x1 - 5, sy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (60, 60, 60), 1, cv2.LINE_AA)

    for label, y1, y2, color in ref_swatches:
        cv2.rectangle(badge, (ref_x1, y1), (ref_x2, y2), color, -1)
        cv2.rectangle(badge, (ref_x1, y1), (ref_x2, y2), (90, 90, 90), 1)
        cv2.putText(badge, label, (ref_x2 + 5, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 80), 1, cv2.LINE_AA)

    # 4. QR Code & Digital Badge Identity Zone (Far Right)
    qr_payload = build_badge_qr_payload(
        badge_id=badge_id,
        batch_id=batch_id,
        manufactured_date=manufactured_date,
        expiry_date=expiry_date,
        calibration_version=calibration_version,
        is_expired=is_expired
    )
    
    # Generate QR Code image with exact integer module pixels for perfect detection
    qr_maker = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=3,
        border=1,
    )
    qr_maker.add_data(json.dumps(qr_payload))
    qr_maker.make(fit=True)
    pil_qr = qr_maker.make_image(fill_color="black", back_color="white")
    qr_np = np.array(pil_qr.convert("RGB"))[:, :, ::-1]  # Convert RGB to OpenCV BGR
    
    qr_h, qr_w = qr_np.shape[:2]
    q_x1, q_y1 = 418, 85
    q_x2, q_y2 = q_x1 + qr_w, q_y1 + qr_h
    
    cv2.putText(badge, "BADGE IDENTITY (QR)", (q_x1, q_y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1, cv2.LINE_AA)
    
    # Paste QR code with border
    badge[q_y1:q_y2, q_x1:q_x2] = qr_np
    cv2.rectangle(badge, (q_x1 - 1, q_y1 - 1), (q_x2 + 1, q_y2 + 1), (60, 60, 60), 1)

    # 5. Badge Human-Readable Metadata Text (Under QR Code)
    cv2.putText(badge, f"ID: {qr_payload['badge_id']}", (q_x1, q_y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(badge, f"BATCH: {qr_payload['batch_id']} | {qr_payload['calibration_version']}", (q_x1, q_y2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (60, 60, 60), 1, cv2.LINE_AA)
    
    exp_color = (0, 0, 180) if is_expired else (0, 140, 0)
    exp_label = f"EXP: {qr_payload['expiry_date']} ({'EXPIRED' if is_expired else 'VALID'})"
    cv2.putText(badge, exp_label, (q_x1, q_y2 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.33, exp_color, 1, cv2.LINE_AA)
    
    # 6. Serial Footer
    footer_text = f"SN: {qr_payload['badge_id']} | MFG: {qr_payload['manufactured_date']} | CAL: {qr_payload['calibration_version']}"
    cv2.putText(badge, footer_text, (130, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA)

    return badge, qr_payload


def render_canonical_badge(
    dose_ppm_h: float = 10.0,
    temp_c: float = 25.0,
    humidity_pct: float = 50.0,
    is_expired: bool = False,
    badge_id: str = "H2S-B17-00482",
    batch_id: str = "B17",
    manufactured_date: Optional[str] = None,
    expiry_date: Optional[str] = None,
    calibration_version: str = "CAL-03",
    width: int = 600,
    height: int = 400
) -> np.ndarray:
    """
    Render an upright canonical badge image of dimensions (height, width, 3) in BGR format.
    Returns the badge np.ndarray image.
    """
    badge, _ = render_canonical_badge_with_payload(
        dose_ppm_h=dose_ppm_h,
        temp_c=temp_c,
        humidity_pct=humidity_pct,
        is_expired=is_expired,
        badge_id=badge_id,
        batch_id=batch_id,
        manufactured_date=manufactured_date,
        expiry_date=expiry_date,
        calibration_version=calibration_version,
        width=width,
        height=height
    )
    return badge


def apply_photorealistic_effects(
    badge: np.ndarray,
    lighting: str = "daylight",
    add_shadow: bool = False,
    tilt_angle: float = 0.0,
    background_size: Tuple[int, int] = (600, 800)
) -> np.ndarray:
    """
    Place canonical badge onto a realistic industrial tabletop / background and apply
    color temperature shifts, uneven lighting, shadows, and camera perspective tilt.
    """
    bg_h, bg_w = background_size
    bh, bw = badge.shape[:2]
    
    # 1. Create realistic textured background
    bg = np.zeros((bg_h, bg_w, 3), dtype=np.uint8)
    bg_color = np.array([165, 175, 185], dtype=np.float32)  # slate workbench gray
    bg[:] = bg_color + np.random.normal(0, 4, (bg_h, bg_w, 3)).clip(-15, 15)
    
    # 2. Position badge in center of background
    pad_y = (bg_h - bh) // 2
    pad_x = (bg_w - bw) // 2
    
    src_pts = np.array([
        [0, 0],
        [bw - 1, 0],
        [bw - 1, bh - 1],
        [0, bh - 1]
    ], dtype=np.float32)
    
    if abs(tilt_angle) > 1e-2:
        rad = np.radians(tilt_angle)
        dx_top = int(np.sin(rad) * 45)
        dy_right = int(np.sin(rad) * 35)
        
        dst_pts = np.array([
            [pad_x + dx_top + 15, pad_y + 10],
            [pad_x + bw - 15, pad_y - dy_right + 10],
            [pad_x + bw - dx_top - 5, pad_y + bh + dy_right - 10],
            [pad_x + 5, pad_y + bh - 10]
        ], dtype=np.float32)
        
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_badge = cv2.warpPerspective(badge, matrix, (bg_w, bg_h), borderValue=(0, 0, 0))
        
        mask = np.zeros((bh, bw), dtype=np.uint8)
        mask[:] = 255
        warped_mask = cv2.warpPerspective(mask, matrix, (bg_w, bg_h), borderValue=0)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        shadow_mask = cv2.dilate(warped_mask, kernel, iterations=2)
        shadow_blur = cv2.GaussianBlur(shadow_mask.astype(np.float32), (31, 31), 0) / 255.0
        
        for c in range(3):
            bg[:, :, c] = (bg[:, :, c] * (1.0 - shadow_blur * 0.35)).astype(np.uint8)
            bg[:, :, c] = np.where(warped_mask > 0, warped_badge[:, :, c], bg[:, :, c])
            
        composite = bg
    else:
        composite = bg.copy()
        composite[pad_y:pad_y + bh, pad_x:pad_x + bw] = badge
        
    composite = composite.astype(np.float32)

    # 3. Apply Color Temperature Shift
    if lighting.lower() == "warm":
        composite[:, :, 0] *= 0.82  # Blue down
        composite[:, :, 1] *= 0.98  # Green neutral
        composite[:, :, 2] *= 1.18  # Red up
    elif lighting.lower() == "cool":
        composite[:, :, 0] *= 1.15
        composite[:, :, 1] *= 1.02
        composite[:, :, 2] *= 0.88
    elif lighting.lower() == "fluorescent":
        composite[:, :, 0] *= 0.92
        composite[:, :, 1] *= 1.12
        composite[:, :, 2] *= 0.94

    # 4. Apply Uneven Lighting / Shadow Gradient
    if add_shadow or lighting.lower() == "shadow":
        y_coords = np.linspace(1.15, 0.70, bg_h).reshape(bg_h, 1)
        x_coords = np.linspace(1.10, 0.80, bg_w).reshape(1, bg_w)
        gradient = (y_coords * x_coords)[:, :, np.newaxis]
        composite *= gradient

    # 5. Add realistic camera optical sensor noise
    sensor_noise = np.random.normal(0, 1.5, composite.shape)
    composite = np.clip(composite + sensor_noise, 0, 255).astype(np.uint8)

    return composite


def generate_test_badge_image(
    dose_ppm_h: float = 12.0,
    temp_c: float = 28.0,
    humidity_pct: float = 55.0,
    is_expired: bool = False,
    badge_id: str = "H2S-B17-00482",
    batch_id: str = "B17",
    manufactured_date: Optional[str] = None,
    expiry_date: Optional[str] = None,
    calibration_version: str = "CAL-03",
    lighting: str = "daylight",
    add_shadow: bool = False,
    tilt_angle: float = 0.0,
    output_path: Optional[Path] = None
) -> Tuple[np.ndarray, Path]:
    """
    Generate and save a realistic synthetic badge test image with an embedded QR code.
    Returns (final_image_bgr, saved_image_path).
    """
    ensure_directories()
    
    canonical, qr_payload = render_canonical_badge_with_payload(
        dose_ppm_h=dose_ppm_h,
        temp_c=temp_c,
        humidity_pct=humidity_pct,
        is_expired=is_expired,
        badge_id=badge_id,
        batch_id=batch_id,
        manufactured_date=manufactured_date,
        expiry_date=expiry_date,
        calibration_version=calibration_version
    )
    
    final_image = apply_photorealistic_effects(
        canonical,
        lighting=lighting,
        add_shadow=add_shadow,
        tilt_angle=tilt_angle
    )
    
    if output_path is None:
        status = "expired" if is_expired else "valid"
        filename = f"badge_dose_{dose_ppm_h:.1f}ppm_{lighting}_{status}.png"
        output_path = TEST_IMAGES_DIR / filename
        
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), final_image)
    
    return final_image, output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic H2S Dosimeter badge with scannable QR code.")
    parser.add_argument("--dose", type=float, default=12.0, help="Cumulative exposure dose (ppm·h)")
    parser.add_argument("--temp", type=float, default=28.0, help="Ambient temperature in °C")
    parser.add_argument("--humidity", type=float, default=55.0, help="Ambient relative humidity in %")
    parser.add_argument("--expired", action="store_true", help="Generate an expired badge")
    parser.add_argument("--badge-id", type=str, default="H2S-B17-00482", help="Badge ID")
    parser.add_argument("--batch-id", type=str, default="B17", help="Batch ID")
    parser.add_argument("--cal-version", type=str, default="CAL-03", help="Calibration profile version")
    parser.add_argument("--lighting", type=str, default="daylight", choices=["daylight", "warm", "cool", "shadow"], help="Lighting profile")
    parser.add_argument("--tilt", type=float, default=0.0, help="Perspective tilt angle in degrees")
    parser.add_argument("--output", type=str, default=None, help="Output image file path")
    parser.add_argument("--generate-suite", action="store_true", help="Generate complete suite of standard test badges")

    args = parser.parse_args()

    if args.generate_suite:
        print("[+] Generating standard test badge suite with scannable QR codes...")
        test_badges = [
            {"dose": 0.0,  "lighting": "daylight", "expired": False, "badge_id": "H2S-B17-00001", "batch_id": "B17", "name": "badge_fresh_0.0ppm.png"},
            {"dose": 4.5,  "lighting": "daylight", "expired": False, "badge_id": "H2S-B17-00482", "batch_id": "B17", "name": "badge_safe_4.5ppm.png"},
            {"dose": 18.0, "lighting": "warm",     "expired": False, "badge_id": "H2S-B18-00104", "batch_id": "B18", "name": "badge_warning_18.0ppm.png"},
            {"dose": 55.0, "lighting": "shadow",   "expired": False, "badge_id": "H2S-B19-00912", "batch_id": "B19", "name": "badge_danger_55.0ppm.png"},
            {"dose": 10.0, "lighting": "daylight", "expired": True,  "badge_id": "H2S-B12-00019", "batch_id": "B12", "name": "badge_expired_10.0ppm.png"},
        ]
        for b in test_badges:
            out_file = TEST_IMAGES_DIR / b["name"]
            _, saved = generate_test_badge_image(
                dose_ppm_h=b["dose"],
                lighting=b["lighting"],
                is_expired=b["expired"],
                badge_id=b["badge_id"],
                batch_id=b["batch_id"],
                output_path=out_file
            )
            print(f"  [OK] Created: {saved.name} (Dose: {b['dose']} ppm*h, Expired: {b['expired']})")
        print(f"\nAll badges generated in {TEST_IMAGES_DIR}")
    else:
        out_p = Path(args.output) if args.output else None
        img, saved = generate_test_badge_image(
            dose_ppm_h=args.dose,
            temp_c=args.temp,
            humidity_pct=args.humidity,
            is_expired=args.expired,
            badge_id=args.badge_id,
            batch_id=args.batch_id,
            calibration_version=args.cal_version,
            lighting=args.lighting,
            tilt_angle=args.tilt,
            output_path=out_p
        )
        print(f"[OK] Generated badge: {saved.name}")
        print(f"  - Badge ID: {args.badge_id}")
        print(f"  - Batch: {args.batch_id}")
        print(f"  - Dose: {args.dose} ppm*h")
        print(f"  - Status: {'EXPIRED' if args.expired else 'VALID'}")

