"""
Generate 20 full badge images using the pipeline-compatible 600x400 badge template
with real lead acetate sensing zone colors (warm tan -> deep brown).

Badge layout matches badge_generator.py exactly:
- Sensing zone: x[55,255] y[90,330]
- CAL SCALE: warm-tan lead acetate swatches matching illumination.py CALIBRATED_TARGETS_RGB
- QR zone: x[418,575] y[85,245]

15 VALID badges: 5-19 ppm (1 ppm steps)
5 EXPIRED badges: 5, 8, 12, 15, 18 ppm
"""
import json
import cv2
import numpy as np
import qrcode
import os
from datetime import datetime, timedelta
from pathlib import Path

OUT_DIR = Path(__file__).parent / "images" / "test" / "real_color_badges_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRINT_DARKEN = 20  # printer compensation

# Real lead acetate color anchors (R, G, B) — warm tan to dark brown
ANCHORS = [
    (0.0,  210, 195, 155),
    (3.0,  198, 178, 138),
    (5.0,  182, 158, 112),
    (7.0,  168, 140,  90),
    (10.0, 150, 118,  68),
    (13.0, 132,  98,  50),
    (15.0, 115,  82,  40),
    (18.0,  95,  65,  30),
    (20.0,  80,  52,  22),
    (30.0,  58,  38,  15),
]

# CAL SCALE swatches: warm-tan lead acetate family matching illumination.py CALIBRATED_TARGETS_RGB
# ref_white:      RGB (200,185,145) -> BGR (145,185,200)
# ref_light_gray: RGB (161,133, 83) -> BGR ( 83,133,161)
# ref_med_gray:   RGB (113, 81, 39) -> BGR ( 39, 81,113)
# ref_dark:       RGB ( 59, 39, 16) -> BGR ( 16, 39, 59)
CAL_SWATCHES = [
    (90,  140, (145, 185, 200), "L1"),
    (150, 200, ( 83, 133, 161), "L2"),
    (210, 260, ( 39,  81, 113), "L3"),
    (270, 320, ( 16,  39,  59), "L4"),
]


def sensing_color_bgr(dose: float) -> tuple:
    """Interpolate real lead acetate BGR color for given dose."""
    for i in range(len(ANCHORS) - 1):
        d0, r0, g0, b0 = ANCHORS[i]
        d1, r1, g1, b1 = ANCHORS[i + 1]
        if d0 <= dose <= d1:
            t = (dose - d0) / (d1 - d0)
            R = int(r0 + t * (r1 - r0)) - PRINT_DARKEN
            G = int(g0 + t * (g1 - g0)) - PRINT_DARKEN
            B = int(b0 + t * (b1 - b0)) - PRINT_DARKEN
            return (max(5, B), max(8, G), max(10, R))
    d, r, g, b = ANCHORS[-1]
    return (max(5, b - PRINT_DARKEN), max(8, g - PRINT_DARKEN), max(10, r - PRINT_DARKEN))


def make_qr(payload: dict, box_size: int = 3) -> np.ndarray:
    qr = qrcode.QRCode(version=None, error_correction=qrcode.constants.ERROR_CORRECT_M,
                       box_size=box_size, border=1)
    qr.add_data(json.dumps(payload))
    qr.make(fit=True)
    pil = qr.make_image(fill_color="black", back_color="white")
    return np.array(pil.convert("RGB"))[:, :, ::-1]


def render_badge(dose: float, is_expired: bool,
                 badge_id: str, batch_id: str,
                 cal_version: str = "CAL-03",
                 width: int = 600, height: int = 400) -> np.ndarray:
    """
    Render pipeline-compatible badge at 600x400 matching badge_generator.py layout exactly.
    Sensing zone uses real lead acetate color.
    CAL SCALE uses warm-tan swatches matching illumination.py CALIBRATED_TARGETS_RGB.
    """
    now = datetime.utcnow().date()
    mfg_date = (now - timedelta(days=5)).strftime("%Y-%m-%d")
    if is_expired:
        exp_date = (now - timedelta(days=20)).strftime("%Y-%m-%d")
    else:
        exp_date = (now + timedelta(days=90)).strftime("%Y-%m-%d")

    qr_payload = {
        "badge_id": badge_id,
        "batch_id": batch_id,
        "manufactured_date": mfg_date,
        "expiry_date": exp_date,
        "calibration_version": cal_version
    }

    # ── Canvas ───────────────────────────────────────────────────────────────
    badge = np.ones((height, width, 3), dtype=np.uint8) * 242

    # Outer border
    border_thickness = 12
    cv2.rectangle(badge, (border_thickness // 2, border_thickness // 2),
                  (width - border_thickness // 2, height - border_thickness // 2),
                  (25, 25, 25), border_thickness)

    # Corner fiducials — matches badge_generator.py exactly
    marker_size = 28
    for pt in [(18, 18), (width - 18 - marker_size, 18),
               (18, height - 18 - marker_size), (width - 18 - marker_size, height - 18 - marker_size)]:
        cv2.rectangle(badge, pt, (pt[0] + marker_size, pt[1] + marker_size), (20, 20, 20), -1)
        cv2.rectangle(badge, (pt[0] + 6, pt[1] + 6),
                      (pt[0] + marker_size - 6, pt[1] + marker_size - 6), (240, 240, 240), -1)
        cv2.circle(badge, (pt[0] + marker_size // 2, pt[1] + marker_size // 2), 4, (20, 20, 20), -1)

    # Header
    cv2.putText(badge, "H2S PASSIVE DOSIMETER",
                (150, 45), cv2.FONT_HERSHEY_DUPLEX, 0.65, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(badge, "DGMS / OISD COMPLIANT - OPTICAL READOUT",
                (160, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1, cv2.LINE_AA)

    # ── Sensing Zone — x[55,255] y[90,330] matching badge_generator.py ──────
    sx1, sy1, sx2, sy2 = 55, 90, 255, 330
    cv2.rectangle(badge, (sx1 - 3, sy1 - 3), (sx2 + 3, sy2 + 3), (120, 120, 120), 2)
    cv2.putText(badge, "SENSING ZONE (PbAc)", (sx1 + 15, sy1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (60, 60, 60), 1, cv2.LINE_AA)

    B, G, R = sensing_color_bgr(dose)
    patch = np.zeros((sy2 - sy1, sx2 - sx1, 3), dtype=np.float32)
    patch[:, :, 0] = B
    patch[:, :, 1] = G
    patch[:, :, 2] = R
    patch += np.random.normal(0, 1.8, patch.shape)
    badge[sy1:sy2, sx1:sx2] = np.clip(patch, 0, 255).astype(np.uint8)

    # ── CAL SCALE — x[295,375] warm-tan swatches matching illumination.py ───
    ref_x1, ref_x2 = 295, 375
    cv2.putText(badge, "CAL SCALE", (ref_x1 - 5, sy1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (60, 60, 60), 1, cv2.LINE_AA)
    for y1, y2, col, lbl in CAL_SWATCHES:
        cv2.rectangle(badge, (ref_x1, y1), (ref_x2, y2), col, -1)
        cv2.rectangle(badge, (ref_x1, y1), (ref_x2, y2), (90, 90, 90), 1)
        cv2.putText(badge, lbl, (ref_x2 + 5, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 80), 1, cv2.LINE_AA)

    # ── QR Code — x[418,575] y[85,...] matching badge_generator.py ──────────
    qr_img = make_qr(qr_payload, box_size=3)
    qh, qw = qr_img.shape[:2]
    qx, qy = 418, 85
    if qy + qh > height - 20:
        scale = (height - 20 - qy) / qh
        qr_img = cv2.resize(qr_img, (int(qw * scale), int(qh * scale)),
                            interpolation=cv2.INTER_NEAREST)
        qh, qw = qr_img.shape[:2]
    badge[qy:qy + qh, qx:qx + qw] = qr_img
    cv2.rectangle(badge, (qx - 1, qy - 1), (qx + qw + 1, qy + qh + 1), (60, 60, 60), 1)
    cv2.putText(badge, "BADGE IDENTITY (QR)", (qx, qy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1, cv2.LINE_AA)

    # Metadata under QR
    meta_y = qy + qh + 18
    cv2.putText(badge, f"ID: {badge_id}", (qx, meta_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(badge, f"BATCH: {batch_id} | {cal_version}", (qx, meta_y + 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (60, 60, 60), 1, cv2.LINE_AA)
    exp_color = (0, 0, 180) if is_expired else (0, 140, 0)
    exp_label = f"EXP: {exp_date} ({'EXPIRED' if is_expired else 'VALID'})"
    cv2.putText(badge, exp_label, (qx, meta_y + 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, exp_color, 1, cv2.LINE_AA)

    # Footer
    footer = f"SN: {badge_id} | MFG: {mfg_date} | CAL: {cal_version}"
    cv2.putText(badge, footer, (130, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA)

    return badge


# ── Badge definitions ─────────────────────────────────────────────────────────

# 15 VALID badges: 5 ppm to 19 ppm (1 ppm steps)
valid_badges = [
    {"dose": d, "expired": False,
     "badge_id": f"H2S-B21-{500 + d:05d}",
     "batch_id": "B21", "cal": "CAL-03"}
    for d in range(5, 20)
]

# 5 EXPIRED badges
expired_badges = [
    {"dose": d, "expired": True,
     "badge_id": f"H2S-B15-{100 + d:05d}",
     "batch_id": "B15", "cal": "CAL-02"}
    for d in [5, 8, 12, 15, 18]
]

all_badges = valid_badges + expired_badges

print(f"Generating {len(valid_badges)} VALID + {len(expired_badges)} EXPIRED = {len(all_badges)} badges\n")

for b in all_badges:
    img = render_badge(
        dose=b["dose"], is_expired=b["expired"],
        badge_id=b["badge_id"], batch_id=b["batch_id"],
        cal_version=b["cal"]
    )
    status = "EXPIRED" if b["expired"] else "VALID"
    fname = f"badge_{b['dose']:02.0f}ppm_{status}_{b['badge_id']}.png"
    path = OUT_DIR / fname
    cv2.imwrite(str(path), img)
    B, G, R = sensing_color_bgr(b["dose"])
    print(f"  [{status:7s}] {b['dose']:2.0f} ppm  RGB=({R},{G},{B})  -> {fname}")

print(f"\nAll {len(all_badges)} badges saved to:\n  {OUT_DIR}")
print("\nBadges are pipeline-compatible: 600x400, warm-tan CAL SCALE, matching badge_generator.py layout.")
