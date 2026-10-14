import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.badge_generator import generate_test_badge_image

img5, _ = generate_test_badge_image(
    dose_ppm_h=5.0, temp_c=28.0, humidity_pct=55.0,
    lighting="daylight", tilt_angle=0.0, is_expired=False,
    badge_id="H2S-B20-00550", batch_id="B20", calibration_version="CAL-03"
)

img6, _ = generate_test_badge_image(
    dose_ppm_h=6.0, temp_c=28.0, humidity_pct=55.0,
    lighting="daylight", tilt_angle=0.0, is_expired=False,
    badge_id="H2S-B20-00551", batch_id="B20", calibration_version="CAL-03"
)

print(f"5 ppm badge shape: {img5.shape}")
print(f"6 ppm badge shape: {img6.shape}")

h, w = img5.shape[:2]
gap = 24
canvas = np.ones((h + 70, w * 2 + gap * 3, 3), dtype=np.uint8) * 18

canvas[50:50+h, gap:gap+w] = img5
canvas[50:50+h, gap*2+w:gap*2+w*2] = img6

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(canvas, "5 ppm*h  [SAFE / Batch B20]",  (gap, 34),       font, 0.62, (60, 220, 60),  2, cv2.LINE_AA)
cv2.putText(canvas, "6 ppm*h  [SAFE / Batch B20]",  (gap*2+w, 34),   font, 0.62, (60, 220, 60),  2, cv2.LINE_AA)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "test", "badge_5vs6_comparison.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
ok = cv2.imwrite(out, canvas)
print(f"Saved ({ok}): {out}")
print(f"Canvas size: {canvas.shape[1]}x{canvas.shape[0]}")
