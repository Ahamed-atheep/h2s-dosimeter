"""
Simulate camera variance: same badge, different conditions.
Shows exactly how much prediction varies and from which stage.
"""
import cv2, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.ml.predict import DosimeterPredictor
from src.utils.badge_generator import generate_test_badge_image

predictor = DosimeterPredictor()
TRUE_DOSE = 10.0

# Generate clean canonical badge
base_img, _ = generate_test_badge_image(
    dose_ppm_h=TRUE_DOSE, temp_c=28.0, humidity_pct=55.0,
    lighting="daylight", tilt_angle=0.0, is_expired=False,
    badge_id="H2S-B21-00510", batch_id="B21"
)

def predict(img, label):
    try:
        res, pipe = predictor.predict_from_image(img, temp_c=28.0, humidity_pct=55.0, exposure_time_min=240.0)
        f = pipe["color_features"]
        norm = pipe["normalization_diagnostics"]
        print(f"  {label:35s} | pred={res.cumulative_dose_ppm_h:6.2f} | "
              f"B={f['B_mean']:6.1f} R={f['R_mean']:6.1f} dE={f['delta_E']:5.1f} | "
              f"conf={pipe['alignment_confidence']:.2f} | "
              f"slopes={norm['channel_gains_bgr']}")
    except Exception as e:
        print(f"  {label:35s} | ERROR: {e}")

print(f"True dose: {TRUE_DOSE} ppm*h\n")
print(f"  {'Condition':35s} | {'pred':6s} | {'B_mean':6s} {'R_mean':6s} {'dE':5s} | conf | slopes")
print("  " + "-"*100)

# 1. Clean baseline
predict(base_img, "Clean canonical (baseline)")

# 2. Brightness variations (phone in different lighting)
for factor, lbl in [(0.5,"Dark room (0.5x)"), (0.7,"Dim light (0.7x)"),
                     (1.3,"Bright light (1.3x)"), (1.6,"Overexposed (1.6x)")]:
    img = np.clip(base_img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    predict(img, lbl)

# 3. Color temperature shifts (warm/cool ambient light)
for shift, lbl in [
    ((20, 0, -20),  "Warm light (+R -B)"),
    ((-20, 0, 20),  "Cool light (-R +B)"),
    ((0, 15, 0),    "Green tint (+G)"),
]:
    img = base_img.astype(np.int16)
    img[:,:,0] = np.clip(img[:,:,0] + shift[2], 0, 255)  # B
    img[:,:,1] = np.clip(img[:,:,1] + shift[1], 0, 255)  # G
    img[:,:,2] = np.clip(img[:,:,2] + shift[0], 0, 255)  # R
    predict(img.astype(np.uint8), lbl)

# 4. JPEG compression (phone saves at different quality)
for q, lbl in [(95,"JPEG q=95"), (80,"JPEG q=80"), (60,"JPEG q=60"), (40,"JPEG q=40")]:
    _, buf = cv2.imencode(".jpg", base_img, [cv2.IMWRITE_JPEG_QUALITY, q])
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    predict(img, lbl)

# 5. Blur (camera shake / out of focus)
for k, lbl in [(3,"Slight blur (k=3)"), (7,"Moderate blur (k=7)"), (15,"Heavy blur (k=15)")]:
    img = cv2.GaussianBlur(base_img, (k, k), 0)
    predict(img, lbl)

# 6. Perspective tilt (phone not held flat)
for tilt, lbl in [(5.0,"Tilt 5deg"), (10.0,"Tilt 10deg"), (15.0,"Tilt 15deg")]:
    img, _ = generate_test_badge_image(
        dose_ppm_h=TRUE_DOSE, temp_c=28.0, humidity_pct=55.0,
        lighting="daylight", tilt_angle=tilt, is_expired=False,
        badge_id="H2S-B21-00510", batch_id="B21"
    )
    predict(img, f"Tilt {tilt}deg")

# 7. Distance (badge too small in frame)
for scale, lbl in [(0.6,"Badge 60% of frame"), (0.4,"Badge 40% of frame"), (0.25,"Badge 25% of frame")]:
    h, w = base_img.shape[:2]
    small = cv2.resize(base_img, (int(w*scale), int(h*scale)))
    canvas = np.ones((h*2, w*2, 3), dtype=np.uint8) * 160
    y = (h*2 - small.shape[0])//2
    x = (w*2 - small.shape[1])//2
    canvas[y:y+small.shape[0], x:x+small.shape[1]] = small
    predict(canvas, lbl)
