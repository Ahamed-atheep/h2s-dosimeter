"""
Compares two paths:
  A) Direct file load  (what test_whatsapp_image.py does)
  B) base64 encode -> decode -> cv2.imdecode  (what the mobile app + backend does)

If results differ, the mobile path has a lossy encoding issue.
"""
import cv2
import base64
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IMAGE_PATH = r"D:\Downloads\WhatsApp Image 2026-09-07 at 12.00.36 PM.jpeg"

from src.ml.predict import DosimeterPredictor
predictor = DosimeterPredictor()

# ── PATH A: Direct imread (test_whatsapp_image.py) ──────────────────────────
img_direct = cv2.imread(IMAGE_PATH)
res_a, pipe_a = predictor.predict_from_image(img_direct, temp_c=28.0, humidity_pct=55.0, exposure_time_min=240.0)

feats_a = pipe_a["color_features"]
print("=== PATH A: Direct imread (script) ===")
print(f"  B_mean={feats_a['B_mean']}  R_mean={feats_a['R_mean']}  G_mean={feats_a['G_mean']}")
print(f"  delta_E={feats_a['delta_E']}  ratio_GB={feats_a['ratio_GB']}")
print(f"  Dose: {res_a.cumulative_dose_ppm_h} ppm*h  Risk: {res_a.risk_level}")

# ── PATH B: JPEG encode -> base64 -> decode (mobile app path) ───────────────
# Step 1: encode to JPEG at quality=92 (what expo-image-picker uses)
_, buf_92 = cv2.imencode(".jpg", img_direct, [cv2.IMWRITE_JPEG_QUALITY, 92])
b64_92 = base64.b64encode(buf_92).decode("utf-8")
img_b_92 = cv2.imdecode(np.frombuffer(base64.b64decode(b64_92), np.uint8), cv2.IMREAD_COLOR)
res_b92, pipe_b92 = predictor.predict_from_image(img_b_92, temp_c=28.0, humidity_pct=55.0, exposure_time_min=240.0)
feats_b92 = pipe_b92["color_features"]
print("\n=== PATH B1: JPEG q=92 encode/decode (gallery pick) ===")
print(f"  B_mean={feats_b92['B_mean']}  R_mean={feats_b92['R_mean']}  G_mean={feats_b92['G_mean']}")
print(f"  delta_E={feats_b92['delta_E']}  ratio_GB={feats_b92['ratio_GB']}")
print(f"  Dose: {res_b92.cumulative_dose_ppm_h} ppm*h  Risk: {res_b92.risk_level}")

# Step 2: encode to JPEG at quality=65 (what the live frame analysis uses — 360x270 downscale)
img_small = cv2.resize(img_direct, (360, 270))
_, buf_65 = cv2.imencode(".jpg", img_small, [cv2.IMWRITE_JPEG_QUALITY, 65])
b64_65 = base64.b64encode(buf_65).decode("utf-8")
img_b_65 = cv2.imdecode(np.frombuffer(base64.b64decode(b64_65), np.uint8), cv2.IMREAD_COLOR)
res_b65, pipe_b65 = predictor.predict_from_image(img_b_65, temp_c=28.0, humidity_pct=55.0, exposure_time_min=240.0)
feats_b65 = pipe_b65["color_features"]
print("\n=== PATH B2: 360x270 JPEG q=65 (live frame — wrong, should NOT be used for scan) ===")
print(f"  B_mean={feats_b65['B_mean']}  R_mean={feats_b65['R_mean']}  G_mean={feats_b65['G_mean']}")
print(f"  delta_E={feats_b65['delta_E']}  ratio_GB={feats_b65['ratio_GB']}")
print(f"  Dose: {res_b65.cumulative_dose_ppm_h} ppm*h  Risk: {res_b65.risk_level}")

# Step 3: encode to PNG lossless (what skipProcessing=true + quality=1.0 gives)
_, buf_png = cv2.imencode(".png", img_direct)
b64_png = base64.b64encode(buf_png).decode("utf-8")
img_b_png = cv2.imdecode(np.frombuffer(base64.b64decode(b64_png), np.uint8), cv2.IMREAD_COLOR)
res_bpng, pipe_bpng = predictor.predict_from_image(img_b_png, temp_c=28.0, humidity_pct=55.0, exposure_time_min=240.0)
feats_bpng = pipe_bpng["color_features"]
print("\n=== PATH B3: PNG lossless (skipProcessing=true equivalent) ===")
print(f"  B_mean={feats_bpng['B_mean']}  R_mean={feats_bpng['R_mean']}  G_mean={feats_bpng['G_mean']}")
print(f"  delta_E={feats_bpng['delta_E']}  ratio_GB={feats_bpng['ratio_GB']}")
print(f"  Dose: {res_bpng.cumulative_dose_ppm_h} ppm*h  Risk: {res_bpng.risk_level}")

print("\n=== DELTA SUMMARY (vs PATH A) ===")
for label, feats, res in [("q=92 JPEG", feats_b92, res_b92), ("360x270 q=65", feats_b65, res_b65), ("PNG lossless", feats_bpng, res_bpng)]:
    d_dose = abs(res.cumulative_dose_ppm_h - res_a.cumulative_dose_ppm_h)
    d_B    = abs(feats["B_mean"] - feats_a["B_mean"])
    print(f"  {label:20s}  |  B_mean diff={d_B:.2f}  |  Dose diff={d_dose:.3f} ppm*h")
