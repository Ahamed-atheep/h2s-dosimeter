import cv2
from pathlib import Path
from src.ml.predict import DosimeterPredictor

predictor = DosimeterPredictor()
out_dir = Path(r"e:\Sih\h2s-camera-solution\images\test\clean_badges_12_15_20ppm")

for ppm in [12, 15, 20]:
    matches = list(out_dir.glob(f"badge_{ppm:02d}ppm*.png"))
    if not matches:
        continue
    img_path = matches[0]
    img = cv2.imread(str(img_path))
    res, pipe = predictor.predict_from_image(img, temp_c=25.0, humidity_pct=50.0, exposure_time_min=480.0)
    feats = pipe["color_features"]
    print(f"=== BADGE {ppm} PPM ===")
    print(f"  Path: {img_path.name}")
    print(f"  Detected RGB: R={feats['R_mean']:.1f}, G={feats['G_mean']:.1f}, B={feats['B_mean']:.1f}")
    print(f"  Predicted Dose: {res.cumulative_dose_ppm_h:.2f} ppm*h")
    print(f"  Risk Level: {res.risk_level}")
    print(f"  TWA (8hr): {res.twa_8h_equivalent_ppm:.2f} ppm")
    print(f"  Confidence Interval: [{res.confidence_interval_low:.2f}, {res.confidence_interval_high:.2f}] ppm*h")
    print(f"  Is Valid: {res.is_badge_valid}\n")
