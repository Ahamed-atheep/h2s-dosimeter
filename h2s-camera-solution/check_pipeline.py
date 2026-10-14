import cv2
import pathlib
from src.utils.badge_generator import generate_test_badge_image
from src.preprocessing.color_features import process_badge_image_pipeline
from src.ml.predict import DosimeterPredictor

out = pathlib.Path("images/test/debug_check")
out.mkdir(parents=True, exist_ok=True)

predictor = DosimeterPredictor()

cases = [
    (4.5,  "safe",    False),
    (18.0, "warning", False),
    (55.0, "danger",  False),
    (10.0, "expired", True),
]

for dose, label, expired in cases:
    img, _ = generate_test_badge_image(
        dose_ppm_h=dose, lighting="daylight", is_expired=expired,
        output_path=out / f"badge_{label}.png"
    )

    result = process_badge_image_pipeline(img)
    cv2.imwrite(str(out / f"overlay_{label}.png"), result["debug_overlay"])
    cv2.imwrite(str(out / f"sensing_{label}.png"), result["normalized_sensing_patch"])
    cv2.imwrite(str(out / f"aligned_{label}.png"), result["aligned_badge"])

    pred, _ = predictor.predict_from_image(img)
    f = result["color_features"]

    print(f"\n{'='*50}")
    print(f"  CASE : {label.upper()}  (true dose = {dose} ppm*h)")
    print(f"{'='*50}")
    print(f"  Auto-aligned     : {result['is_auto_aligned']}")
    print(f"  Lighting profile : {result['lighting_qa']['lighting_profile']}")
    print(f"  Lighting OK      : {result['lighting_qa']['is_lighting_suitable']}")
    print(f"  L* mean          : {f['L_mean']:.2f}  (fresh~74, saturated~30)")
    print(f"  delta_E          : {f['delta_E']:.2f}")
    print(f"  R / G / B mean   : {f['R_mean']:.1f} / {f['G_mean']:.1f} / {f['B_mean']:.1f}")
    print(f"  Chroma           : {f['chroma']:.3f}")
    print(f"  Expiry (QR/chem) : {result['expiry_report']['expiry_status']}")
    print(f"  --- ML Prediction ---")
    print(f"  Predicted dose   : {pred.cumulative_dose_ppm_h:.2f} ppm*h")
    print(f"  CI               : [{pred.confidence_interval_low:.2f}, {pred.confidence_interval_high:.2f}]")
    print(f"  Risk             : {pred.risk_level}")
    print(f"  TWA 8h           : {pred.twa_8h_equivalent_ppm:.3f} ppm")

print(f"\nDebug images saved to: {out.resolve()}")
