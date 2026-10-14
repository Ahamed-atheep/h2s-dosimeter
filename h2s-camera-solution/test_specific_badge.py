import cv2
import pathlib
from src.ml.predict import DosimeterPredictor
from src.preprocessing.color_features import process_badge_image_pipeline

img_path = r"E:\Sih\h2s-camera-solution\images\test\real_color_badges_v2\badge_05ppm_VALID_H2S-B21-00505.png"
out = pathlib.Path("images/test/debug_check")
out.mkdir(parents=True, exist_ok=True)

# Load image
img = cv2.imread(img_path)
if img is None:
    print("ERROR: Could not load image at", img_path)
    exit(1)

print(f"Image loaded: {img.shape[1]}x{img.shape[0]} px")

# Run CV pipeline
result = process_badge_image_pipeline(img)

# Save debug outputs
cv2.imwrite(str(out / "test_overlay.png"),  result["debug_overlay"])
cv2.imwrite(str(out / "test_aligned.png"),  result["aligned_badge"])
cv2.imwrite(str(out / "test_sensing.png"),  result["normalized_sensing_patch"])

f = result["color_features"]
lqa = result["lighting_qa"]
exp = result["expiry_report"]
norm = result["normalization_diagnostics"]

print("\n========== CV PIPELINE ==========")
print(f"  Auto-aligned       : {result['is_auto_aligned']}")
print(f"  Lighting profile   : {lqa['lighting_profile']}")
print(f"  Lighting suitable  : {lqa['is_lighting_suitable']}")
print(f"  Glare detected     : {lqa['is_glare_detected']}  ({lqa['glare_pixel_percentage']}%)")
print(f"  Underexposed       : {lqa['is_underexposed']}")
print(f"  White patch intens : {lqa['white_patch_intensity']}")
print(f"  Norm method        : {norm.get('method')}")
print(f"  Channel gains BGR  : {norm.get('channel_gains_bgr')}")

print("\n========== COLOR FEATURES ==========")
print(f"  L* mean            : {f['L_mean']:.2f}  (fresh~74, saturated~30)")
print(f"  a* mean            : {f['a_mean']:.3f}")
print(f"  b* mean            : {f['b_mean']:.3f}")
print(f"  delta_E            : {f['delta_E']:.3f}")
print(f"  Chroma             : {f['chroma']:.3f}")
print(f"  Hue angle          : {f['hue_angle']:.2f}°")
print(f"  R / G / B mean     : {f['R_mean']:.1f} / {f['G_mean']:.1f} / {f['B_mean']:.1f}")
print(f"  R / G / B std      : {f['R_std']:.2f} / {f['G_std']:.2f} / {f['B_std']:.2f}")
print(f"  H / S / V mean     : {f['H_mean']:.1f} / {f['S_mean']:.1f} / {f['V_mean']:.1f}")
print(f"  Ratio R/G          : {f['ratio_RG']:.4f}")
print(f"  Ratio G/B          : {f['ratio_GB']:.4f}")

print("\n========== EXPIRY CHECK ==========")
print(f"  Status             : {exp['expiry_status']}")
print(f"  Is valid           : {exp['is_valid']}")
print(f"  Degradation %      : {exp['degradation_percentage']}%")
print(f"  Expiry L* / a* / b*: {exp['mean_L']} / {exp['mean_a']} / {exp['mean_b']}")

# Run ML prediction
predictor = DosimeterPredictor()
pred, _ = predictor.predict_from_image(img)

print("\n========== ML PREDICTION ==========")
print(f"  True dose (label)  : 5.0 ppm*h")
print(f"  Predicted dose     : {pred.cumulative_dose_ppm_h:.3f} ppm*h")
print(f"  Error              : {abs(pred.cumulative_dose_ppm_h - 5.0):.3f} ppm*h")
print(f"  95% CI             : [{pred.confidence_interval_low:.2f}, {pred.confidence_interval_high:.2f}]")
print(f"  TWA 8h             : {pred.twa_8h_equivalent_ppm:.4f} ppm")
print(f"  Risk level         : {pred.risk_level}")
print(f"  Badge valid        : {pred.is_badge_valid}")
print(f"  Model type         : {pred.model_type}")
print(f"  Uncertainty sigma  : {predictor.uncertainty_sigma:.4f}")

print(f"\nDebug images saved to: {out.resolve()}")
