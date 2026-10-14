import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils.badge_generator import generate_test_badge_image
from src.preprocessing.roi import detect_badge_boundary_with_confidence, warp_to_canonical_badge, extract_badge_rois
from src.preprocessing.illumination import compute_channel_transfer_gains, normalize_patch_illumination, apply_clahe_preprocess, CALIBRATED_TARGETS_RGB
from src.preprocessing.color_features import extract_color_features_from_patch
from src.ml.predict import DosimeterPredictor

predictor = DosimeterPredictor()

print("=== CALIBRATED TARGETS (illumination.py) ===")
for k, v in CALIBRATED_TARGETS_RGB.items():
    print(f"  {k}: RGB=({v[0]:.0f},{v[1]:.0f},{v[2]:.0f})")

print()
for dose in [5.0, 10.0, 14.5, 18.0]:
    img, _ = generate_test_badge_image(
        dose_ppm_h=dose, temp_c=28.0, humidity_pct=55.0,
        lighting="daylight", tilt_angle=0.0, is_expired=False,
        badge_id="H2S-B21-00510", batch_id="B21"
    )
    img_c = apply_clahe_preprocess(img)
    corners, conf = detect_badge_boundary_with_confidence(img_c)
    aligned, _ = warp_to_canonical_badge(img_c, corners=corners)
    rois = extract_badge_rois(aligned)
    sensing = rois["sensing_patch"]["image"]
    ref_patches = {k: rois[k]["image"] for k in ["ref_white","ref_light_gray","ref_med_gray","ref_dark"]}

    slopes, offsets, _ = compute_channel_transfer_gains(ref_patches)
    norm = normalize_patch_illumination(sensing, slopes, offsets)
    feats = extract_color_features_from_patch(norm)

    # swatch raw colors
    sw_colors = {k: ref_patches[k].mean(axis=(0,1)) for k in ref_patches}

    res = predictor.predict_single(feats)

    print(f"Dose={dose:5.1f} | pred={res.cumulative_dose_ppm_h:6.2f} | "
          f"slopes={[round(float(s),3) for s in slopes]} | "
          f"raw_R={sensing.mean(axis=(0,1))[2]:.0f} norm_R={norm.mean(axis=(0,1))[2]:.0f} | "
          f"dE={feats['delta_E']:.1f}")
    print(f"         Swatches raw BGR: "
          f"W=({sw_colors['ref_white'][2]:.0f},{sw_colors['ref_white'][1]:.0f},{sw_colors['ref_white'][0]:.0f}) "
          f"LG=({sw_colors['ref_light_gray'][2]:.0f},{sw_colors['ref_light_gray'][1]:.0f},{sw_colors['ref_light_gray'][0]:.0f}) "
          f"MG=({sw_colors['ref_med_gray'][2]:.0f},{sw_colors['ref_med_gray'][1]:.0f},{sw_colors['ref_med_gray'][0]:.0f}) "
          f"DK=({sw_colors['ref_dark'][2]:.0f},{sw_colors['ref_dark'][1]:.0f},{sw_colors['ref_dark'][0]:.0f})")
