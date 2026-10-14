import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from src.utils.badge_generator import generate_test_badge_image
from src.preprocessing.roi import detect_badge_boundary_with_confidence, warp_to_canonical_badge, extract_badge_rois
from src.preprocessing.illumination import compute_channel_transfer_gains, normalize_patch_illumination, apply_clahe_preprocess
from src.preprocessing.color_features import extract_color_features_from_patch
from src.utils.config import physics_config

df = pd.read_csv("data/synthetic/synthetic_h2s_exposure_dataset.csv")

print(f"Baseline in config: L={physics_config.BASELINE_L} a={physics_config.BASELINE_A} b={physics_config.BASELINE_B}")
print()
print(f"{'dose':>6} | {'R_pipe':>7} {'dE_pipe':>8} {'L_pipe':>7} {'a_pipe':>7} {'b_pipe':>7} | "
      f"{'R_train':>7} {'dE_train':>8} {'L_train':>7}")
print("-"*80)

for dose in [3.0, 5.0, 7.0, 10.0, 13.0, 15.0, 18.0]:
    img, _ = generate_test_badge_image(
        dose_ppm_h=dose, temp_c=28.0, humidity_pct=55.0,
        lighting="daylight", tilt_angle=0.0, is_expired=False,
        badge_id="H2S-B21-00510", batch_id="B21"
    )
    img_c = apply_clahe_preprocess(img)
    corners, _ = detect_badge_boundary_with_confidence(img_c)
    aligned, _ = warp_to_canonical_badge(img_c, corners=corners)
    rois = extract_badge_rois(aligned)
    sensing = rois["sensing_patch"]["image"]
    ref_patches = {k: rois[k]["image"] for k in ["ref_white","ref_light_gray","ref_med_gray","ref_dark"]}
    slopes, offsets, _ = compute_channel_transfer_gains(ref_patches)
    norm = normalize_patch_illumination(sensing, slopes, offsets)
    feats = extract_color_features_from_patch(norm)

    mask = (df["cumulative_dose_ppm_h"] >= dose-0.8) & (df["cumulative_dose_ppm_h"] <= dose+0.8)
    R_tr = df[mask]["R_mean"].mean()
    dE_tr = df[mask]["delta_E"].mean()
    L_tr = df[mask]["L_mean"].mean()

    print(f"{dose:6.1f} | {feats['R_mean']:7.1f} {feats['delta_E']:8.1f} {feats['L_mean']:7.1f} "
          f"{feats['a_mean']:7.2f} {feats['b_mean']:7.2f} | "
          f"{R_tr:7.1f} {dE_tr:8.1f} {L_tr:7.1f}")
