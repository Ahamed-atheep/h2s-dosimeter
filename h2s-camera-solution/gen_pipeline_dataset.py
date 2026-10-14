"""
Generate training dataset by running the ACTUAL CV pipeline on generated badges.
This guarantees 100% feature alignment between training and inference.
Replaces the synthetic formula-based approach for RGB/CIELAB features.
"""
import sys, os, cv2, numpy as np, pandas as pd, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.badge_generator import generate_test_badge_image
from src.preprocessing.roi import detect_badge_boundary_with_confidence, warp_to_canonical_badge, extract_badge_rois
from src.preprocessing.illumination import compute_channel_transfer_gains, normalize_patch_illumination, apply_clahe_preprocess
from src.preprocessing.color_features import extract_color_features_from_patch
from src.ml.synthetic import calculate_darkening_progress
from src.utils.config import SYNTHETIC_DATA_DIR, ensure_directories
import warnings
warnings.filterwarnings("ignore")

ensure_directories()

rng = np.random.default_rng(42)

# Dose distribution: concentrated in 0-30 ppm range, sparse above
doses_low    = rng.uniform(0.05, 10.0, 1800)   # 36% low
doses_mid    = rng.uniform(10.0, 25.0, 1500)   # 30% mid
doses_high   = rng.uniform(25.0, 50.0, 900)    # 18% high
doses_vhigh  = rng.uniform(50.0, 85.0, 800)    # 16% very high
all_doses    = np.concatenate([doses_low, doses_mid, doses_high, doses_vhigh])
rng.shuffle(all_doses)
N = len(all_doses)

temps    = np.clip(rng.normal(28.0, 6.5, N), 14.0, 48.0)
humids   = np.clip(rng.normal(55.0, 15.0, N), 18.0, 92.0)
exp_mins = np.clip(rng.uniform(30.0, 480.0, N), 30.0, 480.0)

lightings = rng.choice(["daylight", "warm", "cool", "shadow"], N)
tilts     = rng.uniform(-8.0, 8.0, N)

records = []
failed  = 0

print(f"Generating {N} pipeline-extracted samples...")

for i in range(N):
    dose  = float(all_doses[i])
    temp  = float(temps[i])
    humid = float(humids[i])
    exp_t = float(exp_mins[i])
    light = str(lightings[i])
    tilt  = float(tilts[i])

    try:
        img, _ = generate_test_badge_image(
            dose_ppm_h=dose, temp_c=temp, humidity_pct=humid,
            lighting=light, tilt_angle=tilt, is_expired=False,
            badge_id=f"H2S-TR-{i:05d}", batch_id="TR"
        )

        img_c = apply_clahe_preprocess(img)
        corners, conf = detect_badge_boundary_with_confidence(img_c)
        if conf < 0.3:
            failed += 1
            continue

        aligned, _ = warp_to_canonical_badge(img_c, corners=corners)
        rois = extract_badge_rois(aligned)
        sensing = rois["sensing_patch"]["image"]
        ref_patches = {k: rois[k]["image"] for k in ["ref_white","ref_light_gray","ref_med_gray","ref_dark"]}

        slopes, offsets, norm_diag = compute_channel_transfer_gains(ref_patches)
        if norm_diag.get("method") == "fallback_low_spread":
            failed += 1
            continue

        norm = normalize_patch_illumination(sensing, slopes, offsets)
        feats = extract_color_features_from_patch(norm, temp_c=temp, humidity_pct=humid, exposure_time_min=exp_t)

        rec = {
            "sample_id": f"SMP_{i+1:06d}",
            "strip_id": f"STRIP_{(i % 400)+1:04d}",
            "is_synthetic": True,
            "h2s_concentration_ppm": round(dose / (exp_t / 60.0), 3),
            "exposure_time_minutes": round(exp_t, 1),
            "temperature_c": round(temp, 2),
            "humidity_percent": round(humid, 2),
            "cumulative_dose_ppm_h": round(dose, 3),
        }
        rec.update({k: round(float(v), 4) for k, v in feats.items()
                    if isinstance(v, (int, float))})
        records.append(rec)

    except Exception:
        failed += 1

    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{N}  ok={len(records)}  failed={failed}")

df = pd.DataFrame(records)
out = SYNTHETIC_DATA_DIR / "synthetic_h2s_exposure_dataset.csv"
df.to_csv(out, index=False)

meta = {
    "is_synthetic": True,
    "pipeline_extracted": True,
    "n_samples": len(df),
    "failed": failed,
    "dose_range_ppm_h": [float(df["cumulative_dose_ppm_h"].min()), float(df["cumulative_dose_ppm_h"].max())],
}
with open(SYNTHETIC_DATA_DIR / "synthetic_h2s_exposure_dataset_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nDone: {len(df)} samples saved ({failed} failed/skipped)")
print(f"Saved to: {out}")
