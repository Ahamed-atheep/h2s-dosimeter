"""
Diagnose video scan: extract frames, run full pipeline, report per-frame diagnostics.
"""
import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VIDEO_PATH = r"D:\Downloads\WhatsApp Video 2026-09-07 at 2.20.28 PM.mp4"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "debug", "video_diag")
os.makedirs(OUT_DIR, exist_ok=True)

from src.ml.predict import DosimeterPredictor
from src.preprocessing.illumination import assess_capture_quality, apply_clahe_preprocess
from src.preprocessing.roi import detect_badge_boundary_with_confidence, warp_to_canonical_badge, extract_badge_rois
from src.preprocessing.illumination import compute_channel_transfer_gains, normalize_patch_illumination
from src.preprocessing.color_features import extract_color_features_from_patch

predictor = DosimeterPredictor()

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
    sys.exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps         = cap.get(cv2.CAP_PROP_FPS)
width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration    = total_frames / fps if fps > 0 else 0

print(f"Video: {width}x{height}  {fps:.1f}fps  {total_frames} frames  {duration:.1f}s")
print(f"Sampling every 0.5s\n")

# Sample every 0.5 seconds
sample_interval = max(1, int(fps * 0.5))
results = []

frame_idx = 0
sample_num = 0

print(f"{'#':>3} {'t(s)':>5} | {'pred':>6} | {'conf':>5} | {'bright':>6} | {'blur':>6} | "
      f"{'B_mean':>7} {'R_mean':>7} {'dE':>6} | {'slopes':>20} | notes")
print("-" * 120)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % sample_interval == 0:
        t = frame_idx / fps
        sample_num += 1

        # Basic quality
        cq = assess_capture_quality(frame)
        notes = []

        if not cq["is_acceptable"]:
            notes.append(cq["retake_reason"])
            print(f"{sample_num:>3} {t:>5.1f}s | {'REJECT':>6} | {'--':>5} | "
                  f"{cq['brightness']:>6.1f} | {cq['blur_score']:>6.1f} | "
                  f"{'--':>7} {'--':>7} {'--':>6} | {'--':>20} | {cq['retake_reason'][:60]}")
            # Save rejected frame
            cv2.imwrite(os.path.join(OUT_DIR, f"frame_{sample_num:03d}_t{t:.1f}s_REJECT.jpg"), frame)
            frame_idx += 1
            continue

        # CLAHE
        frame_clahe = apply_clahe_preprocess(frame)

        # Alignment
        corners, conf = detect_badge_boundary_with_confidence(frame_clahe)
        aligned, is_auto = warp_to_canonical_badge(frame_clahe, corners=corners)

        if not is_auto and conf < 0.15:
            notes.append("badge not detected")
            print(f"{sample_num:>3} {t:>5.1f}s | {'NO_BADGE':>6} | {conf:>5.3f} | "
                  f"{cq['brightness']:>6.1f} | {cq['blur_score']:>6.1f} | "
                  f"{'--':>7} {'--':>7} {'--':>6} | {'--':>20} | badge not detected in frame")
            cv2.imwrite(os.path.join(OUT_DIR, f"frame_{sample_num:03d}_t{t:.1f}s_NOBADGE.jpg"), frame)
            frame_idx += 1
            continue

        # ROI extraction
        try:
            rois = extract_badge_rois(aligned, apply_erosion=True)
        except Exception as e:
            print(f"{sample_num:>3} {t:>5.1f}s | {'ROI_ERR':>6} | {conf:>5.3f} | -- | -- | -- | {str(e)[:60]}")
            frame_idx += 1
            continue

        sensing = rois["sensing_patch"]["image"]
        ref_patches = {k: rois[k]["image"] for k in ["ref_white","ref_light_gray","ref_med_gray","ref_dark"]}

        # Normalization
        slopes, offsets, norm_diag = compute_channel_transfer_gains(ref_patches)
        norm_sensing = normalize_patch_illumination(sensing, slopes, offsets)

        # Raw sensing patch color (BEFORE normalization)
        raw_bgr = sensing.mean(axis=(0,1))
        raw_R, raw_G, raw_B = raw_bgr[2], raw_bgr[1], raw_bgr[0]

        # Normalized sensing patch color
        norm_bgr = norm_sensing.mean(axis=(0,1))
        norm_R, norm_G, norm_B = norm_bgr[2], norm_bgr[1], norm_bgr[0]

        # Features
        feats = extract_color_features_from_patch(norm_sensing, temp_c=28.0, humidity_pct=55.0, exposure_time_min=240.0)

        # Prediction
        try:
            res = predictor.predict_single(feats)
            pred = res.cumulative_dose_ppm_h
        except Exception as e:
            pred = -1.0
            notes.append(str(e))

        slope_str = str([round(s,2) for s in slopes])
        note_str = " | ".join(notes) if notes else "OK"

        print(f"{sample_num:>3} {t:>5.1f}s | {pred:>6.2f} | {conf:>5.3f} | "
              f"{cq['brightness']:>6.1f} | {cq['blur_score']:>6.1f} | "
              f"{norm_B:>7.1f} {norm_R:>7.1f} {feats['delta_E']:>6.1f} | "
              f"{slope_str:>20} | raw_R={raw_R:.0f} raw_B={raw_B:.0f} | {note_str}")

        results.append({
            "t": t, "pred": pred, "conf": conf,
            "brightness": cq["brightness"], "blur": cq["blur_score"],
            "norm_B": norm_B, "norm_R": norm_R, "raw_R": raw_R, "raw_B": raw_B,
            "delta_E": feats["delta_E"], "slopes": slopes.tolist(),
            "norm_method": norm_diag.get("method","?")
        })

        # Save annotated frame
        ann = aligned.copy()
        cv2.putText(ann, f"pred={pred:.1f}ppm conf={conf:.2f}", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(ann, f"B={norm_B:.0f} R={norm_R:.0f} dE={feats['delta_E']:.1f}", (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1)
        cv2.imwrite(os.path.join(OUT_DIR, f"frame_{sample_num:03d}_t{t:.1f}s_pred{pred:.1f}.jpg"), ann)

    frame_idx += 1

cap.release()

# Summary
if results:
    preds = [r["pred"] for r in results if r["pred"] > 0]
    confs = [r["conf"] for r in results]
    raw_Rs = [r["raw_R"] for r in results]
    norm_Rs = [r["norm_R"] for r in results]
    dEs = [r["delta_E"] for r in results]

    print(f"\n{'='*80}")
    print(f"SUMMARY  ({len(results)} valid frames)")
    print(f"  Prediction:  min={min(preds):.2f}  max={max(preds):.2f}  mean={np.mean(preds):.2f}  std={np.std(preds):.2f}")
    print(f"  Alignment:   min_conf={min(confs):.3f}  mean_conf={np.mean(confs):.3f}")
    print(f"  Raw R_mean:  min={min(raw_Rs):.1f}  max={max(raw_Rs):.1f}  mean={np.mean(raw_Rs):.1f}")
    print(f"  Norm R_mean: min={min(norm_Rs):.1f}  max={max(norm_Rs):.1f}  mean={np.mean(norm_Rs):.1f}")
    print(f"  delta_E:     min={min(dEs):.1f}  max={max(dEs):.1f}  mean={np.mean(dEs):.1f}")
    print(f"\n  TRUE DOSE: 14.5 ppm*h  |  Expected R_mean ~103  B_mean ~48  dE ~38")
    print(f"  ACTUAL mean pred: {np.mean(preds):.2f} ppm*h")
    print(f"\n  Annotated frames saved to: {OUT_DIR}")

    # Find the best frame (highest conf)
    best = max(results, key=lambda r: r["conf"])
    print(f"\n  Best frame: t={best['t']:.1f}s  pred={best['pred']:.2f}  conf={best['conf']:.3f}  "
          f"raw_R={best['raw_R']:.0f}  norm_R={best['norm_R']:.0f}  dE={best['delta_E']:.1f}")
    print(f"  Slopes on best frame: {[round(s,3) for s in best['slopes']]}")
