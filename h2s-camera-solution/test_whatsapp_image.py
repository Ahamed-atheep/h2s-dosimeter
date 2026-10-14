"""
Test the WhatsApp badge image through the full CV + ML pipeline.
"""
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IMAGE_PATH = r"D:\Downloads\WhatsApp Image 2026-09-07 at 12.00.36 PM.jpeg"

def main():
    # 1. Load image
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"[ERROR] Could not load image: {IMAGE_PATH}")
        return

    h, w = img.shape[:2]
    print(f"[IMAGE] Loaded: {w}x{h} px")
    print(f"[IMAGE] Channels: {img.shape[2] if len(img.shape) == 3 else 1}")
    print(f"[IMAGE] Min/Max pixel: {img.min()} / {img.max()}")

    # 2. Capture quality check
    from src.preprocessing.illumination import assess_capture_quality
    cq = assess_capture_quality(img)
    print(f"\n[CAPTURE QUALITY]")
    for k, v in cq.items():
        print(f"  {k}: {v}")

    # 3. Badge boundary detection
    from src.preprocessing.roi import detect_badge_boundary_with_confidence, warp_to_canonical_badge
    corners, conf = detect_badge_boundary_with_confidence(img)
    print(f"\n[ALIGNMENT]")
    print(f"  Alignment confidence: {conf:.3f}")
    print(f"  Corners detected: {corners is not None}")
    if corners is not None:
        print(f"  Corners: {corners.tolist()}")

    aligned, is_auto = warp_to_canonical_badge(img, corners=corners)
    print(f"  Auto-aligned: {is_auto}")
    print(f"  Aligned shape: {aligned.shape[1]}x{aligned.shape[0]}")

    # 4. ROI extraction
    from src.preprocessing.roi import extract_badge_rois
    rois = extract_badge_rois(aligned, apply_erosion=True)
    sensing = rois["sensing_patch"]["image"]
    expiry  = rois["expiry_patch"]["image"]
    print(f"\n[ROIs]")
    print(f"  Sensing patch: {sensing.shape[1]}x{sensing.shape[0]}")
    print(f"  Expiry patch:  {expiry.shape[1]}x{expiry.shape[0]}")

    # 5. Illumination normalization
    from src.preprocessing.illumination import compute_channel_transfer_gains, normalize_patch_illumination
    ref_patches = {
        "ref_white":      rois["ref_white"]["image"],
        "ref_light_gray": rois["ref_light_gray"]["image"],
        "ref_med_gray":   rois["ref_med_gray"]["image"],
        "ref_dark":       rois["ref_dark"]["image"],
    }
    slopes, offsets, norm_diag = compute_channel_transfer_gains(ref_patches)
    norm_sensing = normalize_patch_illumination(sensing, slopes, offsets)
    print(f"\n[ILLUMINATION NORMALIZATION]")
    print(f"  Slopes  (B,G,R): {[round(s,4) for s in slopes]}")
    print(f"  Offsets (B,G,R): {[round(o,4) for o in offsets]}")

    # 6. Color features
    from src.preprocessing.color_features import extract_color_features_from_patch
    feats = extract_color_features_from_patch(norm_sensing)
    print(f"\n[COLOR FEATURES]")
    key_feats = ["R_mean","G_mean","B_mean","L_mean","a_mean","b_mean",
                 "delta_E","chroma","hue_angle","ratio_RG","ratio_GB","ratio_RB"]
    for k in key_feats:
        print(f"  {k}: {feats[k]}")

    # 7. QR decode
    from src.utils.qr_validator import decode_qr_from_image, validate_badge_qr_payload
    qr_raw = decode_qr_from_image(aligned)
    print(f"\n[QR CODE]")
    if qr_raw:
        print(f"  Decoded: {qr_raw}")
        val = validate_badge_qr_payload(qr_raw)
        print(f"  Status:  {val['badge_status']}")
        print(f"  Valid:   {val['is_valid']}")
        print(f"  Message: {val['message']}")
    else:
        print("  No QR code detected in image.")

    # 8. Expiry check
    from src.preprocessing.color_features import check_badge_expiry
    expiry_report = check_badge_expiry(
        normalize_patch_illumination(expiry, slopes, offsets),
        qr_payload=qr_raw,
        aligned_badge_bgr=aligned
    )
    print(f"\n[EXPIRY CHECK]")
    for k, v in expiry_report.items():
        if k != "qr_data":
            print(f"  {k}: {v}")

    # 9. Full ML prediction
    from src.ml.predict import DosimeterPredictor
    predictor = DosimeterPredictor()
    result, pipe_res = predictor.predict_from_image(img, temp_c=28.0, humidity_pct=55.0, exposure_time_min=240.0)

    print(f"\n[ML PREDICTION]")
    print(f"  Cumulative Dose:  {result.cumulative_dose_ppm_h:.3f} ppm·h")
    print(f"  95% CI:           [{result.confidence_interval_low:.3f}, {result.confidence_interval_high:.3f}]")
    print(f"  TWA (8h):         {result.twa_8h_equivalent_ppm:.3f} ppm")
    print(f"  Risk Level:       {result.risk_level}")
    print(f"  Risk Color:       {result.risk_color}")
    print(f"  Badge Valid:      {result.is_badge_valid}")
    print(f"  Expiry Status:    {result.expiry_status}")
    print(f"  Lighting Profile: {result.lighting_profile}")
    print(f"  Action:           {result.action_recommendation}")

    # 10. Save debug overlay
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "debug", "whatsapp_test_overlay.png")
    cv2.imwrite(out_path, pipe_res["debug_overlay"])
    print(f"\n[DEBUG OVERLAY] Saved → {out_path}")

if __name__ == "__main__":
    main()
