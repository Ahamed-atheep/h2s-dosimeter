import cv2
import numpy as np
from src.utils.badge_generator import generate_test_badge_image
from src.preprocessing.roi import warp_to_canonical_badge, extract_badge_rois
from src.preprocessing.illumination import compute_channel_transfer_gains, normalize_patch_illumination
import pathlib

img_path = r"E:\Sih\h2s-camera-solution\images\test\badge_dose_5.0ppm_daylight_valid.png"
img = cv2.imread(img_path)

# Step 1: warp
aligned, auto = warp_to_canonical_badge(img)
print(f"Aligned shape: {aligned.shape}, auto={auto}")

# Step 2: extract ROIs
rois = extract_badge_rois(aligned, apply_erosion=True)

# Step 3: check raw sensing patch BEFORE normalization
raw_sensing = rois["sensing_patch"]["image"]
print(f"\nRaw sensing patch shape: {raw_sensing.shape}")
print(f"Raw sensing B mean: {raw_sensing[:,:,0].mean():.1f}")
print(f"Raw sensing G mean: {raw_sensing[:,:,1].mean():.1f}")
print(f"Raw sensing R mean: {raw_sensing[:,:,2].mean():.1f}")

# Step 4: check reference patches
for name in ["ref_white", "ref_light_gray", "ref_med_gray", "ref_dark"]:
    p = rois[name]["image"]
    print(f"  {name}: B={p[:,:,0].mean():.1f} G={p[:,:,1].mean():.1f} R={p[:,:,2].mean():.1f}")

# Step 5: compute gains
ref_patches = {k: rois[k]["image"] for k in ["ref_white","ref_light_gray","ref_med_gray","ref_dark"]}
slopes, offsets, diag = compute_channel_transfer_gains(ref_patches)
print(f"\nSlopes BGR: {slopes}")
print(f"Offsets BGR: {offsets}")

# Step 6: normalize
norm = normalize_patch_illumination(raw_sensing, slopes, offsets)
print(f"\nNormalized sensing B mean: {norm[:,:,0].mean():.1f}")
print(f"Normalized sensing G mean: {norm[:,:,1].mean():.1f}")
print(f"Normalized sensing R mean: {norm[:,:,2].mean():.1f}")

# Step 7: convert to RGB for feature extraction
import cv2
norm_rgb = cv2.cvtColor(norm, cv2.COLOR_BGR2RGB).astype(np.float32)
print(f"\nAfter BGR->RGB conversion:")
print(f"  R mean: {norm_rgb[:,:,0].mean():.1f}")
print(f"  G mean: {norm_rgb[:,:,1].mean():.1f}")
print(f"  B mean: {norm_rgb[:,:,2].mean():.1f}  <-- this is what goes into features")

# Save patches for visual inspection
out = pathlib.Path("images/test/debug_check")
cv2.imwrite(str(out / "raw_sensing.png"), raw_sensing)
cv2.imwrite(str(out / "norm_sensing.png"), norm)
cv2.imwrite(str(out / "ref_white.png"), rois["ref_white"]["image"])
print("\nPatches saved.")
