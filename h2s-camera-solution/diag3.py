import cv2, numpy as np
from src.preprocessing.color_features import extract_color_features_from_patch
from src.preprocessing.roi import warp_to_canonical_badge, extract_badge_rois
from src.preprocessing.illumination import compute_channel_transfer_gains, normalize_patch_illumination

img = cv2.imread(r'E:\Sih\h2s-camera-solution\images\test\badge_dose_5.0ppm_daylight_valid.png')
aligned, _ = warp_to_canonical_badge(img)
rois = extract_badge_rois(aligned)
raw = rois['sensing_patch']['image']
ref = {k: rois[k]['image'] for k in ['ref_white','ref_light_gray','ref_med_gray','ref_dark']}
slopes, offsets, _ = compute_channel_transfer_gains(ref)
norm = normalize_patch_illumination(raw, slopes, offsets)

print('norm BGR channels:')
print('  B:', norm[:,:,0].mean().round(1))
print('  G:', norm[:,:,1].mean().round(1))
print('  R:', norm[:,:,2].mean().round(1))

# Convert to RGB manually
norm_rgb = cv2.cvtColor(norm, cv2.COLOR_BGR2RGB).astype(np.float32)
print('\nnorm_rgb channels:')
print('  R:', norm_rgb[:,:,0].mean().round(1))
print('  G:', norm_rgb[:,:,1].mean().round(1))
print('  B:', norm_rgb[:,:,2].mean().round(1))

# Now call feature extraction
f = extract_color_features_from_patch(norm, 28.0, 55.0, 240.0)
print('\nExtracted features:')
print('  R_mean:', f['R_mean'])
print('  G_mean:', f['G_mean'])
print('  B_mean:', f['B_mean'])
print('  L_mean:', f['L_mean'])
print('  a_mean:', f['a_mean'])
print('  b_mean:', f['b_mean'])

# Check HSV
norm_hsv = cv2.cvtColor(norm, cv2.COLOR_BGR2HSV).astype(np.float32)
print('\nHSV channels:')
print('  H:', norm_hsv[:,:,0].mean().round(1), '(OpenCV 0-180)')
print('  S:', (norm_hsv[:,:,1] * 100/255).mean().round(1), '%')
print('  V:', (norm_hsv[:,:,2] * 100/255).mean().round(1), '%')
