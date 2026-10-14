import pandas as pd
import numpy as np

df = pd.read_csv('data/synthetic/synthetic_h2s_exposure_dataset.csv')

print("=== TRAINING DATA STATS ===")
print(f"Total samples : {len(df)}")
print(f"Dose range    : {df.cumulative_dose_ppm_h.min():.2f} - {df.cumulative_dose_ppm_h.max():.2f} ppm*h")
print()

# Low dose bucket
low = df[df['cumulative_dose_ppm_h'] <= 8.0]
print(f"Low dose (<=8 ppm*h) samples: {len(low)}")
print(f"  L_mean  : {low.L_mean.min():.1f} - {low.L_mean.max():.1f}  mean={low.L_mean.mean():.1f}")
print(f"  B_mean  : {low.B_mean.min():.1f} - {low.B_mean.max():.1f}  mean={low.B_mean.mean():.1f}")
print(f"  R_mean  : {low.R_mean.min():.1f} - {low.R_mean.max():.1f}  mean={low.R_mean.mean():.1f}")
print(f"  delta_E : {low.delta_E.min():.1f} - {low.delta_E.max():.1f}  mean={low.delta_E.mean():.1f}")
print()

# What training samples have L* similar to what CV extracts (74.4)?
similar_L = df[(df['L_mean'] >= 72) & (df['L_mean'] <= 77)]
print(f"Training samples with L* in [72-77]: {len(similar_L)}")
print(f"  dose range : {similar_L.cumulative_dose_ppm_h.min():.1f} - {similar_L.cumulative_dose_ppm_h.max():.1f}")
print(f"  dose mean  : {similar_L.cumulative_dose_ppm_h.mean():.1f}  <-- model predicts ~this")
print(f"  B_mean     : {similar_L.B_mean.mean():.1f}  vs CV extracts: 4.8  <-- MISMATCH")
print(f"  R_mean     : {similar_L.R_mean.mean():.1f}  vs CV extracts: 185")
print()

print("=== B_mean full distribution ===")
print(df['B_mean'].describe().round(2))
print()
print(f"CV pipeline B_mean = 4.8  (training mean = {df.B_mean.mean():.1f})")
print()

# ratio_GB mismatch
print("=== ratio_GB ===")
print(f"Training ratio_GB mean : {df.ratio_GB.mean():.2f}")
print(f"CV extracts ratio_GB   : ~38.0  (G=183 / B=4.8)")
print()

# Dose distribution
print("=== Dose distribution in training data ===")
bins = [0, 5, 10, 20, 40, 85]
labels = ['0-5', '5-10', '10-20', '20-40', '40-85']
df['dose_bin'] = pd.cut(df['cumulative_dose_ppm_h'], bins=bins, labels=labels)
print(df['dose_bin'].value_counts().sort_index())
