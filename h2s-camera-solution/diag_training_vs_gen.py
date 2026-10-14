import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv("data/synthetic/synthetic_h2s_exposure_dataset.csv")

print("Training data R_mean by dose bucket:")
bins = [0,3,5,7,10,13,15,18,20,30,85]
labels = ["0-3","3-5","5-7","7-10","10-13","13-15","15-18","18-20","20-30","30+"]
df["dose_bin"] = pd.cut(df["cumulative_dose_ppm_h"], bins=bins, labels=labels)
g = df.groupby("dose_bin")[["R_mean","G_mean","B_mean","L_mean","delta_E"]].mean()
print(g.round(1).to_string())

print("\nBadge generator actual values (from diag_alpha.py):")
from src.ml.synthetic import calculate_darkening_progress
print(f"{'dose':>6} | {'alpha':>6} | {'R_gen':>6} | {'R_train_expected':>16}")
for dose in [3.0, 5.0, 7.0, 10.0, 13.0, 15.0, 18.0, 20.0]:
    alpha = float(calculate_darkening_progress(np.array([dose]), np.array([28.0]), np.array([55.0]), np.array([1.0]))[0])
    R_gen = 210.0 - alpha*(210.0-58.0)
    # What R does training data have at this dose?
    mask = (df["cumulative_dose_ppm_h"] >= dose-1) & (df["cumulative_dose_ppm_h"] <= dose+1)
    R_train = df[mask]["R_mean"].mean() if mask.sum() > 0 else float("nan")
    print(f"{dose:6.1f} | {alpha:6.3f} | {R_gen:6.1f} | {R_train:16.1f}")
