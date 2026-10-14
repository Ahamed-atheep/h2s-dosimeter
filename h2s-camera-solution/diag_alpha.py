import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.ml.synthetic import calculate_darkening_progress

print(f"{'dose':>6} | {'alpha':>6} | {'R':>4} {'G':>4} {'B':>4} | norm_R (after ~1.08x slope)")
print("-"*55)
for dose in [0.0, 3.0, 5.0, 7.0, 10.0, 13.0, 14.5, 15.0, 18.0, 20.0, 30.0]:
    alpha = float(calculate_darkening_progress(
        np.array([dose]), np.array([28.0]), np.array([55.0]), np.array([1.0]))[0])
    R = int(np.clip(210.0 - alpha*(210.0-58.0), 20, 255))
    G = int(np.clip(195.0 - alpha*(195.0-38.0), 15, 255))
    B = int(np.clip(155.0 - alpha*(155.0-15.0), 10, 255))
    norm_R = int(R * 1.085)  # approximate slope
    print(f"{dose:6.1f} | {alpha:6.3f} | {R:4d} {G:4d} {B:4d} | norm_R~{norm_R}")
