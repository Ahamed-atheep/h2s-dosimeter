"""
Physics-informed Synthetic Dataset Generator for H2S Passive Dosimeter.
Simulates colorimetric response curves (RGB, HSV, CIELAB) of lead acetate strips
under varying H2S gas concentrations, exposure times, temperatures, and humidities.

DISCLAIMER:
This module produces SYNTHETIC data for pipeline development and algorithmic benchmarking.
Synthetic data is NOT experimental validation and must be replaced by real calibration data.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from src.utils.config import (
    physics_config, features_config, SYNTHETIC_DATA_DIR, ensure_directories
)
from src.utils.logger import get_logger

logger = get_logger("ML_Synthetic")


def kelvin(celsius: np.ndarray) -> np.ndarray:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def temperature_factor(temp_c: np.ndarray) -> np.ndarray:
    """
    Arrhenius temperature acceleration factor for chemical reaction and gas diffusion.
    f(T) = exp( -(Ea / R) * (1/T - 1/T_ref) )
    """
    t_k = kelvin(temp_c)
    t_ref_k = kelvin(physics_config.T_REF_CELSIUS)
    exponent = -(physics_config.ACTIVATION_ENERGY_KJ_MOL / physics_config.GAS_CONSTANT_R) * (1.0 / t_k - 1.0 / t_ref_k)
    # Clip exponent to prevent numerical instability
    exponent = np.clip(exponent, -3.0, 3.0)
    return np.exp(exponent)


def humidity_factor(humidity_pct: np.ndarray) -> np.ndarray:
    """
    Sigmoidal humidity dependence modeling humectant moisture retention and ionic mobility.
    At low RH (<30%), reaction is slower; reaches optimal plateau around 50-80% RH.
    """
    # Sigmoidal response bounded between ~0.65 and 1.15
    k = 0.08
    rh_mid = 45.0
    return 0.60 + 0.55 / (1.0 + np.exp(-k * (humidity_pct - rh_mid)))


def calculate_darkening_progress(
    dose_ppm_h: np.ndarray,
    temp_c: np.ndarray,
    humidity_pct: np.ndarray,
    strip_sensitivity_factor: np.ndarray
) -> np.ndarray:
    """
    Calculate dimensionless reaction progress alpha in [0, 1] using modified Hill kinetics.
    alpha = (Dose_eff ^ n) / (Kd ^ n + Dose_eff ^ n)
    """
    f_t = temperature_factor(temp_c)
    g_rh = humidity_factor(humidity_pct)
    
    # Effective dose taking environmental acceleration and strip formulation into account
    dose_effective = dose_ppm_h * f_t * g_rh * strip_sensitivity_factor
    
    kd = physics_config.KD_DOSE_PPM_H
    n = physics_config.HILL_COEFFICIENT
    
    # Hill equation
    dose_pow_n = np.power(np.maximum(dose_effective, 1e-6), n)
    kd_pow_n = np.power(kd, n)
    alpha = dose_pow_n / (kd_pow_n + dose_pow_n)
    return np.clip(alpha, 0.0, 1.0)


def generate_synthetic_dataset(
    n_samples: int = 5000,
    n_strips: int = 400,
    random_seed: int = 42,
    output_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate a physically grounded synthetic tabular dataset for camera-based dosimeter.
    
    Parameters
    ----------
    n_samples : int
        Total number of measurement samples to generate.
    n_strips : int
        Number of distinct physical strips / experiment batches (enables GroupKFold).
    random_seed : int
        Random seed for reproducibility.
    output_path : Optional[Path]
        Destination CSV file path. If None, saves to data/synthetic/synthetic_dataset.csv.
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        The generated DataFrame and associated metadata dict.
    """
    ensure_directories()
    rng = np.random.default_rng(random_seed)
    
    logger.info(f"Generating synthetic dataset: {n_samples} samples across {n_strips} strips (Seed={random_seed})")
    
    # 1. Assign samples to strips
    samples_per_strip = n_samples // n_strips
    remainder = n_samples % n_strips
    
    strip_ids = []
    for i in range(n_strips):
        count = samples_per_strip + (1 if i < remainder else 0)
        strip_ids.extend([f"STRIP_{i+1:04d}"] * count)
    rng.shuffle(strip_ids)
    
    # Strip-to-strip formulation variability (mean sensitivity centered at 1.0, std = 0.06)
    strip_unique = sorted(list(set(strip_ids)))
    strip_sensitivities = {
        s_id: rng.normal(1.0, 0.06) for s_id in strip_unique
    }
    sample_strip_sensitivities = np.array([strip_sensitivities[s] for s in strip_ids])
    
    # 2. Environmental conditions (realistic industrial ambient range)
    # Temperature: 15°C to 45°C (mean 28°C)
    temperature_c = np.clip(rng.normal(28.0, 6.5, n_samples), 14.0, 48.0)
    
    # Relative Humidity: 20% to 90% (mean 55%)
    humidity_percent = np.clip(rng.normal(55.0, 15.0, n_samples), 18.0, 92.0)
    
    # 3. Exposure variables: Concentration and Duration
    # H2S Concentration: log-normal distribution from low ppb up to 35 ppm
    # 60% low-level chronic (0.2 - 5 ppm), 30% moderate (5 - 15 ppm), 10% high (15 - 35 ppm)
    log_conc = rng.normal(0.8, 0.9, n_samples)
    h2s_concentration_ppm = np.clip(np.exp(log_conc), 0.1, 40.0)
    
    # Exposure time: 10 minutes to 480 minutes (8-hour shift)
    exposure_time_minutes = np.clip(rng.uniform(10.0, 480.0, n_samples), 5.0, 480.0)
    
    # Cumulative Dose (Target): C * (t / 60) ppm*h
    cumulative_dose_ppm_h = h2s_concentration_ppm * (exposure_time_minutes / 60.0)
    # Clip dose to typical dosimeter range [0.05, 85.0] ppm*h
    cumulative_dose_ppm_h = np.clip(cumulative_dose_ppm_h, 0.05, 85.0)
    
    # 4. Chemical reaction progress (alpha in [0, 1])
    alpha = calculate_darkening_progress(
        cumulative_dose_ppm_h,
        temperature_c,
        humidity_percent,
        sample_strip_sensitivities
    )
    
    # 5. Simulate Multi-space Color Transformations
    # CIELAB anchors match pipeline-extracted values after CLAHE + normalization:
    # Fresh (alpha=0): L*=74  a*=-0.2  b*=21.5
    # Saturated (alpha=1): L*=22  a*=4.5  b*=8.0
    base_L, base_a, base_b = 74.0, -0.2, 21.5
    sat_L,  sat_a,  sat_b  = physics_config.SATURATED_L, physics_config.SATURATED_A, physics_config.SATURATED_B

    # CIELAB values with camera optical noise
    noise_L = rng.normal(0.0, 1.1, n_samples)
    noise_a = rng.normal(0.0, 0.4, n_samples)
    noise_b = rng.normal(0.0, 0.6, n_samples)

    L_mean = np.clip(base_L - alpha * (base_L - sat_L) + noise_L, 15.0, 98.0)
    a_mean = np.clip(base_a + alpha * (sat_a - base_a) + noise_a, -5.0, 15.0)
    b_mean = np.clip(base_b + alpha * (sat_b - base_b) + noise_b, 0.0, 22.0)
    
    # Spatial standard deviations (surface texture / graininess increases at intermediate reaction)
    grain_factor = 4.0 * alpha * (1.0 - alpha)  # peaks at alpha = 0.5
    L_std = np.clip(rng.normal(1.2, 0.2, n_samples) + 0.8 * grain_factor, 0.5, 3.5)
    a_std = np.clip(rng.normal(0.6, 0.1, n_samples) + 0.4 * grain_factor, 0.2, 2.0)
    b_std = np.clip(rng.normal(0.8, 0.1, n_samples) + 0.5 * grain_factor, 0.3, 2.5)
    
    # Total color difference Delta E* from baseline
    delta_E = np.sqrt(
        (L_mean - base_L) ** 2 +
        (a_mean - base_a) ** 2 +
        (b_mean - base_b) ** 2
    )
    
    # Chroma and Hue angle
    chroma = np.sqrt(a_mean ** 2 + b_mean ** 2)
    hue_angle = np.degrees(np.arctan2(b_mean, a_mean)) % 360.0
    
    # 6. Synthesize RGB space
    # Uses IDENTICAL anchor interpolation as badge_generator.py and gen_real_color_badges.py
    # Fresh (alpha=0): R=210 G=195 B=155  (ANCHORS[0])
    # Saturated (alpha=1): R=58  G=38  B=15  (ANCHORS[-1])
    norm_gain = rng.uniform(0.95, 1.05, n_samples)
    rgb_noise = rng.normal(0.0, 2.5, (n_samples, 3))
    R_mean = np.clip((210.0 - alpha * (210.0 - 58.0)) * norm_gain + rgb_noise[:, 0], 20.0, 255.0)
    G_mean = np.clip((195.0 - alpha * (195.0 - 38.0)) * norm_gain + rgb_noise[:, 1], 15.0, 255.0)
    B_mean = np.clip((155.0 - alpha * (155.0 - 15.0)) * norm_gain + rgb_noise[:, 2], 10.0, 255.0)
    
    R_std = np.clip(rng.normal(2.5, 0.4, n_samples) + 1.5 * grain_factor, 1.0, 6.5)
    G_std = np.clip(rng.normal(2.5, 0.4, n_samples) + 1.5 * grain_factor, 1.0, 6.5)
    B_std = np.clip(rng.normal(2.2, 0.4, n_samples) + 1.4 * grain_factor, 1.0, 6.0)
    
    # Cross-channel ratios
    ratio_RG = R_mean / np.maximum(G_mean, 1e-4)
    ratio_GB = G_mean / np.maximum(B_mean, 1e-4)
    ratio_RB = R_mean / np.maximum(B_mean, 1e-4)
    
    # 7. Synthesize HSV space
    # Updated to match real warm-tan strip HSV:
    # Fresh: V=82% S=12% H=32 (warm yellow-tan hue)
    # Saturated: V=18% S=55% H=22 (dark brown)
    V_mean = np.clip(82.0 - alpha * (82.0 - 18.0) + rng.normal(0, 1.2, n_samples), 10.0, 100.0)
    S_mean = np.clip(12.0 + alpha * 43.0 + rng.normal(0, 1.5, n_samples), 0.0, 70.0)
    H_mean = np.clip(32.0 - alpha * 10.0 + rng.normal(0, 2.0, n_samples), 10.0, 60.0)
    
    H_std = np.clip(rng.normal(1.5, 0.3, n_samples), 0.5, 4.0)
    S_std = np.clip(rng.normal(1.8, 0.3, n_samples), 0.5, 4.5)
    V_std = np.clip(rng.normal(1.4, 0.3, n_samples), 0.5, 3.8)
    
    # 8. Assemble DataFrame
    df = pd.DataFrame({
        "sample_id": [f"SMP_{i+1:06d}" for i in range(n_samples)],
        "strip_id": strip_ids,
        "is_synthetic": True,
        
        # Exposure context
        "h2s_concentration_ppm": np.round(h2s_concentration_ppm, 3),
        "exposure_time_minutes": np.round(exposure_time_minutes, 1),
        "temperature_c": np.round(temperature_c, 2),
        "humidity_percent": np.round(humidity_percent, 2),
        
        # RGB features
        "R_mean": np.round(R_mean, 3),
        "G_mean": np.round(G_mean, 3),
        "B_mean": np.round(B_mean, 3),
        "R_std": np.round(R_std, 3),
        "G_std": np.round(G_std, 3),
        "B_std": np.round(B_std, 3),
        
        # HSV features
        "H_mean": np.round(H_mean, 3),
        "S_mean": np.round(S_mean, 3),
        "V_mean": np.round(V_mean, 3),
        "H_std": np.round(H_std, 3),
        "S_std": np.round(S_std, 3),
        "V_std": np.round(V_std, 3),
        
        # CIELAB features
        "L_mean": np.round(L_mean, 3),
        "a_mean": np.round(a_mean, 3),
        "b_mean": np.round(b_mean, 3),
        "L_std": np.round(L_std, 3),
        "a_std": np.round(a_std, 3),
        "b_std": np.round(b_std, 3),
        
        # Derived color metrics
        "delta_E": np.round(delta_E, 3),
        "chroma": np.round(chroma, 3),
        "hue_angle": np.round(hue_angle, 3),
        "ratio_RG": np.round(ratio_RG, 4),
        "ratio_GB": np.round(ratio_GB, 4),
        "ratio_RB": np.round(ratio_RB, 4),
        
        # Target
        "cumulative_dose_ppm_h": np.round(cumulative_dose_ppm_h, 3)
    })
    
    # 9. Metadata summary
    metadata = {
        "is_synthetic": True,
        "disclaimer": "SYNTHETIC DATASET FOR ALGORITHM BENCHMARKING ONLY. NOT REAL EXPERIMENTAL DATA.",
        "n_samples": n_samples,
        "n_strips": n_strips,
        "random_seed": random_seed,
        "dose_range_ppm_h": [float(df["cumulative_dose_ppm_h"].min()), float(df["cumulative_dose_ppm_h"].max())],
        "temperature_range_c": [float(df["temperature_c"].min()), float(df["temperature_c"].max())],
        "humidity_range_pct": [float(df["humidity_percent"].min()), float(df["humidity_percent"].max())],
        "physics_constants": {
            "T_ref_celsius": physics_config.T_REF_CELSIUS,
            "RH_ref_percent": physics_config.RH_REF_PERCENT,
            "activation_energy_kJ_mol": physics_config.ACTIVATION_ENERGY_KJ_MOL,
            "kd_dose_ppm_h": physics_config.KD_DOSE_PPM_H,
            "hill_coefficient": physics_config.HILL_COEFFICIENT,
            "delta_E_max": physics_config.DELTA_E_MAX
        }
    }
    
    # Save files
    if output_path is None:
        output_path = SYNTHETIC_DATA_DIR / "synthetic_h2s_exposure_dataset.csv"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    meta_path = output_path.parent / (output_path.stem + "_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    logger.info(f"Synthetic dataset successfully saved to: {output_path}")
    logger.info(f"Metadata saved to: {meta_path}")
    
    return df, metadata


if __name__ == "__main__":
    generate_synthetic_dataset(n_samples=5000, n_strips=400)
