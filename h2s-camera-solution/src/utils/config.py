"""
Central configuration module for the H2S Passive Dosimeter Software Solution.
Defines paths, feature definitions, physical/kinetic parameters, and ML configurations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any


# Root directory of the camera solution
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"

# Images directories
IMAGES_DIR = BASE_DIR / "images"
RAW_IMAGES_DIR = IMAGES_DIR / "raw"
TEST_IMAGES_DIR = IMAGES_DIR / "test"

# Artifacts & Outputs
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"


@dataclass
class PhysicsConfig:
    """
    Physical & Chemical parameters simulating lead acetate strip kinetics.
    Pb(CH3COO)2 + H2S -> PbS (dark brown/black precipitate) + 2 CH3COOH
    """
    # Reference environmental conditions
    T_REF_CELSIUS: float = 25.0       # Reference temperature in °C
    RH_REF_PERCENT: float = 50.0      # Reference relative humidity in %
    
    # Arrhenius activation energy approximation (kJ/mol) for gas-solid diffusion
    ACTIVATION_ENERGY_KJ_MOL: float = 28.5
    GAS_CONSTANT_R: float = 8.314e-3  # kJ / (mol * K)
    
    # Maximum optical color shift delta E* at full saturation
    DELTA_E_MAX: float = 65.0
    
    # Half-saturation dose Kd (ppm*h) where Delta E is 50% of max
    KD_DOSE_PPM_H: float = 14.0
    
    # Hill cooperativity coefficient
    HILL_COEFFICIENT: float = 1.15
    
    # Baseline unexposed strip color — real warm-tan lead acetate on white porous substrate
    # Matches ANCHORS[0] in gen_real_color_badges.py and badge_generator.py (dose=0 ppm)
    BASELINE_L: float = 74.0
    BASELINE_A: float = -0.2
    BASELINE_B: float = 21.5
    BASELINE_RGB: tuple = (210, 195, 155)

    # Fully saturated lead sulfide (PbS) color
    # Matches ANCHORS[-1] in gen_real_color_badges.py (dose=30 ppm)
    SATURATED_L: float = 22.0
    SATURATED_A: float = 4.5
    SATURATED_B: float = 8.0
    SATURATED_RGB: tuple = (58, 38, 15)
    
    # Occupational exposure threshold limits (OISD / DGMS / OSHA standards)
    # TWA (Time Weighted Average) = 10 ppm for 8h shift -> 80 ppm*h max limit
    # Recommended alert warning dose: 10 ppm*h (e.g. 1.25 ppm average over 8h)
    DOSE_ALERT_WARNING: float = 10.0   # ppm*h
    DOSE_ALERT_DANGER: float = 40.0    # ppm*h
    DOSE_ALERT_CRITICAL: float = 80.0  # ppm*h


@dataclass
class FeaturesConfig:
    """
    Feature schema definitions.
    """
    # Multi-space color features extracted from the sensing patch
    COLOR_FEATURES: List[str] = field(default_factory=lambda: [
        "R_mean", "G_mean", "B_mean",
        "R_std", "G_std", "B_std",
        "H_mean", "S_mean", "V_mean",
        "H_std", "S_std", "V_std",
        "L_mean", "a_mean", "b_mean",
        "L_std", "a_std", "b_std",
        "delta_E", "chroma", "hue_angle",
        "ratio_RG", "ratio_GB", "ratio_RB"
    ])
    
    # Environmental & acquisition context features
    CONTEXT_FEATURES: List[str] = field(default_factory=lambda: [
        "temperature_c",
        "humidity_percent",
        "exposure_time_minutes"
    ])
    
    # Target column
    TARGET_COLUMN: str = "cumulative_dose_ppm_h"
    
    # Grouping column for preventing data leakage across frames of the same strip
    GROUP_COLUMN: str = "strip_id"
    
    @property
    def ALL_INPUT_FEATURES(self) -> List[str]:
        return self.COLOR_FEATURES + self.CONTEXT_FEATURES


@dataclass
class MLConfig:
    """
    Machine Learning training & validation configuration.
    """
    RANDOM_SEED: int = 42
    N_SPLITS_CV: int = 5
    TEST_SIZE: float = 0.2
    
    # XGBoost hyperparameter defaults
    XGB_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 250,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1
    })
    
    # Random Forest hyperparameter defaults
    RF_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 12,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1
    })
    
    # Ridge Regression hyperparameter defaults
    RIDGE_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        "alpha": 1.0,
        "random_state": 42
    })


# Global instances
physics_config = PhysicsConfig()
features_config = FeaturesConfig()
ml_config = MLConfig()


def ensure_directories():
    """Create all essential project directories if they do not exist."""
    for directory in [
        RAW_DATA_DIR, PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR,
        RAW_IMAGES_DIR, TEST_IMAGES_DIR,
        MODELS_DIR, RESULTS_DIR, NOTEBOOKS_DIR
    ]:
        directory.mkdir(parents=True, exist_ok=True)
