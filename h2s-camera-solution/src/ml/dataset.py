"""
Dataset management, validation, feature engineering, and leakage-free grouped splitting.
Supports both synthetic datasets and real experimental calibration datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from src.utils.config import (
    features_config, ml_config, PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR
)
from src.utils.logger import get_logger

logger = get_logger("ML_Dataset")


def load_dataset(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load exposure dataset from CSV. Defaults to the synthetic dataset if path is not provided.
    """
    if file_path is None:
        file_path = SYNTHETIC_DATA_DIR / "synthetic_h2s_exposure_dataset.csv"
        
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{file_path}'. Please run synthetic generator or provide valid calibration data path."
        )
        
    df = pd.read_csv(file_path)
    logger.info(f"Loaded dataset from '{file_path}' with {len(df)} rows and {len(df.columns)} columns.")
    return df


def validate_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate presence of essential columns and compute derived color metrics if absent.
    """
    df_proc = df.copy()
    
    # 1. Check target
    target_col = features_config.TARGET_COLUMN
    if target_col not in df_proc.columns:
        # Check if concentration and exposure time exist to derive dose
        if "h2s_concentration_ppm" in df_proc.columns and "exposure_time_minutes" in df_proc.columns:
            logger.warning(f"Target '{target_col}' not found. Deriving from concentration * time.")
            df_proc[target_col] = df_proc["h2s_concentration_ppm"] * (df_proc["exposure_time_minutes"] / 60.0)
        else:
            raise ValueError(f"Required target column '{target_col}' is missing from dataset.")

    # 2. Check group identifier (for data leakage prevention)
    group_col = features_config.GROUP_COLUMN
    if group_col not in df_proc.columns:
        logger.warning(f"Group column '{group_col}' missing. Creating default sample grouping.")
        df_proc[group_col] = [f"STRIP_{i//10:04d}" for i in range(len(df_proc))]

    # 3. Derived color features if missing
    if "delta_E" not in df_proc.columns and {"L_mean", "a_mean", "b_mean"}.issubset(df_proc.columns):
        from src.utils.config import physics_config
        df_proc["delta_E"] = np.sqrt(
            (df_proc["L_mean"] - physics_config.BASELINE_L) ** 2 +
            (df_proc["a_mean"] - physics_config.BASELINE_A) ** 2 +
            (df_proc["b_mean"] - physics_config.BASELINE_B) ** 2
        )
        
    if "chroma" not in df_proc.columns and {"a_mean", "b_mean"}.issubset(df_proc.columns):
        df_proc["chroma"] = np.sqrt(df_proc["a_mean"] ** 2 + df_proc["b_mean"] ** 2)
        
    if "hue_angle" not in df_proc.columns and {"a_mean", "b_mean"}.issubset(df_proc.columns):
        df_proc["hue_angle"] = np.degrees(np.arctan2(df_proc["b_mean"], df_proc["a_mean"])) % 360.0

    if "ratio_RG" not in df_proc.columns and {"R_mean", "G_mean"}.issubset(df_proc.columns):
        df_proc["ratio_RG"] = df_proc["R_mean"] / np.maximum(df_proc["G_mean"], 1e-4)

    if "ratio_GB" not in df_proc.columns and {"G_mean", "B_mean"}.issubset(df_proc.columns):
        df_proc["ratio_GB"] = df_proc["G_mean"] / np.maximum(df_proc["B_mean"], 1e-4)

    if "ratio_RB" not in df_proc.columns and {"R_mean", "B_mean"}.issubset(df_proc.columns):
        df_proc["ratio_RB"] = df_proc["R_mean"] / np.maximum(df_proc["B_mean"], 1e-4)

    # 4. Verify all expected features are present
    missing_features = [f for f in features_config.ALL_INPUT_FEATURES if f not in df_proc.columns]
    if missing_features:
        raise ValueError(f"Dataset is missing required feature columns: {missing_features}")

    return df_proc


def split_grouped_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into Train and Test sets using GroupShuffleSplit on 'strip_id'.
    Guarantees that no physical strip/experiment appears in both train and test partitions.
    """
    df_valid = validate_and_engineer_features(df)
    group_col = features_config.GROUP_COLUMN
    
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_idx, test_idx = next(gss.split(df_valid, groups=df_valid[group_col]))
    
    train_df = df_valid.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df_valid.iloc[test_idx].copy().reset_index(drop=True)
    
    # Verify no overlap between groups
    train_groups = set(train_df[group_col].unique())
    test_groups = set(test_df[group_col].unique())
    overlap = train_groups.intersection(test_groups)
    assert len(overlap) == 0, f"Data leakage detected! Overlapping groups: {overlap}"
    
    logger.info(
        f"Group-aware split completed: "
        f"Train={len(train_df)} samples ({len(train_groups)} strips), "
        f"Test={len(test_df)} samples ({len(test_groups)} strips) | Overlap = 0"
    )
    
    return train_df, test_df


def get_feature_matrices(
    df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Extract feature matrix X, target vector y, group vector groups, and feature names.
    """
    features = features_config.ALL_INPUT_FEATURES
    target = features_config.TARGET_COLUMN
    group_col = features_config.GROUP_COLUMN
    
    X = df[features].values.astype(np.float32)
    y = df[target].values.astype(np.float32)
    groups = df[group_col].values
    
    return X, y, groups, features
