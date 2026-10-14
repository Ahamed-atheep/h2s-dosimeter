# Digital Wearable H₂S Passive Dosimeter — Camera Software Solution

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent, low-cost, ML-assisted passive $H_2S$ cumulative exposure dosimeter designed for occupational health monitoring in hazardous industrial environments (oil refineries, petrochemical plants, sewers, mining).

Unlike standard electronic gas detectors that only report instantaneous peak gas levels, this system quantifies **cumulative exposure dose** ($\text{dose} = \int C(t)\,dt$ in $\text{ppm}\cdot\text{h}$) from disposable colorimetric chemical badges, objectively calibrated against reference optical scales and ambient conditions.

---

## Key Features

- **Physics-Grounded Chemical Modeling**: Simulates non-linear reaction kinetics ($Pb(Ac)_2 \to PbS$), Arrhenius temperature acceleration ($T \in [15, 45]^\circ\text{C}$), and humidity buffer scaling ($RH \in [20, 90]\%$).
- **Zero-Leakage Grouped Validation**: Strict `GroupKFold` cross-validation grouped by physical `strip_id` ensures models generalize to unseen physical strips without data leakage.
- **Multi-Model Benchmark**: Trains and compares **XGBoost**, **Random Forest**, and **Ridge Regression** to identify the best architecture.
- **Uncertainty & Risk Classification**: Provides estimated cumulative dose ($\text{ppm}\cdot\text{h}$), 95% confidence intervals, 8-hour equivalent TWA ($ppm$), and occupational risk categorization aligned with DGMS / OISD standards.
- **Drop-in Experimental Pluggability**: Seamlessly transitions from synthetic training datasets to real gas-chamber calibration datasets without changing downstream code.

---

## Directory Structure

```
h2s-camera-solution/
├── app.py                     # Unified CLI entry point
├── requirements.txt           # Python dependencies
├── data/
│   ├── raw/                   # Destination for real gas-chamber calibration data
│   ├── processed/             # Engineered train & test feature sets
│   └── synthetic/             # Generated synthetic exposure datasets
├── models/                    # Serialized model artifacts (.joblib)
├── results/                   # Diagnostic plots & evaluation metrics
├── src/
│   ├── ml/
│   │   ├── synthetic.py       # Physics-informed synthetic dataset generator
│   │   ├── dataset.py         # Data validation, feature engineering, group splitting
│   │   ├── train.py           # Multi-model training pipeline with GroupKFold
│   │   ├── evaluate.py        # Diagnostic metrics & plot generator
│   │   └── predict.py         # Single inference & uncertainty estimation
│   ├── preprocessing/         # (Module 2) OpenCV ROI & illumination correction
│   └── utils/
│       ├── config.py          # Central physics constants, thresholds & hyperparams
│       └── logger.py          # Rich console and file logging
└── tests/
    └── test_ml.py             # Automated unit & integration tests
```

---

## Quickstart & CLI Commands

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Training Dataset
Generates 5,000 synthetic exposure samples across 400 distinct strips:
```bash
python app.py generate-synthetic --samples 5000 --strips 400
```

### 3. Train and Benchmark Models
Trains XGBoost, Random Forest, and Ridge models with 5-fold `GroupKFold` cross-validation:
```bash
python app.py train --model all
```

### 4. Evaluate and Generate Diagnostic Plots
Computes MAE, RMSE, $R^2$, dose-tier performance, and produces diagnostic plots in `results/`:
```bash
python app.py evaluate
```
Generated diagnostic plots:
- `results/actual_vs_predicted.png`
- `results/residual_analysis.png`
- `results/error_vs_dose.png`
- `results/feature_importance.png`

### 5. Estimate Dose (Single Inference)
```bash
# Run with a benchmark test sample
python app.py predict --sample-test --temp 28 --humidity 60 --duration 240

# Or specify custom color coordinates and conditions
python app.py predict --L 52.3 --a 2.8 --b 6.2 --temp 30.0 --humidity 65.0 --duration 180.0
```

## OpenCV Reliability Improvement Plan

The current computer-vision pipeline is suitable for controlled demonstrations. The next
iteration focuses on preventing poor photographs from reaching the dose model and making
badge alignment and color calibration more robust in field conditions.

### Implementation phases

1. **Capture quality gates**
    - Measure blur, brightness, contrast, glare, and badge coverage.
    - Return diagnostics and a clear retake reason for unusable images.
    - Reject empty or invalid ROI crops before feature extraction.

2. **More reliable badge alignment**
    - Score quadrilateral candidates by area, aspect ratio, rectangularity, and corner angles.
    - Preserve the existing canonical `600x400` coordinate contract.
    - Avoid silently stretching an arbitrary image when automatic alignment fails.

3. **Robust illumination correction**
    - Use trimmed/median reference-patch statistics that ignore glare and shadows.
    - Report calibration quality and reject inconsistent reference swatches.
    - Keep channel gains bounded to prevent color over-amplification.

4. **Stable sensing features**
    - Sample the sensing patch from its interior and use robust statistics.
    - Add percentile, dark-pixel, glare, and spatial-uniformity measurements where the
      trained model contract permits them.
    - Treat expiry as `VALID`, `EXPIRED`, or `UNCERTAIN` when calibration data supports it.

5. **Real-image validation**
    - Test across cameras, lighting, blur, perspective, badge batches, and backgrounds.
    - Track boundary success rate, ROI alignment error, quality-gate accuracy, and dose error
      on physical strips that were not used during training.

### Acceptance criteria

- Poorly focused, clipped, dark, or low-contrast captures produce a deterministic retake reason.
- Valid generated-badge tests continue to pass through the complete CV and prediction pipeline.
- Reference calibration reports the number of usable swatches and bounded channel gains.
- Automatic alignment remains compatible with the existing API, debug overlays, and model feature schema.
- No claim of certified field accuracy is made until real gas-chamber calibration data is available.

---

## Scientific & Occupational Safety Guidelines

1. **Synthetic Data Disclaimer**: Synthetic data is used strictly for algorithmic development and software pipeline verification. It does NOT represent certified sensor accuracy until experimentally validated against known $H_2S$ gas exposures in controlled calibration chambers.
2. **Exposure Limits**:
   - **Safe / Normal**: $< 10.0\text{ ppm}\cdot\text{h}$
   - **Warning**: $10.0 - 40.0\text{ ppm}\cdot\text{h}$
   - **Danger / Ceiling Exceeded**: $> 40.0\text{ ppm}\cdot\text{h}$ (Approaching/exceeding 8-hr OSHA/OISD Permissible TWA limit of $10\text{ ppm}$).
