# Digital Wearable H₂S Passive Dosimeter — Team Briefing & Slide Guide

> **For:** SIH Team Members  
> **Project:** Smart Occupational H₂S Exposure Monitoring System  
> **Stack:** Python · FastAPI · OpenCV · XGBoost · React Native (Expo)

---

## 1. What Problem Are We Solving?

Workers in oil refineries, petrochemical plants, sewage treatment plants, and mines are exposed to **Hydrogen Sulfide (H₂S)** — a colorless, toxic gas that smells like rotten eggs. At high concentrations it causes unconsciousness and death within minutes.

**Current Problem:**
- Existing electronic gas detectors only measure **instantaneous peak concentration** (ppm at that moment)
- They do NOT track **cumulative exposure** over a full 8-hour shift
- Disposable colorimetric chemical badges exist but are **read manually by eye** — subjective, error-prone, and not digitally logged

**Our Solution:**
- A **camera-based AI system** that photographs a disposable chemical badge worn by the worker
- Automatically extracts color features using **Computer Vision**
- Predicts **cumulative dose (ppm·h)** using a trained **XGBoost ML model**
- Logs results to a database and displays them on a **mobile app**
- Generates **DGMS / OISD compliance audit reports**

---

## 2. What is Cumulative Dose?

```
Dose (ppm·h) = Concentration (ppm) × Time (hours)
```

Example: A worker exposed to 2 ppm H₂S for 4 hours = **8 ppm·h**

| Dose Range     | Risk Level         | Action                          |
|----------------|--------------------|---------------------------------|
| 0 – 10 ppm·h   | ✅ SAFE / NORMAL   | Continue shift normally         |
| 10 – 40 ppm·h  | ⚠️ WARNING         | Inspect ventilation, monitor    |
| 40 – 80 ppm·h  | 🚨 DANGER          | Rotate worker to fresh air      |
| > 80 ppm·h     | ☢️ CRITICAL        | Evacuate immediately            |

**OISD-STD-155 / DGMS Standard:** 8-hour TWA PEL = **10 ppm**  
Equivalent shift ceiling = **80 ppm·h**

---

## 3. How the Chemical Badge Works

The badge contains a **lead acetate (Pb(CH₃COO)₂) impregnated paper strip**.

When H₂S gas contacts the strip:
```
Pb(CH₃COO)₂  +  H₂S  →  PbS (dark brown/black)  +  2 CH₃COOH
```

- Fresh badge = **cream/white** color
- Higher exposure = **darker brown → black**
- The darkening follows **Hill kinetics** (non-linear saturation curve)
- Temperature and humidity **accelerate or slow** the reaction (Arrhenius equation)

Our system reads this color change and converts it to a dose estimate.

---

## 4. System Architecture (Big Picture)

```
┌─────────────────────────────────────────────────────────────┐
│                     MOBILE APP (React Native)               │
│  Worker photographs badge → sends to backend API            │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP POST /api/scan
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   FASTAPI BACKEND (Python)                  │
│                                                             │
│  ┌─────────────────┐    ┌──────────────────────────────┐   │
│  │  Computer Vision │    │      ML Inference Engine     │   │
│  │                 │    │                              │   │
│  │ 1. Perspective  │    │  XGBoost Model               │   │
│  │    correction   │───▶│  27 input features           │   │
│  │ 2. ROI extract  │    │  → Dose (ppm·h)              │   │
│  │ 3. Illumination │    │  → 95% Confidence Interval   │   │
│  │    normalize    │    │  → Risk Classification       │   │
│  │ 4. Color extract│    │                              │   │
│  └─────────────────┘    └──────────────────────────────┘   │
│                                    │                        │
│                                    ▼                        │
│                         ┌──────────────────┐               │
│                         │  SQLite Database  │               │
│                         │  Workers + Records│               │
│                         └──────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Project Folder Structure

```
e:/Sih/
├── h2s-camera-solution/          ← Python Backend
│   ├── app.py                    ← CLI entry point (7 commands)
│   ├── requirements.txt
│   ├── data/
│   │   ├── synthetic/            ← Generated training data (5000 samples)
│   │   ├── processed/            ← Train/test feature CSVs
│   │   └── dosimeter_records.db  ← SQLite database
│   ├── models/
│   │   ├── best_model.joblib     ← Best trained model (XGBoost)
│   │   ├── model_xgboost.joblib
│   │   ├── model_random_forest.joblib
│   │   └── model_ridge.joblib
│   ├── results/
│   │   ├── actual_vs_predicted.png
│   │   ├── residual_analysis.png
│   │   ├── error_vs_dose.png
│   │   ├── feature_importance.png
│   │   └── metrics_summary.json  ← R²=0.9854, MAE=0.97 ppm·h
│   ├── images/
│   │   ├── test/                 ← Sample badge images
│   │   └── debug/                ← Annotated ROI overlays
│   └── src/
│       ├── ml/                   ← ML pipeline
│       ├── preprocessing/        ← OpenCV pipeline
│       ├── backend/              ← FastAPI server
│       └── utils/                ← Config, logger, badge generator
│
└── h2s-mobile-app/               ← React Native App
    ├── App.js                    ← Root navigation
    ├── app.json                  ← Expo config
    ├── eas.json                  ← EAS build config (APK)
    └── src/
        ├── api/client.js         ← REST API calls
        ├── theme/colors.js       ← Design system
        ├── components/           ← DoseGauge, RiskBanner, Header, ExpiryBadge
        └── screens/              ← Dashboard, Scanner, Result, History, Simulator, Profile
```

---

## 6. ML Pipeline — How the Model Was Built

### Step 1: Synthetic Data Generation
Since real gas-chamber calibration data is expensive to collect, we built a **physics-informed synthetic dataset**:

- **5,000 samples** across **400 distinct badge strips**
- Simulates Arrhenius temperature acceleration + sigmoidal humidity response
- Generates realistic RGB, HSV, CIELAB color values for each dose level
- Adds strip-to-strip manufacturing variability (±6% sensitivity)

### Step 2: Feature Engineering (27 features)
| Category | Features |
|----------|----------|
| RGB | R_mean, G_mean, B_mean, R_std, G_std, B_std |
| HSV | H_mean, S_mean, V_mean, H_std, S_std, V_std |
| CIELAB | L_mean, a_mean, b_mean, L_std, a_std, b_std |
| Derived | delta_E, chroma, hue_angle, ratio_RG, ratio_GB, ratio_RB |
| Context | temperature_c, humidity_percent, exposure_time_minutes |

### Step 3: Zero-Leakage Grouped Validation
- Uses **GroupKFold** (5-fold) grouped by physical `strip_id`
- Guarantees no physical strip appears in both train and test
- Prevents data leakage that would inflate accuracy metrics

### Step 4: Model Comparison
| Model | CV MAE | Test R² |
|-------|--------|---------|
| **XGBoost** | **~0.97 ppm·h** | **0.9854** ✅ Best |
| Random Forest | ~1.1 ppm·h | ~0.97 |
| Ridge Regression | ~2.8 ppm·h | ~0.91 |

### Step 5: Uncertainty Estimation
- Confidence interval = **±1.96 × RMSE** (95% CI)
- RMSE on test set = **1.99 ppm·h**
- So every prediction comes with **±3.91 ppm·h** bounds

---

## 7. Computer Vision Pipeline

When a badge photo is uploaded, this 6-step pipeline runs:

```
Raw Photo
    │
    ▼
1. PERSPECTIVE CORRECTION
   - Detects 4-corner badge boundary using Canny edges + contour detection
   - Warps to canonical 600×400 px frame
   - Fallback: center-crop resize if corners not found
    │
    ▼
2. ROI EXTRACTION (6 regions)
   - Sensing patch (chemical zone): x[60–270], y[90–330]
   - 4 Reference swatches: White(100%), Light Gray(75%), Med Gray(50%), Dark(20%)
   - Expiry indicator patch: x[470–550], y[110–210]
   - 12% inner erosion margin applied to each ROI
    │
    ▼
3. LIGHTING QUALITY CHECK
   - Detects specular glare (>5% pixels near 255)
   - Checks underexposure (white patch mean < 70)
   - Estimates color temperature (Blue/Red ratio → WARM/NEUTRAL/COOL)
    │
    ▼
4. ILLUMINATION NORMALIZATION
   - Multi-point affine regression across 4 reference swatches
   - Maps measured BGR values to D65 calibrated targets
   - Per-channel: Target = slope × Measured + offset
    │
    ▼
5. EXPIRY VERIFICATION
   - Checks expiry patch CIELAB values
   - Fresh seal: a* < -4.0 AND L* > 55.0 (vivid cyan/green)
   - Expired: shifted brown/gray → badge flagged as INVALID
    │
    ▼
6. COLOR FEATURE EXTRACTION
   - Extracts 27 features from normalized sensing patch
   - Feeds into XGBoost model
```

---

## 8. Model Performance (Actual Results)

From `results/metrics_summary.json`:

| Metric | Value |
|--------|-------|
| **R² Score** | **0.9854** |
| MAE | 0.97 ppm·h |
| RMSE | 1.99 ppm·h |
| Median Absolute Error | 0.37 ppm·h |
| Test Samples | 1,004 |

**By dose tier:**
| Tier | Samples | MAE | Relative Error |
|------|---------|-----|----------------|
| Low (<5 ppm·h) | 361 | 0.22 ppm·h | 15.0% |
| Medium (5–20 ppm·h) | 432 | 0.67 ppm·h | 6.1% |
| High (>20 ppm·h) | 211 | 2.87 ppm·h | 7.1% |

> ⚠️ These metrics are on **synthetic data**. Real-world accuracy requires gas-chamber calibration experiments.

---

## 9. Mobile App — Screen by Screen

| Screen | Purpose |
|--------|---------|
| **Dashboard** | Worker greeting, current dose gauge, 7-day stats, scan button |
| **Scanner** | Camera viewfinder, ambient parameter controls, quick test presets |
| **Dose Result** | Full dose readout, annotated ROI overlay, measurement metadata |
| **History** | Filterable shift logs, DGMS/OISD compliance rate, export audit |
| **Simulator** | Render synthetic badges, run full pipeline, check accuracy % |
| **Profile** | Switch worker, configure backend URL, view safety standards |

---

## 10. API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | System health check |
| GET | `/api/health` | ML model status + calibration thresholds |
| POST | `/api/scan` | **Main endpoint** — upload badge image → get dose result |
| GET | `/api/records` | Fetch shift exposure history |
| GET | `/api/workers` | List all registered workers |
| GET | `/api/workers/{id}` | Worker profile + 7-day summary |
| GET | `/api/report/compliance` | DGMS/OISD audit report |
| POST | `/api/generate-badge` | Render synthetic badge image (base64) |

Full interactive docs at: `http://localhost:8000/docs`

---

## 11. How to Run the Project

### Backend
```bash
cd e:/Sih/h2s-camera-solution
pip install -r requirements.txt

# First time only — generate data and train model
python app.py generate-synthetic --samples 5000 --strips 400
python app.py train --model all
python app.py evaluate

# Start the API server
python app.py serve --port 8000
```

### Mobile App (Web Browser)
```bash
cd e:/Sih/h2s-mobile-app
npm install
npm run web
# Opens at http://localhost:8081
```

### Mobile App (Physical Android Phone)
```bash
# Install Expo Go from Play Store on your phone
npx expo start
# Scan the QR code with Expo Go
# Set backend URL in Profile tab to your PC's LAN IP
```

### Standalone APK Build
```bash
npm install -g eas-cli
eas login                                    # Needs free expo.dev account
eas build --platform android --profile preview
# Download APK from the link provided
```

---

## 12. Slide Deck Structure (Suggested 10 Slides)

| Slide | Title | Key Content |
|-------|-------|-------------|
| 1 | **Title Slide** | Project name, team, SIH problem statement |
| 2 | **The Problem** | H₂S hazard stats, limitations of current detectors, need for cumulative dosimetry |
| 3 | **Our Solution** | System overview diagram (badge → camera → AI → app → compliance) |
| 4 | **The Chemical Badge** | Pb(Ac)₂ → PbS reaction, color progression photos, why camera works |
| 5 | **Computer Vision Pipeline** | 6-step pipeline diagram, ROI overlay image from `images/debug/` |
| 6 | **ML Model** | Feature table, GroupKFold diagram, model comparison table, R²=0.9854 |
| 7 | **Model Results** | Actual vs Predicted plot, residual plot (from `results/` folder) |
| 8 | **Mobile App Demo** | Screenshots of Dashboard, Scanner, Dose Result screens |
| 9 | **Compliance & Impact** | OISD-STD-155 alignment, audit report, industries targeted |
| 10 | **Roadmap** | Real gas-chamber calibration → certified accuracy → deployment |

### Images to use in slides (already in project):
- `results/actual_vs_predicted.png` — Slide 7
- `results/feature_importance.png` — Slide 6
- `results/residual_analysis.png` — Slide 7
- `results/error_vs_dose.png` — Slide 7
- `images/debug/overlay_*.png` — Slide 5 (ROI detection demo)
- `images/test/badge_*.png` — Slide 4 (badge color progression)

---

## 13. Key Technical Terms (Glossary for Teammates)

| Term | Meaning |
|------|---------|
| **ppm** | Parts per million — unit of gas concentration |
| **ppm·h** | ppm × hours — cumulative dose unit |
| **TWA** | Time-Weighted Average — average concentration over 8 hours |
| **PEL** | Permissible Exposure Limit — legal maximum (10 ppm for H₂S) |
| **CIELAB** | Perceptually uniform color space (L*a*b*) — better than RGB for color measurement |
| **Delta E (ΔE*)** | Total color difference from a reference point |
| **Arrhenius** | Equation describing how temperature speeds up chemical reactions |
| **Hill Kinetics** | S-shaped saturation curve for chemical reaction progress |
| **GroupKFold** | Cross-validation that keeps data from the same physical strip together |
| **XGBoost** | Gradient boosted decision tree — our best performing ML model |
| **ROI** | Region of Interest — specific cropped area of the badge image |
| **OISD-STD-155** | Indian Oil Industry Safety Directorate standard for H₂S exposure |
| **DGMS** | Directorate General of Mines Safety — Indian mining safety regulator |
| **FastAPI** | Python web framework for building REST APIs |
| **Expo / EAS** | Tools for building React Native apps and APKs |

---

## 14. Limitations & Honest Disclaimers

1. **Synthetic Training Data** — The ML model is trained on physics-simulated data, not real gas-chamber experiments. Accuracy on real badges needs validation.
2. **Lighting Sensitivity** — Extreme lighting (direct sunlight, very dim) can affect color extraction despite normalization.
3. **Badge Standardization** — The CV pipeline assumes a specific badge layout (600×400 canonical frame). Different badge designs need reconfiguration.
4. **No Real-Time Monitoring** — This is a **shift-end scan** system, not continuous monitoring. Workers scan their badge at end of shift.
5. **Android Only APK** — iOS requires Apple Developer account ($99/year) for standalone build.

---

## 15. Future Roadmap

- [ ] Real gas-chamber calibration experiments to replace synthetic data
- [ ] Continuous Bluetooth-enabled badge with periodic auto-scan
- [ ] Multi-gas support (CO, SO₂, NO₂)
- [ ] Edge deployment on Raspberry Pi for offline industrial use
- [ ] Integration with plant SCADA / DCS systems
- [ ] iOS App Store release

---

*Generated for SIH Team Briefing — Project: Digital Wearable H₂S Passive Dosimeter*
