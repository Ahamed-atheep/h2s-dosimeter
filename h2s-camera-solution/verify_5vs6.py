import sys, os, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.ml.predict import DosimeterPredictor
from src.utils.badge_generator import generate_test_badge_image

p = DosimeterPredictor()
for dose, bid in [(5.0, "H2S-B20-00550"), (6.0, "H2S-B20-00551")]:
    img, _ = generate_test_badge_image(
        dose_ppm_h=dose, temp_c=28.0, humidity_pct=55.0,
        lighting="daylight", tilt_angle=0.0, is_expired=False,
        badge_id=bid, batch_id="B20", calibration_version="CAL-03"
    )
    res, _ = p.predict_from_image(img, temp_c=28.0, humidity_pct=55.0, exposure_time_min=240.0)
    print(f"Actual {dose} ppm*h  ->  Predicted {res.cumulative_dose_ppm_h} ppm*h  |  {res.risk_level}")
