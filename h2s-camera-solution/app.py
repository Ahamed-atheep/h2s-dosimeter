"""
H2S Digital Wearable Passive Dosimeter — CLI Application.
Entry point for image feature extraction, dataset generation, model training,
evaluation, and cumulative dose estimation.
"""

import sys
import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Ensure local package import
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.utils.config import (
    SYNTHETIC_DATA_DIR, MODELS_DIR, RESULTS_DIR, TEST_IMAGES_DIR, IMAGES_DIR, physics_config
)
from src.utils.logger import get_logger
from src.utils.badge_generator import generate_test_badge_image
from src.preprocessing.color_features import process_badge_image_pipeline
from src.ml.synthetic import generate_synthetic_dataset
from src.ml.train import train_model_pipeline, train_and_compare_all_models
from src.ml.evaluate import evaluate_model_performance
from src.ml.predict import DosimeterPredictor

console = Console(highlight=False)
logger = get_logger("H2S_CLI")


def cmd_generate_synthetic(args):
    """Generate physically-grounded synthetic exposure dataset."""
    console.print(Panel(
        f"[bold cyan]Generating Synthetic H2S Exposure Dataset[/bold cyan]\n"
        f"Samples: {args.samples} | Strips: {args.strips} | Seed: {args.seed}",
        title="[bold]Step: Data Simulation[/bold]",
        border_style="cyan"
    ))
    
    out_path = Path(args.output) if args.output else None
    df, meta = generate_synthetic_dataset(
        n_samples=args.samples,
        n_strips=args.strips,
        random_seed=args.seed,
        output_path=out_path
    )
    
    console.print(f"\n[bold green][OK] Successfully generated {len(df)} samples across {args.strips} strips.[/bold green]")
    console.print(f"[dim]Dose Range: {df['cumulative_dose_ppm_h'].min():.2f} to {df['cumulative_dose_ppm_h'].max():.2f} ppm*h[/dim]")
    console.print(f"[dim]Saved to: {meta.get('output_path', out_path or SYNTHETIC_DATA_DIR / 'synthetic_h2s_exposure_dataset.csv')}[/dim]\n")


def cmd_generate_badge(args):
    """Generate a realistic synthetic badge test image."""
    console.print(Panel(
        f"[bold magenta]Rendering Realistic Test Badge Image[/bold magenta]\n"
        f"Exposure Dose: {args.exposure:.1f} ppm*h | Lighting: {args.lighting.upper()} | Tilt: {args.tilt}° | Expired: {args.expired}",
        title="[bold]Step: Badge Image Simulator[/bold]",
        border_style="magenta"
    ))
    
    out_path = Path(args.output) if args.output else None
    img, saved_path = generate_test_badge_image(
        dose_ppm_h=args.exposure,
        temp_c=args.temp,
        humidity_pct=args.humidity,
        is_expired=args.expired,
        lighting=args.lighting,
        add_shadow=args.shadow,
        tilt_angle=args.tilt,
        output_path=out_path
    )
    
    console.print(f"\n[bold green][OK] Badge image generated successfully ({img.shape[1]}x{img.shape[0]} px)[/bold green]")
    console.print(f"[dim]Saved to: {saved_path}[/dim]\n")


def cmd_extract(args):
    """Extract ROIs, illumination correction, and color features from an image."""
    image_path = Path(args.image)
    if not image_path.exists():
        console.print(f"[bold red]Error: Image file '{image_path}' not found.[/bold red]")
        sys.exit(1)
        
    console.print(Panel(
        f"[bold blue]Processing Badge Image & Extracting Color Features[/bold blue]\n"
        f"Image: {image_path.name}",
        title="[bold]Step: CV Preprocessing[/bold]",
        border_style="blue"
    ))
    
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        console.print(f"[bold red]Error: Failed to decode image '{image_path}'.[/bold red]")
        sys.exit(1)
        
    pipeline_res = process_badge_image_pipeline(
        image_bgr, temp_c=args.temp, humidity_pct=args.humidity, exposure_time_min=args.duration
    )
    
    qa = pipeline_res["lighting_qa"]
    exp = pipeline_res["expiry_report"]
    norm = pipeline_res["normalization_diagnostics"]
    feat = pipeline_res["color_features"]
    
    # 1. Lighting QA Table
    table_qa = Table(title="Lighting & Image Quality Diagnostics", border_style="cyan")
    table_qa.add_column("Check", style="cyan")
    table_qa.add_column("Result", style="white")
    table_qa.add_row("Lighting Profile", f"[bold]{qa['lighting_profile']}[/bold]")
    table_qa.add_row("White Patch Intensity", f"{qa['white_patch_intensity']} / 255")
    table_qa.add_row("Specular Glare Detected", f"{'[red]YES[/red]' if qa['is_glare_detected'] else '[green]NO[/green]'} ({qa['glare_pixel_percentage']}%)")
    table_qa.add_row("Lighting Suitable for Readout", f"{'[green]PASS[/green]' if qa['is_lighting_suitable'] else '[yellow]WARNING[/yellow]'}")
    table_qa.add_row("Auto Perspective Alignment", f"{'[green]APPLIED[/green]' if pipeline_res['is_auto_aligned'] else '[yellow]FALLBACK_FRAME[/yellow]'}")
    console.print(table_qa)
    
    # 2. Expiry Table
    table_exp = Table(title="Badge Shelf-Life / Expiry Verification", border_style="yellow")
    table_exp.add_column("Property", style="yellow")
    table_exp.add_column("Status", style="white")
    exp_color = "green" if exp["is_valid"] else "bold red"
    table_exp.add_row("Badge Expiry Status", f"[{exp_color}]{exp['expiry_status']}[/{exp_color}]")
    table_exp.add_row("Estimated Degradation", f"{exp['degradation_percentage']}%")
    table_exp.add_row("Recommendation", f"[italic]{exp['recommendation']}[/italic]")
    console.print(table_exp)
    
    # 3. Features Table
    table_feat = Table(title="Extracted Multi-Space Color Statistics (Normalized Sensing Patch)", border_style="green")
    table_feat.add_column("Color Space / Metric", style="green")
    table_feat.add_column("Extracted Value", style="white", justify="right")
    table_feat.add_row("CIELAB Lightness (L*)", f"{feat['L_mean']:.2f} ± {feat['L_std']:.2f}")
    table_feat.add_row("CIELAB a* (Red-Green)", f"{feat['a_mean']:.2f} ± {feat['a_std']:.2f}")
    table_feat.add_row("CIELAB b* (Blue-Yellow)", f"{feat['b_mean']:.2f} ± {feat['b_std']:.2f}")
    table_feat.add_row("Total Color Difference (Delta E*)", f"[bold magenta]{feat['delta_E']:.2f}[/bold magenta]")
    table_feat.add_row("RGB Means (R, G, B)", f"({feat['R_mean']:.1f}, {feat['G_mean']:.1f}, {feat['B_mean']:.1f})")
    table_feat.add_row("HSV Means (H, S, V)", f"({feat['H_mean']:.1f}°, {feat['S_mean']:.1f}%, {feat['V_mean']:.1f}%)")
    table_feat.add_row("Cross-Channel Ratios (R/G, G/B)", f"{feat['ratio_RG']:.3f}, {feat['ratio_GB']:.3f}")
    console.print(table_feat)
    
    # Save output features if requested
    if args.output:
        out_json_path = Path(args.output)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(feat, f, indent=2)
        console.print(f"[dim]Features JSON saved to: {out_json_path}[/dim]")
        
    # Save debug overlay
    if args.visualize:
        debug_dir = IMAGES_DIR / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir / f"overlay_{image_path.name}"
        cv2.imwrite(str(debug_path), pipeline_res["debug_overlay"])
        console.print(f"[bold green]Annotated ROI overlay saved to: {debug_path}[/bold green]\n")


def cmd_train(args):
    """Train ML models on dataset."""
    console.print(Panel(
        f"[bold yellow]Training H2S Dose Regression Models[/bold yellow]\n"
        f"Model: {args.model.upper()} | Data: {args.data or 'Default Synthetic'}",
        title="[bold]Step: Model Training[/bold]",
        border_style="yellow"
    ))
    
    data_path = Path(args.data) if args.data else None
    
    if args.model.lower() == "all":
        train_and_compare_all_models(data_path=data_path)
    else:
        train_model_pipeline(data_path=data_path, model_type=args.model.lower(), save_artifact=True)


def cmd_evaluate(args):
    """Evaluate trained model against test data."""
    console.print(Panel(
        f"[bold blue]Evaluating Model Performance & Generating Diagnostic Plots[/bold blue]\n"
        f"Model: {args.model_path or 'Default Best Model'} | Test Data: {args.test_data or 'Processed Test Features'}",
        title="[bold]Step: Model Evaluation[/bold]",
        border_style="blue"
    ))
    
    m_path = Path(args.model_path) if args.model_path else None
    t_path = Path(args.test_data) if args.test_data else None
    
    metrics = evaluate_model_performance(model_artifact_path=m_path, test_data_path=t_path, save_plots=True)
    
    overall = metrics["overall_metrics"]
    table = Table(title="Overall Test Performance Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta", justify="right")
    table.add_row("Mean Absolute Error (MAE)", f"{overall['MAE_ppm_h']:.3f} ppm*h")
    table.add_row("Root Mean Squared Error (RMSE)", f"{overall['RMSE_ppm_h']:.3f} ppm*h")
    table.add_row("Coefficient of Determination (R2)", f"{overall['R2_score']:.4f}")
    table.add_row("Median Absolute Error (MedAE)", f"{overall['Median_Absolute_Error_ppm_h']:.3f} ppm*h")
    table.add_row("Maximum Absolute Error", f"{overall['Max_Absolute_Error_ppm_h']:.3f} ppm*h")
    console.print(table)
    
    tier = metrics["dose_tier_breakdown"]
    t_table = Table(title="Performance Breakdown by Cumulative Exposure Tier")
    t_table.add_column("Dose Tier", style="cyan")
    t_table.add_column("Samples", justify="right")
    t_table.add_column("MAE (ppm*h)", style="magenta", justify="right")
    t_table.add_column("RMSE (ppm*h)", style="red", justify="right")
    t_table.add_column("Rel. Error (%)", justify="right")
    
    for key, val in tier.items():
        if val["count"] > 0:
            t_table.add_row(
                val["name"],
                str(val["count"]),
                f"{val['mae']:.3f}",
                f"{val['rmse']:.3f}",
                f"{val['mean_relative_error_pct']:.1f}%"
            )
    console.print(t_table)
    console.print(f"\n[green]Diagnostic plots successfully saved to: {RESULTS_DIR}[/green]\n")


def cmd_predict(args):
    """Run single inference prediction from image or direct features."""
    predictor = DosimeterPredictor(model_artifact_path=Path(args.model_path) if args.model_path else None)
    
    if args.image:
        # End-to-end inference directly from camera image
        image_path = Path(args.image)
        if not image_path.exists():
            console.print(f"[bold red]Error: Image '{image_path}' does not exist.[/bold red]")
            sys.exit(1)
            
        result, pipe_res = predictor.predict_from_image(
            image_path, temp_c=args.temp, humidity_pct=args.humidity, exposure_time_min=args.duration
        )
        
        # Save visual overlay
        debug_dir = IMAGES_DIR / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir / f"predicted_{image_path.name}"
        cv2.imwrite(str(debug_path), pipe_res["debug_overlay"])
        
    elif args.sample_test:
        # Predefined test sample (fresh / lightly exposed strip at ~1.5 ppm for 2 hours = 3.0 ppm*h)
        features = {
            "R_mean": 210.0, "G_mean": 210.0, "B_mean": 198.0,
            "R_std": 2.2, "G_std": 2.2, "B_std": 2.0,
            "H_mean": 30.0, "S_mean": 6.0, "V_mean": 82.0,
            "H_std": 1.2, "S_std": 1.2, "V_std": 1.2,
            "L_mean": 82.5, "a_mean": 0.2, "b_mean": 4.1,
            "L_std": 1.4, "a_std": 0.6, "b_std": 0.8,
            "temperature_c": args.temp,
            "humidity_percent": args.humidity,
            "exposure_time_minutes": args.duration
        }
        result = predictor.predict_single(features)
    elif args.json_input:
        with open(args.json_input, "r", encoding="utf-8") as f:
            features = json.load(f)
            features["temperature_c"] = features.get("temperature_c", args.temp)
            features["humidity_percent"] = features.get("humidity_percent", args.humidity)
            features["exposure_time_minutes"] = features.get("exposure_time_minutes", args.duration)
        result = predictor.predict_single(features)
    else:
        # Compute consistent synchronized RGB/HSV if only Lab L was passed
        l_val = args.L
        # Reaction progress approximate from L: alpha = (92.0 - L) / (92.0 - 22.0)
        alpha = float(np.clip((92.0 - l_val) / 70.0, 0.0, 1.0))
        r_calc = float(np.clip(235.0 - alpha * (235.0 - 45.0), 30, 255))
        g_calc = float(np.clip(235.0 - alpha * (235.0 - 38.0), 30, 255))
        b_calc = float(np.clip(225.0 - alpha * (225.0 - 32.0), 25, 255))
        
        v_calc = float(np.clip(92.0 - alpha * 74.0, 10, 100))
        s_calc = float(np.clip(4.0 + alpha * 28.0, 0, 60))
        h_calc = float(np.clip(28.0 + alpha * 8.0, 10, 60))
        
        features = {
            "L_mean": args.L, "a_mean": args.a, "b_mean": args.b,
            "L_std": 1.5, "a_std": 0.8, "b_std": 1.0,
            "R_mean": r_calc, "G_mean": g_calc, "B_mean": b_calc,
            "R_std": 2.5, "G_std": 2.5, "B_std": 2.2,
            "H_mean": h_calc, "S_mean": s_calc, "V_mean": v_calc,
            "H_std": 1.5, "S_std": 1.5, "V_std": 1.5,
            "temperature_c": args.temp,
            "humidity_percent": args.humidity,
            "exposure_time_minutes": args.duration
        }
        result = predictor.predict_single(features)
        
    # Styled Output Table
    color_tag = "green" if "SAFE" in result.risk_level else ("yellow" if "WARNING" in result.risk_level else "bold red")
    
    table = Table(title="H2S Dosimeter Cumulative Dose Estimation", show_header=False, border_style="cyan")
    table.add_column("Field", style="bold cyan")
    table.add_column("Value", style="bold white")
    
    if args.image:
        table.add_row("Input Source", f"Image: {Path(args.image).name}")
        table.add_row("Badge Shelf-Life Status", f"{'[green]VALID[/green]' if result.is_badge_valid else '[bold red]EXPIRED / COMPROMISED[/bold red]'}")
        table.add_row("Illumination Condition", f"{result.lighting_profile}")
        
    table.add_row("Estimated Cumulative Dose", f"[{color_tag}]{result.cumulative_dose_ppm_h:.2f} ppm*h[/{color_tag}]")
    table.add_row("Confidence Interval (95%)", f"[{result.confidence_interval_low:.2f} - {result.confidence_interval_high:.2f}] ppm*h (+/-{result.uncertainty_margin:.2f})")
    table.add_row("8-hr Equivalent TWA", f"{result.twa_8h_equivalent_ppm:.2f} ppm")
    table.add_row("Occupational Risk Level", f"[{color_tag}]{result.risk_level}[/{color_tag}]")
    table.add_row("Ambient Conditions", f"{result.temperature_c:.1f}°C | {result.humidity_percent:.1f}% RH")
    table.add_row("Model Architecture", f"{result.model_type.upper()} ({'Synthetic-Trained' if result.is_synthetic_model else 'Experimental-Calibrated'})")
    table.add_row("Safety Action", f"[italic]{result.action_recommendation}[/italic]")
    
    console.print(table)
    if args.image:
        console.print(f"[dim]Annotated prediction overlay saved to: images/debug/predicted_{Path(args.image).name}[/dim]\n")


def cmd_live_zones(args):
    """
    Run real-time dynamic zone detection & tracking on webcam, video file, or test image.
    Demonstrates object-detection style bounding boxes that dynamically follow the badge as it moves/rotates.
    """
    from src.preprocessing.zone_detector import BadgeZoneDetector, draw_zone_detections
    import time
    
    console.print(Panel(
        f"[bold green]Live Camera Dynamic Zone Detection & Object-Tracking HUD[/bold green]\n"
        f"Source: {'Webcam Device ' + str(args.camera) if not args.input else args.input}\n"
        f"Controls: [bold white]S[/bold white] = Capture & Predict | [bold white]T[/bold white] = Toggle HUD | [bold white]Q/ESC[/bold white] = Exit",
        title="[bold]Computer Vision Live Zone Detection[/bold]",
        border_style="green"
    ))
    
    detector = BadgeZoneDetector(use_temporal_smoothing=True, smoothing_factor=args.smoothing)
    show_hud = True
    
    # If a static image input was provided
    if args.input and Path(args.input).is_file() and Path(args.input).suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
        img = cv2.imread(str(args.input))
        if img is None:
            console.print(f"[bold red]Error: Failed to load image '{args.input}'[/bold red]")
            sys.exit(1)
            
        res = detector.detect_zones(img)
        annotated = draw_zone_detections(img, res, show_hud=True)
        
        out_path = IMAGES_DIR / "debug" / f"zones_{Path(args.input).name}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated)
        
        console.print(f"\n[bold green][OK] Processed image '{Path(args.input).name}'[/bold green]")
        console.print(f"Badge Detected: {'[green]YES[/green]' if res['badge_detected'] else '[red]NO[/red]'}")
        console.print(f"Tracking Confidence: {res['confidence']:.2%}")
        console.print(f"Rotation Angle: {res['rotation_deg']:.1f}°")
        console.print(f"Detected Zones: {len(res['zones'])} / 6 zones")
        for z in res["zones"]:
            console.print(f"  • {z.icon} [bold]{z.name}[/bold] ({z.category}): {z.confidence:.1%} confidence")
        console.print(f"[dim]Annotated output saved to: {out_path}[/dim]\n")
        
        if not args.no_window:
            try:
                cv2.imshow("H2S Dosimeter — Dynamic Zone Detection", annotated)
                console.print("[dim]Press any key in window to close...[/dim]")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except cv2.error as e:
                console.print(f"[yellow]Note: GUI window not available ({e.args[0].split(':')[0] if e.args else e}). Annotated image saved to: {out_path}[/yellow]")
        return

    # Live Webcam or Video Stream
    src = int(args.camera) if not args.input else args.input
    cap = cv2.VideoCapture(src)
    
    if not cap.isOpened():
        console.print(f"[bold red]Error: Could not open video source {src}. Check webcam connection.[/bold red]")
        sys.exit(1)
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    prev_time = time.time()
    fps_smooth = 30.0
    has_gui = True
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                console.print("[yellow]End of video stream or camera disconnected.[/yellow]")
                break
                
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt)
                
            # Dynamic Zone Detection with Temporal Tracking
            res = detector.detect_zones(frame)
            annotated = draw_zone_detections(frame, res, show_hud=show_hud, show_fps=fps_smooth)
            
            if has_gui:
                try:
                    cv2.imshow("H2S Dosimeter — Live Camera Zone Detection HUD", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord("q"), ord("Q"), 27]:  # Q or ESC
                        break
                    elif key in [ord("t"), ord("T")]:
                        show_hud = not show_hud
                    elif key in [ord("s"), ord("S"), 32]:  # S or Space
                        # Trigger instant capture & inference
                        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                        snap_path = IMAGES_DIR / "debug" / f"live_snap_{timestamp_str}.png"
                        snap_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(snap_path), frame)
                        
                        pred = DosimeterPredictor()
                        result, pipe_res = pred.predict_from_image(snap_path, temp_c=25.0, humidity_pct=50.0)
                        
                        overlay_path = IMAGES_DIR / "debug" / f"live_analysis_{timestamp_str}.png"
                        cv2.imwrite(str(overlay_path), pipe_res["debug_overlay"])
                        
                        console.print(f"\n[bold green]Captured frame & analyzed![/bold green]")
                        console.print(f"Dose: [bold cyan]{result.cumulative_dose_ppm_h:.2f} ppm*h[/bold cyan] ({result.risk_level})")
                        console.print(f"Saved analysis to: {overlay_path}\n")
                except cv2.error:
                    has_gui = False
                    console.print("[yellow]Headless OpenCV detected (no GUI window support). Running in headless mode — saving preview frames to images/debug/live_hud_stream.png[/yellow]")
            else:
                # Headless mode loop: periodically update live_hud_stream.png
                stream_path = IMAGES_DIR / "debug" / "live_hud_stream.png"
                stream_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(stream_path), annotated)
                time.sleep(0.05)
    finally:
        cap.release()
        if has_gui:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


def cmd_serve(args):
    """Start the FastAPI REST backend server."""
    import uvicorn
    console.print(Panel(
        f"[bold green]Starting H2S Dosimeter REST API Server[/bold green]\n"
        f"Host: {args.host} | Port: {args.port} | API Docs: http://localhost:{args.port}/docs",
        title="[bold]FastAPI Backend Service[/bold]",
        border_style="green"
    ))
    uvicorn.run("src.backend.api:app", host=args.host, port=args.port, reload=args.reload)


def main():
    parser = argparse.ArgumentParser(
        description="Digital Wearable H2S Passive Dosimeter — ML & Computer Vision Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # 1. generate-synthetic
    p_gen = subparsers.add_parser("generate-synthetic", help="Generate physically grounded synthetic exposure dataset")
    p_gen.add_argument("--samples", type=int, default=5000, help="Total number of samples")
    p_gen.add_argument("--strips", type=int, default=400, help="Number of distinct strip groups")
    p_gen.add_argument("--seed", type=int, default=42, help="Random seed")
    p_gen.add_argument("--output", type=str, default=None, help="Custom output CSV path")
    
    # 2. generate-badge
    p_badge = subparsers.add_parser("generate-badge", help="Render a photorealistic test badge image with optical artifacts")
    p_badge.add_argument("--exposure", type=float, default=12.0, help="Simulated cumulative dose in ppm*h")
    p_badge.add_argument("--temp", type=float, default=28.0, help="Ambient temperature in °C")
    p_badge.add_argument("--humidity", type=float, default=55.0, help="Relative humidity in %")
    p_badge.add_argument("--lighting", type=str, default="daylight", choices=["daylight", "warm", "cool", "fluorescent", "shadow"], help="Lighting condition")
    p_badge.add_argument("--shadow", action="store_true", help="Add linear shadow gradient")
    p_badge.add_argument("--tilt", type=float, default=0.0, help="Camera perspective tilt angle (-20 to +20 degrees)")
    p_badge.add_argument("--expired", action="store_true", help="Simulate an expired/degraded badge")
    p_badge.add_argument("--output", type=str, default=None, help="Output image file path")
    
    # 3. extract
    p_ext = subparsers.add_parser("extract", help="Extract ROIs, illumination correction, and color features from an image")
    p_ext.add_argument("--image", type=str, required=True, help="Path to input badge image")
    p_ext.add_argument("--output", type=str, default=None, help="Path to save extracted features JSON")
    p_ext.add_argument("--visualize", action="store_true", help="Save annotated ROI debug overlay")
    p_ext.add_argument("--temp", type=float, default=25.0, help="Ambient temperature in °C")
    p_ext.add_argument("--humidity", type=float, default=50.0, help="Relative humidity in %")
    p_ext.add_argument("--duration", type=float, default=240.0, help="Exposure duration in minutes")
    
    # 4. live-zones / live-detect
    p_live = subparsers.add_parser("live-zones", aliases=["live-detect"], help="Launch live camera dynamic zone detection and object-tracking HUD")
    p_live.add_argument("--camera", type=int, default=0, help="Webcam device index (default 0)")
    p_live.add_argument("--input", type=str, default=None, help="Optional image or video file path to test")
    p_live.add_argument("--smoothing", type=float, default=0.40, help="Temporal tracking smoothing factor (0.1 to 0.9)")
    p_live.add_argument("--no-window", action="store_true", help="Do not display GUI window (for headless scripts)")
    
    # 5. train
    p_train = subparsers.add_parser("train", help="Train dose regression models")
    p_train.add_argument("--data", type=str, default=None, help="Path to training CSV dataset")
    p_train.add_argument("--model", type=str, default="all", choices=["xgboost", "random_forest", "ridge", "all"], help="Model architecture")
    
    # 6. evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate model performance and produce diagnostic plots")
    p_eval.add_argument("--model-path", type=str, default=None, help="Path to trained .joblib model")
    p_eval.add_argument("--test-data", type=str, default=None, help="Path to test CSV features")
    
    # 7. predict
    p_pred = subparsers.add_parser("predict", help="Estimate cumulative H2S dose from an image or color/ambient features")
    p_pred.add_argument("--image", type=str, default=None, help="Path to badge image (runs full CV + ML pipeline)")
    p_pred.add_argument("--model-path", type=str, default=None, help="Path to .joblib model")
    p_pred.add_argument("--sample-test", action="store_true", help="Run with a built-in benchmark test sample")
    p_pred.add_argument("--json-input", type=str, default=None, help="Path to JSON file with feature values")
    p_pred.add_argument("--L", type=float, default=55.0, help="CIELAB L* mean")
    p_pred.add_argument("--a", type=float, default=2.5, help="CIELAB a* mean")
    p_pred.add_argument("--b", type=float, default=6.0, help="CIELAB b* mean")
    p_pred.add_argument("--temp", type=float, default=28.0, help="Ambient temperature in °C")
    p_pred.add_argument("--humidity", type=float, default=60.0, help="Relative humidity in %")
    p_pred.add_argument("--duration", type=float, default=240.0, help="Exposure duration in minutes")
    
    # 8. serve
    p_serve = subparsers.add_parser("serve", help="Launch FastAPI REST API server for mobile and web clients")
    p_serve.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    p_serve.add_argument("--port", type=int, default=8000, help="Server port")
    p_serve.add_argument("--reload", action="store_true", help="Enable hot reload")
    
    args = parser.parse_args()
    
    if args.command == "generate-synthetic":
        cmd_generate_synthetic(args)
    elif args.command == "generate-badge":
        cmd_generate_badge(args)
    elif args.command == "extract":
        cmd_extract(args)
    elif args.command in ["live-zones", "live-detect"]:
        cmd_live_zones(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

