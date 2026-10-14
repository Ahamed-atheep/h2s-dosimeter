"""
Pydantic Schemas for FastAPI API Request & Response validation.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime


class WorkerBase(BaseModel):
    id: str
    name: str
    department: str
    designation: str
    active: bool = True


class WorkerCreate(WorkerBase):
    pass


class WorkerResponse(WorkerBase):
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class ValidateQRRequest(BaseModel):
    qr_payload: str = Field(..., description="Raw QR string or JSON payload from physical badge")
    reference_date: Optional[str] = Field(None, description="Optional YYYY-MM-DD reference date for expiry check")


class ValidateQRResponse(BaseModel):
    is_valid: bool
    badge_status: str  # "VALID" | "EXPIRED" | "INVALID_FORMAT"
    badge_id: Optional[str] = None
    batch_id: Optional[str] = None
    manufactured_date: Optional[str] = None
    expiry_date: Optional[str] = None
    calibration_version: Optional[str] = None
    message: str


class ExposureRecordResponse(BaseModel):
    id: int
    timestamp: datetime
    worker_id: str
    badge_id: str
    batch_id: Optional[str] = "B17"
    calibration_version: Optional[str] = "CAL-03"
    badge_status: Optional[str] = "VALID"
    cumulative_dose_ppm_h: float
    confidence_interval_low: float
    confidence_interval_high: float
    uncertainty_margin: float
    twa_8h_ppm: float
    risk_level: str
    action_recommendation: Optional[str]
    temperature_c: float
    humidity_percent: float
    exposure_duration_minutes: float
    is_badge_valid: bool
    expiry_status: str
    reliability_score: Optional[float] = 0.95
    lighting_profile: str
    model_type: str
    model_version: Optional[str] = "v1.0-xgboost"
    image_filename: Optional[str]
    measurement_source: Optional[str] = "CAMERA"
    spectral_data: Optional[str] = None
    gas_resistance_ohms: Optional[float] = None
    pressure_hpa: Optional[float] = None
    is_saturated: Optional[bool] = False
    led_current_ma: Optional[int] = None
    model_config = ConfigDict(from_attributes=True)


class ScanResponse(BaseModel):
    success: bool
    record_id: Optional[int] = None
    worker_id: str
    badge_id: str = "H2S-B17-00482"
    batch_id: str = "B17"
    calibration_version: str = "CAL-03"
    badge_status: str = "VALID"
    expiry_date: Optional[str] = None
    cumulative_dose_ppm_h: float
    dose_ppm_h: Optional[float] = None
    confidence_interval_low: float
    confidence_interval_high: float
    uncertainty_margin: float
    twa_8h_equivalent_ppm: float
    twa_ppm: Optional[float] = None
    risk_level: str
    risk: Optional[str] = None
    risk_color: str
    action_recommendation: str
    model_type: str
    model_version: Optional[str] = "v1.0-xgboost"
    is_synthetic_model: bool
    temperature_c: float
    humidity_percent: float
    exposure_duration_minutes: float
    is_badge_valid: bool
    expiry_status: str
    reliability_score: Optional[float] = 0.95
    image_quality: Optional[str] = "HIGH"
    lighting_profile: str
    overlay_image_url: Optional[str] = None
    color_features: Optional[Dict[str, float]] = None
    measurement_source: Optional[str] = "CAMERA"


class HardwareReadingRequest(BaseModel):
    worker_id: str = Field(default="WRK-1001", description="Worker ID to associate scan with")
    badge_id: Optional[str] = Field(default="H2S-HW-001", description="Badge Serial / Identifier")
    batch_id: Optional[str] = Field(default="HW-01", description="Batch identifier")
    calibration_version: Optional[str] = Field(default="AS7343-CAL-01", description="Spectral ratio calibration version")
    channels: List[float] = Field(..., description="Array of AS7343 channel counts (12, 14, or 18 elements)")
    clear_count: Optional[float] = Field(None, description="Clear / VIS broadband channel count")
    temperature_c: float = Field(default=25.0, description="BME688 Ambient Temperature °C")
    humidity_percent: float = Field(default=50.0, description="BME688 Ambient Relative Humidity %")
    pressure_hpa: Optional[float] = Field(default=1013.25, description="BME688 Barometric Pressure hPa")
    gas_resistance_ohms: Optional[float] = Field(default=50000.0, description="BME688 Gas resistance Ω")
    exposure_duration_minutes: float = Field(default=240.0, description="Badge exposure duration in minutes")
    is_saturated: bool = Field(default=False, description="AS7343 analog or digital saturation flag")
    led_current_ma: Optional[int] = Field(default=25, description="AS7343 LED drive current in mA")
    firmware_dose_ppm_h: Optional[float] = Field(None, description="Optional dose calculated on-device by ESP32")
    firmware_risk_tier: Optional[str] = Field(None, description="Risk tier reported by ESP32")


class HardwareReadingResponse(BaseModel):
    success: bool
    record_id: Optional[int] = None
    worker_id: str
    badge_id: str
    measurement_source: str = "SPECTRAL_HARDWARE_AS7343"
    cumulative_dose_ppm_h: float
    dose_ppm_h: float
    twa_8h_equivalent_ppm: float
    twa_ppm: float
    risk_level: str
    risk_tier: str
    risk_color: str
    action_recommendation: str
    temperature_c: float
    humidity_percent: float
    pressure_hpa: Optional[float] = None
    gas_resistance_ohms: Optional[float] = None
    exposure_duration_minutes: float
    is_saturated: bool
    led_current_ma: Optional[int] = None
    clear_count: float
    ratios: Dict[str, float]
    raw_channels: Dict[str, float]
    model_type: str = "RANDOM_FOREST_AS7343"
    model_version: str = "v1.0-ratio-50trees"
    timestamp: datetime


class HardwareStationStatus(BaseModel):
    station_ip: str
    connected: bool
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    pressure: Optional[float] = None
    gas_resistance: Optional[float] = None
    led_on: Optional[bool] = None
    led_current_ma: Optional[int] = None
    saturated: Optional[bool] = None
    latest_dose_ppm_h: Optional[float] = None
    risk_tier: Optional[str] = None
    badge_active: Optional[bool] = None
    elapsed_hours: Optional[float] = None
    channels: Optional[List[int]] = None
    error: Optional[str] = None


class HardwareLedCommandRequest(BaseModel):
    station_ip: str
    led_on: Optional[bool] = None
    current_ma: Optional[int] = None


class WorkerSummaryResponse(BaseModel):
    worker: WorkerResponse
    total_scans: int
    recent_dose_ppm_h: float
    cumulative_7day_dose_ppm_h: float
    average_twa_ppm: float
    overall_status: str
    status_color: str
    recent_records: List[ExposureRecordResponse]


class ComplianceAuditItem(BaseModel):
    record_id: int
    timestamp: datetime
    worker_id: str
    worker_name: str
    department: str
    cumulative_dose_ppm_h: float
    twa_8h_ppm: float
    oisd_status: str
    is_within_limit: bool


class ComplianceReportResponse(BaseModel):
    report_title: str
    standard_reference: str
    generated_at: datetime
    total_shifts_logged: int
    shifts_within_limits: int
    shifts_warning_level: int
    shifts_exceeded_ceiling: int
    compliance_rate_percent: float
    twa_permissible_limit_ppm: float = 10.0
    records: List[ComplianceAuditItem]


class GenerateBadgeRequest(BaseModel):
    exposure: float = Field(default=12.0, ge=0.0, le=100.0, description="Dose in ppm*h")
    temp: float = Field(default=28.0, ge=10.0, le=50.0, description="Temperature in °C")
    humidity: float = Field(default=55.0, ge=10.0, le=95.0, description="Relative Humidity in %")
    lighting: str = Field(default="daylight", description="daylight | warm | cool | shadow")
    tilt: float = Field(default=0.0, description="Perspective tilt in degrees")
    expired: bool = Field(default=False, description="Simulate degraded/expired badge")
    badge_id: Optional[str] = Field(default="H2S-B17-00482", description="Custom badge serial ID")
    batch_id: Optional[str] = Field(default="B17", description="Manufacturing batch ID")
    manufactured_date: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    expiry_date: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    calibration_version: Optional[str] = Field(default="CAL-03", description="Calibration code")


class DetectedZone(BaseModel):
    id: str = Field(description="Unique zone identifier: sensing_zone, ref_white, ref_light_gray, ref_med_gray, ref_dark, qr_badge")
    name: str = Field(description="Human readable name")
    short_label: Optional[str] = Field(default=None, description="Short label for HUD display")
    category: str = Field(description="chemical_sensing | calibration_reference | badge_identity_qr")
    color: str = Field(description="Hex color code for bounding box/HUD tag")
    icon: Optional[str] = Field(default=None, description="Emoji icon")
    confidence: float = Field(description="Detection confidence score (0.0 to 1.0)")
    polygon: List[List[float]] = Field(description="4 normalized polygon points [[x0,y0], [x1,y1], [x2,y2], [x3,y3]] in [0.0, 1.0]")
    bbox: Dict[str, float] = Field(description="Normalized bounding box {x, y, width, height}")
    center: Optional[List[float]] = Field(default=None, description="Normalized center [x, y]")
    mean_bgr: Optional[List[float]] = Field(default=None, description="Sampled mean BGR color")


class CameraGuideRequest(BaseModel):
    image_base64: str = Field(description="Base64 encoded JPEG/PNG frame from live camera stream")


class CameraGuideResponse(BaseModel):
    status: str = Field(description="SEARCHING | MOVE_CLOSER | MOVE_BACK | CENTER_BADGE | TOO_DARK | REDUCE_GLARE | HOLD_STEADY | PERFECT_ALIGNMENT")
    message: str = Field(description="Actionable user coaching text")
    color: str = Field(description="Hex color code for HUD banner")
    can_autocapture: bool = Field(description="True if alignment and stability criteria are satisfied")
    badge_detected: bool = Field(description="True if a 4-point badge boundary is identified")
    corners: Optional[List[List[float]]] = Field(default=None, description="Normalized 4-corner coordinates [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] in [0.0, 1.0]")
    coverage_pct: float = Field(default=0.0, description="Badge frame area percentage")
    center_offset_pct: float = Field(default=0.0, description="Normalized offset from frame center")
    blur_score: float = Field(default=0.0, description="Laplacian blur metric")
    brightness: float = Field(default=0.0, description="Mean frame luminance (0-255)")
    rotation_deg: float = Field(default=0.0, description="Badge tilt/rotation in degrees")
    zones: List[DetectedZone] = Field(default_factory=list, description="Dynamically detected badge zones with bounding polygons")


class DetectZonesResponse(BaseModel):
    badge_detected: bool
    confidence: float
    corners: Optional[List[List[float]]] = None
    rotation_deg: float = 0.0
    zones: List[DetectedZone] = Field(default_factory=list)

