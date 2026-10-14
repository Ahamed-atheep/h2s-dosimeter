"""
QR Code Badge Identity and Expiry Validator.
Parses, validates, and decodes QR payloads printed on disposable H2S dosimeter badges.
Ensures digital badge identity, manufacturing batch tracking, declared shelf-life validation,
and calibration version compatibility.
"""

import json
import re
from datetime import datetime, date
from typing import Dict, Any, Optional, Union, Tuple
import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("QR_Validator")

# Known supported calibration versions
SUPPORTED_CALIBRATION_VERSIONS = {"CAL-01", "CAL-02", "CAL-03", "CAL-04"}

# Regex pattern for Badge ID (e.g. H2S-B17-00482, BADGE-STD-2026, etc.)
BADGE_ID_PATTERN = re.compile(r"^[A-Za-z0-9\-_]+$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_date_string(date_str: str) -> Optional[date]:
    """Parse YYYY-MM-DD date string safely."""
    if not isinstance(date_str, str) or not DATE_PATTERN.match(date_str.strip()):
        return None
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def validate_badge_qr_payload(
    payload_input: Union[str, Dict[str, Any]],
    reference_date: Optional[Union[date, str]] = None
) -> Dict[str, Any]:
    """
    Validate a QR code payload from a physical badge.

    Expected payload structure:
    {
        "badge_id": "H2S-B17-00482",
        "batch_id": "B17",
        "manufactured_date": "2026-09-01",
        "expiry_date": "2026-12-01",
        "calibration_version": "CAL-03"
    }

    Parameters
    ----------
    payload_input : str or dict
        Raw QR JSON string or pre-parsed dictionary.
    reference_date : date or str, optional
        Current date for expiry evaluation (defaults to today's UTC date).

    Returns
    -------
    dict
        Structured validation outcome:
        - is_valid: bool
        - badge_status: "VALID" | "EXPIRED" | "INVALID_FORMAT"
        - badge_id: str or None
        - batch_id: str or None
        - manufactured_date: str or None
        - expiry_date: str or None
        - calibration_version: str or None
        - message: str
    """
    # Determine reference date
    if reference_date is None:
        current_dt = datetime.utcnow().date()
    elif isinstance(reference_date, str):
        parsed_ref = parse_date_string(reference_date)
        current_dt = parsed_ref if parsed_ref else datetime.utcnow().date()
    elif isinstance(reference_date, (datetime, date)):
        current_dt = reference_date if isinstance(reference_date, date) else reference_date.date()
    else:
        current_dt = datetime.utcnow().date()

    # 1. Parse JSON if string
    if isinstance(payload_input, str):
        payload_str = payload_input.strip()
        if not payload_str:
            return {
                "is_valid": False,
                "badge_status": "INVALID_FORMAT",
                "badge_id": None,
                "batch_id": None,
                "manufactured_date": None,
                "expiry_date": None,
                "calibration_version": None,
                "message": "QR payload is empty."
            }
        try:
            data = json.loads(payload_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Malformed QR JSON payload: {e}")
            return {
                "is_valid": False,
                "badge_status": "INVALID_FORMAT",
                "badge_id": None,
                "batch_id": None,
                "manufactured_date": None,
                "expiry_date": None,
                "calibration_version": None,
                "message": f"Malformed QR code data. Invalid JSON format: {str(e)}"
            }
    elif isinstance(payload_input, dict):
        data = payload_input
    else:
        return {
            "is_valid": False,
            "badge_status": "INVALID_FORMAT",
            "badge_id": None,
            "batch_id": None,
            "manufactured_date": None,
            "expiry_date": None,
            "calibration_version": None,
            "message": "Invalid QR payload type. Expected JSON string or object."
        }

    # 2. Check required fields
    required_fields = ["badge_id", "batch_id", "expiry_date"]
    missing_fields = [f for f in required_fields if f not in data or not str(data[f]).strip()]
    if missing_fields:
        return {
            "is_valid": False,
            "badge_status": "INVALID_FORMAT",
            "badge_id": str(data.get("badge_id", "")).strip() or None,
            "batch_id": str(data.get("batch_id", "")).strip() or None,
            "manufactured_date": str(data.get("manufactured_date", "")).strip() or None,
            "expiry_date": str(data.get("expiry_date", "")).strip() or None,
            "calibration_version": str(data.get("calibration_version", "")).strip() or None,
            "message": f"QR payload missing required field(s): {', '.join(missing_fields)}"
        }

    badge_id = str(data["badge_id"]).strip()
    batch_id = str(data["batch_id"]).strip()
    expiry_date_str = str(data["expiry_date"]).strip()
    mfg_date_str = str(data.get("manufactured_date", "")).strip() or None
    cal_version = str(data.get("calibration_version", "CAL-03")).strip()

    # 3. Validate Badge ID format
    if not BADGE_ID_PATTERN.match(badge_id):
        return {
            "is_valid": False,
            "badge_status": "INVALID_FORMAT",
            "badge_id": badge_id,
            "batch_id": batch_id,
            "manufactured_date": mfg_date_str,
            "expiry_date": expiry_date_str,
            "calibration_version": cal_version,
            "message": f"Invalid badge_id format: '{badge_id}'. Expected alphanumeric identifier."
        }

    # 4. Validate Expiry Date format
    exp_date = parse_date_string(expiry_date_str)
    if not exp_date:
        return {
            "is_valid": False,
            "badge_status": "INVALID_FORMAT",
            "badge_id": badge_id,
            "batch_id": batch_id,
            "manufactured_date": mfg_date_str,
            "expiry_date": expiry_date_str,
            "calibration_version": cal_version,
            "message": f"Invalid expiry date format: '{expiry_date_str}'. Expected YYYY-MM-DD."
        }

    # 5. Validate Manufacturing Date if present
    if mfg_date_str:
        mfg_date = parse_date_string(mfg_date_str)
        if not mfg_date:
            return {
                "is_valid": False,
                "badge_status": "INVALID_FORMAT",
                "badge_id": badge_id,
                "batch_id": batch_id,
                "manufactured_date": mfg_date_str,
                "expiry_date": expiry_date_str,
                "calibration_version": cal_version,
                "message": f"Invalid manufactured date format: '{mfg_date_str}'. Expected YYYY-MM-DD."
            }
        if mfg_date > exp_date:
            return {
                "is_valid": False,
                "badge_status": "INVALID_FORMAT",
                "badge_id": badge_id,
                "batch_id": batch_id,
                "manufactured_date": mfg_date_str,
                "expiry_date": expiry_date_str,
                "calibration_version": cal_version,
                "message": f"Manufacturing date ({mfg_date_str}) cannot be after expiry date ({expiry_date_str})."
            }

    # 6. Validate Calibration Version
    if cal_version and cal_version not in SUPPORTED_CALIBRATION_VERSIONS:
        logger.info(f"Calibration version '{cal_version}' not in standard list, proceeding with default compatibility.")

    # 7. Expiry check: current_date <= expiry_date -> VALID, else EXPIRED
    if current_dt > exp_date:
        return {
            "is_valid": False,
            "badge_status": "EXPIRED",
            "badge_id": badge_id,
            "batch_id": batch_id,
            "manufactured_date": mfg_date_str,
            "expiry_date": expiry_date_str,
            "calibration_version": cal_version,
            "message": f"Badge expired on {expiry_date_str}. Please use a new badge."
        }

    # Valid Badge
    return {
        "is_valid": True,
        "badge_status": "VALID",
        "badge_id": badge_id,
        "batch_id": batch_id,
        "manufactured_date": mfg_date_str,
        "expiry_date": expiry_date_str,
        "calibration_version": cal_version,
        "message": f"Badge identity verified: {badge_id} (Batch {batch_id}, valid until {expiry_date_str})."
    }


def process_badge_qr(qr_payload: str, reference_date: Optional[Union[date, str]] = None) -> Dict[str, Any]:
    """Alias for validate_badge_qr_payload."""
    return validate_badge_qr_payload(qr_payload, reference_date=reference_date)


def decode_qr_from_image(image_bgr: np.ndarray) -> Optional[str]:
    """
    Locate and decode a QR code from a photographic image using OpenCV.
    Returns decoded text string or None if no QR code is detected.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None

    try:
        detector = cv2.QRCodeDetector()
        decoded_text, points, _ = detector.detectAndDecode(image_bgr)
        if decoded_text and decoded_text.strip():
            return decoded_text.strip()
    except Exception as e:
        logger.warning(f"Error decoding QR from image: {e}")

    return None
