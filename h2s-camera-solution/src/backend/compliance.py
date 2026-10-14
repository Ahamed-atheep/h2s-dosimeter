"""
DGMS & OISD Occupational Safety Compliance Reporting Engine.
Generates standardized compliance audits for Indian Oil & Gas / Mining regulations.
Standards Reference: OISD-STD-155 / DGMS Technical Circular (H2S Occupational Exposure).
"""

from datetime import datetime
from typing import List
from sqlalchemy.orm import Session
from src.backend.database import ExposureRecord, Worker
from src.backend.schemas import ComplianceReportResponse, ComplianceAuditItem


def generate_oisd_compliance_report(db: Session) -> ComplianceReportResponse:
    """
    Generate an occupational safety compliance report auditing all shift exposure records
    against OISD-STD-155 (Time-Weighted Average PEL = 10 ppm for 8-hour shift).
    """
    records: List[ExposureRecord] = (
        db.query(ExposureRecord)
        .order_by(ExposureRecord.timestamp.desc())
        .limit(200)
        .all()
    )

    total_shifts = len(records)
    shifts_within = 0
    shifts_warning = 0
    shifts_exceeded = 0
    audit_items: List[ComplianceAuditItem] = []

    for rec in records:
        worker = db.query(Worker).filter(Worker.id == rec.worker_id).first()
        worker_name = worker.name if worker else "Unknown"
        dept = worker.department if worker else "General Operations"

        twa = rec.twa_8h_ppm
        is_compliant = twa <= 10.0 and rec.is_badge_valid

        if not rec.is_badge_valid:
            status = "EXPIRED_BADGE_INVALID"
            shifts_warning += 1
        elif twa < 1.25:
            status = "NORMAL_PERMISSIBLE"
            shifts_within += 1
        elif twa <= 5.0:
            status = "ELEVATED_ACTION_LEVEL"
            shifts_warning += 1
        elif twa <= 10.0:
            status = "HIGH_MONITOR_REQUIRED"
            shifts_warning += 1
        else:
            status = "CRITICAL_PEL_EXCEEDED"
            shifts_exceeded += 1

        audit_items.append(
            ComplianceAuditItem(
                record_id=rec.id,
                timestamp=rec.timestamp,
                worker_id=rec.worker_id,
                worker_name=worker_name,
                department=dept,
                cumulative_dose_ppm_h=round(rec.cumulative_dose_ppm_h, 2),
                twa_8h_ppm=round(twa, 2),
                oisd_status=status,
                is_within_limit=is_compliant
            )
        )

    compliance_rate = (
        round((shifts_within / total_shifts) * 100.0, 1) if total_shifts > 0 else 100.0
    )

    return ComplianceReportResponse(
        report_title="H2S Occupational Health & Cumulative Exposure Audit Report",
        standard_reference="OISD-STD-155 / DGMS Technical Circular (8-hr TWA PEL <= 10 ppm)",
        generated_at=datetime.utcnow(),
        total_shifts_logged=total_shifts,
        shifts_within_limits=shifts_within,
        shifts_warning_level=shifts_warning,
        shifts_exceeded_ceiling=shifts_exceeded,
        compliance_rate_percent=compliance_rate,
        twa_permissible_limit_ppm=10.0,
        records=audit_items
    )
