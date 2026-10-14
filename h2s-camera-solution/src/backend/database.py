"""
Database layer for H2S Dosimeter System.
Manages persistent records for workers, chemical badge scans, shift exposure history,
and compliance audits using SQLAlchemy (SQLite default, PostgreSQL compatible).
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Generator
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, text
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from src.utils.config import DATA_DIR, ensure_directories

ensure_directories()

# Default SQLite database path
DB_PATH = DATA_DIR / "dosimeter_records.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Worker(Base):
    """Worker entity for occupational health tracking."""
    __tablename__ = "workers"

    id = Column(String(50), primary_key=True, index=True)  # e.g., "WRK-1042"
    name = Column(String(100), nullable=False)
    department = Column(String(100), nullable=False, default="Sulfur Recovery Unit")
    designation = Column(String(100), nullable=False, default="Process Operator")
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    records = relationship("ExposureRecord", back_populates="worker", cascade="all, delete-orphan")


class ExposureRecord(Base):
    """Immutable record of an H2S dosimeter badge scan and cumulative dose estimation."""
    __tablename__ = "exposure_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    worker_id = Column(String(50), ForeignKey("workers.id"), nullable=False, index=True)
    badge_id = Column(String(100), default="H2S-B17-00482", index=True)
    batch_id = Column(String(50), default="B17", index=True)
    calibration_version = Column(String(50), default="CAL-03")
    image_filename = Column(String(255), nullable=True)
    
    # ML & Dosimetry Results
    cumulative_dose_ppm_h = Column(Float, nullable=False)
    confidence_interval_low = Column(Float, nullable=False)
    confidence_interval_high = Column(Float, nullable=False)
    uncertainty_margin = Column(Float, nullable=False)
    twa_8h_ppm = Column(Float, nullable=False)
    risk_level = Column(String(50), nullable=False)  # "SAFE", "WARNING", "CRITICAL", "EXPIRED"
    action_recommendation = Column(Text, nullable=True)
    
    # Environmental Context
    temperature_c = Column(Float, nullable=False)
    humidity_percent = Column(Float, nullable=False)
    exposure_duration_minutes = Column(Float, nullable=False)
    
    # Preprocessing & Badge Verification Diagnostics
    is_badge_valid = Column(Boolean, default=True)
    expiry_status = Column(String(50), default="VALID")
    badge_status = Column(String(50), default="VALID")
    reliability_score = Column(Float, nullable=True, default=0.95)
    lighting_profile = Column(String(50), default="BALANCED")
    model_type = Column(String(50), default="XGBOOST")
    model_version = Column(String(50), default="v1.0-xgboost")
    
    # Hardware Reader & AS7343 / BME688 Fields
    measurement_source = Column(String(50), default="CAMERA")  # "CAMERA" or "SPECTRAL_HARDWARE_AS7343"
    spectral_data = Column(Text, nullable=True)  # JSON string of raw counts and ratios
    gas_resistance_ohms = Column(Float, nullable=True)
    pressure_hpa = Column(Float, nullable=True)
    is_saturated = Column(Boolean, default=False)
    led_current_ma = Column(Integer, nullable=True)
    
    worker = relationship("Worker", back_populates="records")


def init_db():
    """Create tables and seed initial demo workers if database is empty."""
    Base.metadata.create_all(bind=engine)
    
    # Run lightweight schema migrations for existing SQLite databases
    if "sqlite" in DATABASE_URL:
        try:
            with engine.connect() as conn:
                result = conn.execute(text("PRAGMA table_info(exposure_records);"))
                existing_cols = {row[1] for row in result.fetchall()}
                if existing_cols:
                    if "batch_id" not in existing_cols:
                        conn.execute(text("ALTER TABLE exposure_records ADD COLUMN batch_id VARCHAR(50) DEFAULT 'B17';"))
                    if "calibration_version" not in existing_cols:
                        conn.execute(text("ALTER TABLE exposure_records ADD COLUMN calibration_version VARCHAR(50) DEFAULT 'CAL-03';"))
                    if "badge_status" not in existing_cols:
                        conn.execute(text("ALTER TABLE exposure_records ADD COLUMN badge_status VARCHAR(50) DEFAULT 'VALID';"))
                    if "reliability_score" not in existing_cols:
                        conn.execute(text("ALTER TABLE exposure_records ADD COLUMN reliability_score FLOAT DEFAULT 0.95;"))
                    if "model_version" not in existing_cols:
                        conn.execute(text("ALTER TABLE exposure_records ADD COLUMN model_version VARCHAR(50) DEFAULT 'v1.0-xgboost';"))
                    if "measurement_source" not in existing_cols:
                        conn.execute(text("ALTER TABLE exposure_records ADD COLUMN measurement_source VARCHAR(50) DEFAULT 'CAMERA';"))
                    if "spectral_data" not in existing_cols:
                        conn.execute(text("ALTER TABLE exposure_records ADD COLUMN spectral_data TEXT;"))
                    if "gas_resistance_ohms" not in existing_cols:
                        conn.execute(text("ALTER TABLE exposure_records ADD COLUMN gas_resistance_ohms FLOAT;"))
                    if "pressure_hpa" not in existing_cols:
                        conn.execute(text("ALTER TABLE exposure_records ADD COLUMN pressure_hpa FLOAT;"))
                    if "is_saturated" not in existing_cols:
                        conn.execute(text("ALTER TABLE exposure_records ADD COLUMN is_saturated BOOLEAN DEFAULT 0;"))
                    if "led_current_ma" not in existing_cols:
                        conn.execute(text("ALTER TABLE exposure_records ADD COLUMN led_current_ma INTEGER;"))
                    conn.commit()
        except Exception:
            pass

    db: Session = SessionLocal()
    try:
        if db.query(Worker).count() == 0:
            seed_workers = [
                Worker(id="WRK-1001", name="Rajesh Kumar", department="Desulfurization Plant 1", designation="Senior Field Technician"),
                Worker(id="WRK-1002", name="Amit Sharma", department="Crude Distillation Unit", designation="Process Engineer"),
                Worker(id="WRK-1003", name="Sunita Patel", department="Sulfur Recovery Unit 2", designation="Safety Officer"),
                Worker(id="WRK-1004", name="Mohammed Ali", department="Pipeline Transfer Station", designation="Maintenance Specialist")
            ]
            db.add_all(seed_workers)
            db.commit()
    finally:
        db.close()


def get_db() -> Generator[Session, None, None]:
    """Dependency for FastAPI route database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Auto-initialize and run schema migrations on module import
init_db()

