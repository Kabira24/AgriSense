"""
sensor.py
─────────
AgriSense-AI — Live Sensor Feed Service.

Fetches the latest row from a Google Sheets CSV export every time
GET /latest-sensor is called. Falls back to static mock values if the
sheet is unreachable or returns malformed data.

Endpoints
  GET /latest-sensor   → Latest sensor reading (live or mock fallback)
  GET /sensor/status   → Integration status and last-fetch metadata
  GET /health          → Liveness probe

Data source
  Google Sheets CSV export (fetched live on every request):
  https://docs.google.com/spreadsheets/d/1aKxKKmDqiiSGvr3c5tYQjvzRQaLQTIxchRR3iGrINas/export?format=csv&gid=0

  Expected column headers (case-insensitive, whitespace-trimmed):
    Timestamp | Temperature | Humidity | Soil Moisture | pH

Port: 8006
"""

from __future__ import annotations

import csv
import io
import logging
import urllib.request
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

log = logging.getLogger("sensor")

# ── Configuration ──────────────────────────────────────────────────────────────

GOOGLE_SHEET_CSV_URL: str = (
    "https://docs.google.com/spreadsheets/d/"
    "1aKxKKmDqiiSGvr3c5tYQjvzRQaLQTIxchRR3iGrINas"
    "/export?format=csv&gid=0"
)

# Seconds to wait for Google Sheets HTTP response
_FETCH_TIMEOUT = 8

# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AgriSense-AI Live Sensor Feed Service",
    description=(
        "Streams live soil and climate readings from a Google Sheets CSV export "
        "to the AgriSense dashboard. Falls back to mock values if the sheet is "
        "temporarily unavailable."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Response schemas ───────────────────────────────────────────────────────────

class SensorReading(BaseModel):
    """A single sensor data point from the field sensor array."""
    temperature:   float  # °C  — ambient air temperature
    humidity:      float  # %   — relative humidity
    soil_moisture: float  # %   — volumetric soil moisture
    ph:            float  # pH  — soil pH
    timestamp:     str    # ISO-8601 string or "demo" for mock fallback


class SensorStatus(BaseModel):
    """Sensor integration status and configuration summary."""
    data_source:       str
    google_sheet_url:  Optional[str]
    integration_ready: bool
    last_fetch:        Optional[str]
    last_fetch_source: str   # "google_sheets" | "mock_fallback"
    note:              str


# ── Mock fallback values ───────────────────────────────────────────────────────

_MOCK = SensorReading(
    temperature=27.4,
    humidity=52.9,
    soil_moisture=100.0,
    ph=6.8,
    timestamp="demo",
)

# ── Runtime state (updated on every successful sheet fetch) ───────────────────
_last_fetch_time:   Optional[str] = None
_last_fetch_source: str           = "mock_fallback"


# ── Google Sheets live fetch ───────────────────────────────────────────────────

def _normalise_key(raw: str) -> str:
    """Lower-case and collapse whitespace so header matching is robust."""
    return " ".join(raw.strip().lower().split())


# Map from the sheet's normalised column header → SensorReading field name
_COLUMN_MAP: dict[str, str] = {
    "timestamp":    "timestamp",
    "temperature":  "temperature",
    "humidity":     "humidity",
    "soil moisture": "soil_moisture",
    "soil_moisture": "soil_moisture",
    "ph":           "ph",
}


def _fetch_from_google_sheet() -> SensorReading:
    """
    Fetch the CSV from Google Sheets and return the latest row as a
    SensorReading.  Raises RuntimeError on any failure so the caller
    can fall back to mock values gracefully.
    """
    try:
        req = urllib.request.Request(
            GOOGLE_SHEET_CSV_URL,
            headers={"User-Agent": "AgriSense-Sensor/2.0"},
        )
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT) as resp:
            raw_bytes = resp.read()
    except Exception as exc:
        raise RuntimeError(f"Network error fetching Google Sheet: {exc}") from exc

    try:
        content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        content = raw_bytes.decode("latin-1")

    reader = csv.DictReader(io.StringIO(content))

    # Normalise all header keys
    rows: list[dict[str, str]] = []
    for row in reader:
        rows.append({_normalise_key(k): v.strip() for k, v in row.items()})

    if not rows:
        raise RuntimeError("Google Sheet returned no data rows.")

    # Use the last (most recent) row
    latest = rows[-1]

    def _float(key: str, fallback: float) -> float:
        """Read a column value as float, using fallback if missing or invalid."""
        sheet_key = _COLUMN_MAP.get(key, key)
        # Try exact match first, then the canonical key
        val = latest.get(key) or latest.get(sheet_key, "")
        if not val:
            return fallback
        try:
            return float(val)
        except ValueError:
            log.warning("Could not parse column '%s' value '%s' as float.", key, val)
            return fallback

    # Read each column — the sheet headers are already normalised
    temperature   = _float("temperature",   _MOCK.temperature)
    humidity      = _float("humidity",      _MOCK.humidity)
    soil_moisture = _float("soil moisture", _MOCK.soil_moisture)
    ph            = _float("ph",            _MOCK.ph)

    # Timestamp: use the sheet value if present, else current UTC time
    ts_raw = latest.get("timestamp", "").strip()
    timestamp = ts_raw if ts_raw else datetime.now(timezone.utc).isoformat()

    return SensorReading(
        temperature=round(temperature, 1),
        humidity=round(humidity, 1),
        soil_moisture=round(soil_moisture, 1),
        ph=round(ph, 2),
        timestamp=timestamp,
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
def health():
    """Liveness probe for the sensor service."""
    return {
        "status":             "ok",
        "service":            "live-sensor-feed",
        "data_source":        "google_sheets",
        "google_sheet_url":   GOOGLE_SHEET_CSV_URL,
        "integration_active": True,
    }


@app.get("/latest-sensor", response_model=SensorReading, tags=["sensor"])
def get_latest_sensor() -> SensorReading:
    """
    Returns the most recent soil and climate sensor reading.

    Tries to fetch the latest row from Google Sheets on every call.
    If the sheet is unreachable or returns invalid data, returns the
    last known mock values so the dashboard never breaks.
    """
    global _last_fetch_time, _last_fetch_source

    try:
        reading = _fetch_from_google_sheet()
        _last_fetch_time   = datetime.now(timezone.utc).isoformat()
        _last_fetch_source = "google_sheets"
        log.info(
            "Sensor data from Google Sheets — temp=%.1f hum=%.1f moisture=%.1f ph=%.2f ts=%s",
            reading.temperature, reading.humidity,
            reading.soil_moisture, reading.ph, reading.timestamp,
        )
        return reading

    except Exception as exc:
        log.warning("Google Sheets fetch failed (%s). Returning mock fallback.", exc)
        _last_fetch_time   = datetime.now(timezone.utc).isoformat()
        _last_fetch_source = "mock_fallback"
        return _MOCK


@app.get("/sensor/status", response_model=SensorStatus, tags=["sensor"])
def sensor_status() -> SensorStatus:
    """
    Returns the current sensor integration status.
    Useful for displaying connection state on the System Health tab.
    """
    return SensorStatus(
        data_source="google_sheets",
        google_sheet_url=GOOGLE_SHEET_CSV_URL,
        integration_ready=True,
        last_fetch=_last_fetch_time,
        last_fetch_source=_last_fetch_source,
        note=(
            "Live integration active. Reads the latest row from the Google Sheets "
            "CSV export on every request. Falls back to mock values if the sheet "
            "is temporarily unavailable."
        ),
    )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
