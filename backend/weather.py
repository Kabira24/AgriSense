"""
weather.py
──────────
AgriSense-AI — Weather router.

Data source : Open-Meteo (https://open-meteo.com) — free, no API key.
Cache       : In-memory TTL cache, 1-hour expiry per (lat, lon) key.

Endpoints
  GET /weather?lat=&lon=&days=          → raw daily forecast (7–16 days)
  GET /weather/summary?lat=&lon=        → crop-ready averages (temp, rainfall)
  GET /weather/geocode?city=&country=   → city name → lat/lon
"""

from __future__ import annotations

import time
import threading
from datetime import datetime, timezone
from typing import Optional

import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# ── Constants ─────────────────────────────────────────────────────────────────
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODE_URL  = "https://geocoding-api.open-meteo.com/v1/search"

CACHE_TTL_SECONDS = 3600        # 1 hour
REQUEST_TIMEOUT   = 10          # seconds
TIMEZONE          = "Asia/Kolkata"

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
]

# ── In-memory TTL cache ───────────────────────────────────────────────────────

class TTLCache:
    """Thread-safe in-memory cache with per-entry TTL."""

    def __init__(self, ttl: int = CACHE_TTL_SECONDS) -> None:
        self._store: dict[str, tuple[float, object]] = {}
        self._ttl   = ttl
        self._lock  = threading.Lock()

    def get(self, key: str) -> Optional[object]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            ts, value = entry
            if time.monotonic() - ts > self._ttl:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: object) -> None:
        with self._lock:
            self._store[key] = (time.monotonic(), value)

    def invalidate(self, key: str) -> bool:
        with self._lock:
            return self._store.pop(key, None) is not None

    def stats(self) -> dict:
        with self._lock:
            now = time.monotonic()
            live = sum(1 for ts, _ in self._store.values()
                       if now - ts <= self._ttl)
            return {"total_entries": len(self._store), "live_entries": live}


_cache = TTLCache(ttl=CACHE_TTL_SECONDS)


def _cache_key(lat: float, lon: float, days: int) -> str:
    return f"{round(lat, 3)}:{round(lon, 3)}:{days}"


# ── Open-Meteo helpers ────────────────────────────────────────────────────────

def _fetch_forecast(lat: float, lon: float, days: int) -> dict:
    """Call Open-Meteo; raises HTTPException on failure."""
    key = _cache_key(lat, lon, days)
    cached = _cache.get(key)
    if cached is not None:
        cached["_cache_hit"] = True
        return cached

    params = {
        "latitude":     lat,
        "longitude":    lon,
        "daily":        ",".join(DAILY_VARIABLES),
        "forecast_days": days,
        "timezone":     TIMEZONE,
    }
    try:
        resp = requests.get(
            OPEN_METEO_FORECAST_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.Timeout:
        raise HTTPException(504, "Open-Meteo timed out.")
    except requests.RequestException as exc:
        raise HTTPException(502, f"Open-Meteo error: {exc}")

    data = resp.json()
    data["_cache_hit"] = False
    _cache.set(key, data)
    return data


def _fetch_geocode(city: str, country: Optional[str]) -> dict:
    """Resolve city name → lat/lon via Open-Meteo geocoding API."""
    key = f"geo:{city.lower()}:{(country or '').lower()}"
    cached = _cache.get(key)
    if cached is not None:
        return cached

    params = {"name": city, "count": 5, "language": "en", "format": "json"}
    if country:
        params["country"] = country

    try:
        resp = requests.get(
            OPEN_METEO_GEOCODE_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(502, f"Geocoding error: {exc}")

    data = resp.json()
    results = data.get("results", [])
    if not results:
        raise HTTPException(404, f"City '{city}' not found.")

    result = {
        "city":      results[0].get("name"),
        "state":     results[0].get("admin1"),
        "country":   results[0].get("country"),
        "latitude":  results[0].get("latitude"),
        "longitude": results[0].get("longitude"),
        "timezone":  results[0].get("timezone"),
    }
    _cache.set(key, result)
    return result


def _safe_mean(values: list) -> Optional[float]:
    nums = [v for v in values if v is not None]
    return round(sum(nums) / len(nums), 2) if nums else None


def _safe_sum(values: list) -> Optional[float]:
    nums = [v for v in values if v is not None]
    return round(sum(nums), 2) if nums else None


# ── Schemas ───────────────────────────────────────────────────────────────────

class DailyForecast(BaseModel):
    date:              str
    temp_max_c:        Optional[float]
    temp_min_c:        Optional[float]
    temp_mean_c:       Optional[float]
    precipitation_mm:  Optional[float]
    humidity_max_pct:  Optional[float]
    humidity_min_pct:  Optional[float]
    humidity_mean_pct: Optional[float]


class WeatherResponse(BaseModel):
    latitude:    float
    longitude:   float
    timezone:    str
    forecast_days: int
    cache_hit:   bool
    fetched_at:  str
    daily:       list[DailyForecast]


class CropWeatherSummary(BaseModel):
    latitude:    float
    longitude:   float
    timezone:    str
    forecast_days: int
    cache_hit:   bool
    fetched_at:  str

    # Crop-model-ready values
    temperature:   Optional[float] = Field(None, description="Mean daily avg temp (°C)")
    temp_max:      Optional[float] = Field(None, description="Mean daily max temp (°C)")
    temp_min:      Optional[float] = Field(None, description="Mean daily min temp (°C)")
    humidity:      Optional[float] = Field(None, description="Mean daily humidity (%)")
    rainfall:      Optional[float] = Field(None, description="Annualised rainfall estimate (mm/year)")
    rainfall_forecast_mm: Optional[float] = Field(None, description="Total rain over forecast window (mm)")

    note: str = (
        "temperature, humidity, and rainfall are ready to pass to POST /predict."
    )


class GeocodeResponse(BaseModel):
    city:      Optional[str]
    state:     Optional[str]
    country:   Optional[str]
    latitude:  float
    longitude: float
    timezone:  Optional[str]


# ── APIRouter ─────────────────────────────────────────────────────────────────
router = APIRouter()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/weather/geocode", response_model=GeocodeResponse, tags=["geo"])
def geocode(
    city:    str           = Query(..., description="City name, e.g. 'Pune'"),
    country: Optional[str] = Query(None, description="Country code, e.g. 'IN'"),
):
    """Resolve a city name to latitude/longitude."""
    return _fetch_geocode(city, country)


@router.get("/weather", response_model=WeatherResponse, tags=["weather"])
def get_forecast(
    lat:  float = Query(..., ge=-90,  le=90,  description="Latitude"),
    lon:  float = Query(..., ge=-180, le=180, description="Longitude"),
    days: int   = Query(7,  ge=1,    le=16,  description="Forecast days (1–16)"),
):
    """
    Daily temperature and rainfall forecast from Open-Meteo.
    Cached for 1 hour per (lat, lon, days).
    """
    raw  = _fetch_forecast(lat, lon, days)
    daily_raw = raw.get("daily", {})

    dates    = daily_raw.get("time", [])
    t_max    = daily_raw.get("temperature_2m_max", [])
    t_min    = daily_raw.get("temperature_2m_min", [])
    precip   = daily_raw.get("precipitation_sum", [])
    hum_max  = daily_raw.get("relative_humidity_2m_max", [])
    hum_min  = daily_raw.get("relative_humidity_2m_min", [])

    daily = []
    for i, date in enumerate(dates):
        mx  = t_max[i]  if i < len(t_max)   else None
        mn  = t_min[i]  if i < len(t_min)   else None
        hx  = hum_max[i] if i < len(hum_max) else None
        hm  = hum_min[i] if i < len(hum_min) else None
        daily.append(DailyForecast(
            date             = date,
            temp_max_c       = mx,
            temp_min_c       = mn,
            temp_mean_c      = round((mx + mn) / 2, 2) if mx is not None and mn is not None else None,
            precipitation_mm = precip[i] if i < len(precip) else None,
            humidity_max_pct = hx,
            humidity_min_pct = hm,
            humidity_mean_pct= round((hx + hm) / 2, 2) if hx is not None and hm is not None else None,
        ))

    return WeatherResponse(
        latitude     = raw.get("latitude", lat),
        longitude    = raw.get("longitude", lon),
        timezone     = raw.get("timezone", TIMEZONE),
        forecast_days= days,
        cache_hit    = raw.get("_cache_hit", False),
        fetched_at   = datetime.now(timezone.utc).isoformat(),
        daily        = daily,
    )


@router.get("/weather/summary", response_model=CropWeatherSummary, tags=["weather"])
def get_crop_summary(
    lat:  float = Query(..., ge=-90,  le=90,  description="Latitude"),
    lon:  float = Query(..., ge=-180, le=180, description="Longitude"),
    days: int   = Query(7,  ge=1,    le=16,  description="Forecast days (1–16)"),
):
    """
    Crop-model-ready weather summary.

    Returns **temperature**, **humidity**, and **rainfall** values
    formatted to pass directly into the crop recommender's POST /predict.

    - `temperature` = mean of daily (max+min)/2 over forecast window
    - `humidity`    = mean of daily (hum_max+hum_min)/2
    - `rainfall`    = annualised from forecast window total
                      (total_mm / days × 365)
    """
    raw       = _fetch_forecast(lat, lon, days)
    daily_raw = raw.get("daily", {})

    t_max   = daily_raw.get("temperature_2m_max", [])
    t_min   = daily_raw.get("temperature_2m_min", [])
    precip  = daily_raw.get("precipitation_sum", [])
    hum_max = daily_raw.get("relative_humidity_2m_max", [])
    hum_min = daily_raw.get("relative_humidity_2m_min", [])

    # Daily means
    temp_means = [
        (mx + mn) / 2
        for mx, mn in zip(t_max, t_min)
        if mx is not None and mn is not None
    ]
    hum_means = [
        (hx + hm) / 2
        for hx, hm in zip(hum_max, hum_min)
        if hx is not None and hm is not None
    ]

    total_rain_mm = _safe_sum(precip) or 0.0
    annualised    = round(total_rain_mm / days * 365, 2) if days > 0 else None

    return CropWeatherSummary(
        latitude             = raw.get("latitude", lat),
        longitude            = raw.get("longitude", lon),
        timezone             = raw.get("timezone", TIMEZONE),
        forecast_days        = days,
        cache_hit            = raw.get("_cache_hit", False),
        fetched_at           = datetime.now(timezone.utc).isoformat(),
        temperature          = _safe_mean(temp_means),
        temp_max             = _safe_mean(t_max),
        temp_min             = _safe_mean(t_min),
        humidity             = _safe_mean(hum_means),
        rainfall             = annualised,
        rainfall_forecast_mm = total_rain_mm,
    )
