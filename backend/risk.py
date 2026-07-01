"""
risk.py
───────
AgriSense-AI — Risk engine router.

Computes four agronomic risk scores (0–100) plus a composite:

  Soil Risk    – pH deviation from crop optimum
  Disease Risk – humidity × temperature stress index
  Water Risk   – rainfall deficit (drought) or surplus (waterlogging)
  Weather Risk – combined temperature + rainfall deviation

Endpoints
  POST /risk              → full risk breakdown
  GET  /risk/crops        → list crops with their optimal parameters
"""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

# ── Crop optimal parameters ───────────────────────────────────────────────────
# (ph_optimal, temp_optimal °C, rainfall_optimal mm/yr)
CROP_OPTIMA: dict[str, dict] = {
    "apple":       {"ph": 6.3, "temp": 15, "rainfall":  900},
    "banana":      {"ph": 6.0, "temp": 27, "rainfall": 1200},
    "blackgram":   {"ph": 6.5, "temp": 28, "rainfall":  700},
    "chickpea":    {"ph": 6.5, "temp": 20, "rainfall":  450},
    "coconut":     {"ph": 6.0, "temp": 27, "rainfall": 1500},
    "coffee":      {"ph": 6.0, "temp": 23, "rainfall": 1800},
    "cotton":      {"ph": 7.0, "temp": 28, "rainfall":  700},
    "grapes":      {"ph": 6.5, "temp": 22, "rainfall":  700},
    "jute":        {"ph": 7.0, "temp": 28, "rainfall": 1500},
    "kidneybeans": {"ph": 6.5, "temp": 20, "rainfall":  800},
    "lentil":      {"ph": 6.8, "temp": 18, "rainfall":  400},
    "maize":       {"ph": 6.5, "temp": 22, "rainfall":  600},
    "mango":       {"ph": 6.5, "temp": 27, "rainfall": 1000},
    "mothbeans":   {"ph": 7.0, "temp": 30, "rainfall":  450},
    "mungbean":    {"ph": 6.5, "temp": 28, "rainfall":  600},
    "muskmelon":   {"ph": 6.8, "temp": 30, "rainfall":  500},
    "orange":      {"ph": 6.0, "temp": 22, "rainfall":  900},
    "papaya":      {"ph": 6.5, "temp": 28, "rainfall": 1200},
    "pigeonpeas":  {"ph": 6.5, "temp": 27, "rainfall":  650},
    "pomegranate": {"ph": 7.0, "temp": 25, "rainfall":  550},
    "rice":        {"ph": 6.0, "temp": 25, "rainfall": 1500},
    "watermelon":  {"ph": 6.5, "temp": 30, "rainfall":  600},
}

_DEFAULT_OPTIMA = {"ph": 6.5, "temp": 25, "rainfall": 800}

# Composite weights
_WEIGHTS = {
    "soil":    0.30,
    "disease": 0.25,
    "water":   0.25,
    "weather": 0.20,
}

# Disease thresholds
_DISEASE_HUM_THRESHOLD  = 60.0   # % — risk starts above this
_DISEASE_TEMP_THRESHOLD = 25.0   # °C — risk starts above this
_DISEASE_HUM_RANGE      = 40.0   # full-risk span above threshold
_DISEASE_TEMP_RANGE     = 15.0   # full-risk span above threshold


# ── Risk formulae ─────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def soil_risk(ph: float, ph_optimal: float) -> float:
    """Penalise pH deviation from crop optimum (full risk at ±3 units)."""
    return round(_clamp(abs(ph - ph_optimal) / 3.0 * 100), 2)


def disease_risk(humidity: float, temperature: float) -> float:
    """
    High humidity + high temperature drive fungal/bacterial risk.
    Risk is zero below both thresholds; compounds above them.
    """
    h_norm = _clamp(
        max(humidity    - _DISEASE_HUM_THRESHOLD,  0) / _DISEASE_HUM_RANGE  * 100
    )
    t_norm = _clamp(
        max(temperature - _DISEASE_TEMP_THRESHOLD, 0) / _DISEASE_TEMP_RANGE * 100
    )
    return round(0.6 * h_norm + 0.4 * t_norm, 2)


def water_risk(rainfall: float, rainfall_optimal: float) -> float:
    """
    Drought (deficit) penalised at full rate;
    waterlogging (surplus) penalised at half rate.
    """
    if rainfall_optimal <= 0:
        return 0.0
    if rainfall < rainfall_optimal:
        score = (rainfall_optimal - rainfall) / rainfall_optimal * 100
    else:
        score = (rainfall - rainfall_optimal) / rainfall_optimal * 50
    return round(_clamp(score), 2)


def weather_risk(
    temperature: float,
    rainfall:    float,
    temp_optimal: float,
    rainfall_optimal: float,
) -> float:
    """Combined deviation of temperature and rainfall from crop optima."""
    t_risk = _clamp(
        abs(temperature - temp_optimal) / max(temp_optimal, 1) * 100
    )
    r_risk = _clamp(
        abs(rainfall - rainfall_optimal) / max(rainfall_optimal, 1) * 100
    )
    return round(0.5 * t_risk + 0.5 * r_risk, 2)


def composite_risk(soil: float, disease: float, water: float, weather: float) -> float:
    return round(
        _weights_sum(soil=soil, disease=disease, water=water, weather=weather), 2
    )


def _weights_sum(**scores: float) -> float:
    return sum(_WEIGHTS[k] * v for k, v in scores.items())


def _risk_level(score: float) -> str:
    if score < 25:  return "Low"
    if score < 50:  return "Moderate"
    if score < 75:  return "High"
    return "Critical"


# ── Schemas ───────────────────────────────────────────────────────────────────

class RiskRequest(BaseModel):
    crop:        str   = Field(..., description="Crop name (from recommender)")
    ph:          float = Field(..., ge=0,   le=14,    description="Soil pH")
    temperature: float = Field(..., ge=-10, le=60,    description="Temperature (°C)")
    humidity:    float = Field(..., ge=0,   le=100,   description="Relative humidity (%)")
    rainfall:    float = Field(..., ge=0,   le=5000,  description="Annual rainfall (mm)")

    @field_validator("crop", mode="before")
    @classmethod
    def _normalise(cls, v: str) -> str:
        return v.strip().lower()


class RiskScore(BaseModel):
    score:   float = Field(..., description="Risk score 0–100")
    level:   str   = Field(..., description="Low / Moderate / High / Critical")


class RiskResponse(BaseModel):
    crop:     str
    ph:       float
    temperature: float
    humidity: float
    rainfall: float

    # Optima used
    ph_optimal:       float
    temp_optimal:     float
    rainfall_optimal: float
    crop_known:       bool

    # Individual risks
    soil:    RiskScore
    disease: RiskScore
    water:   RiskScore
    weather: RiskScore

    # Composite
    overall: RiskScore

    # Weights
    weights: dict


# ── Core calculation ──────────────────────────────────────────────────────────

def calculate_risk(req: RiskRequest) -> RiskResponse:
    optima     = CROP_OPTIMA.get(req.crop, _DEFAULT_OPTIMA)
    crop_known = req.crop in CROP_OPTIMA

    ph_opt   = optima["ph"]
    temp_opt = optima["temp"]
    rain_opt = optima["rainfall"]

    s_score = soil_risk(req.ph, ph_opt)
    d_score = disease_risk(req.humidity, req.temperature)
    w_score = water_risk(req.rainfall, rain_opt)
    x_score = weather_risk(req.temperature, req.rainfall, temp_opt, rain_opt)
    o_score = composite_risk(s_score, d_score, w_score, x_score)

    def rs(score: float) -> RiskScore:
        return RiskScore(score=score, level=_risk_level(score))

    return RiskResponse(
        crop        = req.crop,
        ph          = req.ph,
        temperature = req.temperature,
        humidity    = req.humidity,
        rainfall    = req.rainfall,
        ph_optimal       = ph_opt,
        temp_optimal     = temp_opt,
        rainfall_optimal = rain_opt,
        crop_known       = crop_known,
        soil    = rs(s_score),
        disease = rs(d_score),
        water   = rs(w_score),
        weather = rs(x_score),
        overall = rs(o_score),
        weights = _WEIGHTS,
    )


# ── APIRouter ─────────────────────────────────────────────────────────────────
router = APIRouter()


@router.get("/risk/crops", tags=["meta"])
def list_crops():
    """List all crops and their optimal parameters."""
    return [
        {"crop": crop, **params}
        for crop, params in sorted(CROP_OPTIMA.items())
    ]


@router.post("/risk", response_model=RiskResponse, tags=["risk"])
def assess_risk(req: RiskRequest):
    """
    Compute four agronomic risk scores and a weighted composite.

    Falls back to default optima (pH 6.5, 25°C, 800 mm) for unknown crops.
    """
    return calculate_risk(req)
