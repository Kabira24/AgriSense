"""
predict.py
──────────
FastAPI crop recommendation service for AgriSense-AI.

Routes
──────
  Meta
    GET  /health               Liveness probe
    GET  /model/info           Pipeline metadata (type, steps, features, classes)

  Crops catalogue
    GET  /crops                List all 22 supported crops with feature optima
    GET  /crops/{crop}         Detail card for a single crop

  Features
    GET  /features             Describe the 4 input features (range, unit, hint)

  Prediction
    POST /predict              Top-N crops for one soil/weather sample
    POST /predict/batch        Top-N crops for multiple samples in one call
    POST /predict/compare      Side-by-side confidence comparison for chosen crops

  Frontend (static)
    GET  /                     Serves frontend/index.html
    GET  /static/**            Static file mount

Usage
  python backend:/predict.py
  uvicorn backend:.predict:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Path as FPath, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).parent    # backend:/
_WORKSPACE = _HERE.parent             # AgriSense-AI:/


def _resolve(base: Path, *candidates: str) -> Path:
    for name in candidates:
        p = base / name
        if p.exists():
            return p
    raise FileNotFoundError(f"None of {candidates} found under {base}")


MODELS_DIR   = _resolve(_WORKSPACE, "models:", "models")
MODEL_PATH   = MODELS_DIR / "crop_recommender.joblib"
CLASSES_PATH = MODELS_DIR / "crop_recommender_classes.json"
FRONTEND_DIR = _resolve(_WORKSPACE, "frontend:", "frontend")

# ── Load model once at startup ────────────────────────────────────────────────
_pipeline = joblib.load(MODEL_PATH)
with open(CLASSES_PATH) as _f:
    _meta = json.load(_f)

_CLASSES:  dict[str, str] = _meta["classes"]    # "0" → "apple", …
_FEATURES: list[str]      = _meta["features"]   # ["ph", "temperature", …]
_CROP_LIST: list[str]     = [_CLASSES[str(i)] for i in range(len(_CLASSES))]

# ── Static crop catalogue ─────────────────────────────────────────────────────
# ph_opt, temp_opt(°C), humidity_opt(%), rainfall_opt(mm/yr), season, zone
_CROP_CATALOGUE: dict[str, dict] = {
    "apple":       {"ph_opt": 6.3, "temp_opt": 15,  "humidity_opt": 65,  "rainfall_opt": 900,  "season": "Rabi",   "zone": "Temperate"},
    "banana":      {"ph_opt": 6.0, "temp_opt": 27,  "humidity_opt": 80,  "rainfall_opt": 1200, "season": "Annual", "zone": "Tropical"},
    "blackgram":   {"ph_opt": 6.5, "temp_opt": 28,  "humidity_opt": 70,  "rainfall_opt": 700,  "season": "Kharif", "zone": "Semi-Arid"},
    "chickpea":    {"ph_opt": 6.5, "temp_opt": 20,  "humidity_opt": 55,  "rainfall_opt": 450,  "season": "Rabi",   "zone": "Semi-Arid"},
    "coconut":     {"ph_opt": 6.0, "temp_opt": 27,  "humidity_opt": 85,  "rainfall_opt": 1500, "season": "Annual", "zone": "Tropical"},
    "coffee":      {"ph_opt": 6.0, "temp_opt": 23,  "humidity_opt": 80,  "rainfall_opt": 1800, "season": "Annual", "zone": "Tropical"},
    "cotton":      {"ph_opt": 7.0, "temp_opt": 28,  "humidity_opt": 60,  "rainfall_opt": 700,  "season": "Kharif", "zone": "Semi-Arid"},
    "grapes":      {"ph_opt": 6.5, "temp_opt": 22,  "humidity_opt": 60,  "rainfall_opt": 700,  "season": "Rabi",   "zone": "Temperate"},
    "jute":        {"ph_opt": 7.0, "temp_opt": 28,  "humidity_opt": 85,  "rainfall_opt": 1500, "season": "Kharif", "zone": "Tropical"},
    "kidneybeans": {"ph_opt": 6.5, "temp_opt": 20,  "humidity_opt": 65,  "rainfall_opt": 800,  "season": "Kharif", "zone": "Temperate"},
    "lentil":      {"ph_opt": 6.8, "temp_opt": 18,  "humidity_opt": 55,  "rainfall_opt": 400,  "season": "Rabi",   "zone": "Semi-Arid"},
    "maize":       {"ph_opt": 6.5, "temp_opt": 22,  "humidity_opt": 65,  "rainfall_opt": 600,  "season": "Kharif", "zone": "Temperate"},
    "mango":       {"ph_opt": 6.5, "temp_opt": 27,  "humidity_opt": 70,  "rainfall_opt": 1000, "season": "Annual", "zone": "Tropical"},
    "mothbeans":   {"ph_opt": 7.0, "temp_opt": 30,  "humidity_opt": 50,  "rainfall_opt": 450,  "season": "Kharif", "zone": "Arid"},
    "mungbean":    {"ph_opt": 6.5, "temp_opt": 28,  "humidity_opt": 70,  "rainfall_opt": 600,  "season": "Kharif", "zone": "Semi-Arid"},
    "muskmelon":   {"ph_opt": 6.8, "temp_opt": 30,  "humidity_opt": 55,  "rainfall_opt": 500,  "season": "Zaid",   "zone": "Arid"},
    "orange":      {"ph_opt": 6.0, "temp_opt": 22,  "humidity_opt": 65,  "rainfall_opt": 900,  "season": "Annual", "zone": "Temperate"},
    "papaya":      {"ph_opt": 6.5, "temp_opt": 28,  "humidity_opt": 80,  "rainfall_opt": 1200, "season": "Annual", "zone": "Tropical"},
    "pigeonpeas":  {"ph_opt": 6.5, "temp_opt": 27,  "humidity_opt": 65,  "rainfall_opt": 650,  "season": "Kharif", "zone": "Semi-Arid"},
    "pomegranate": {"ph_opt": 7.0, "temp_opt": 25,  "humidity_opt": 55,  "rainfall_opt": 550,  "season": "Annual", "zone": "Arid"},
    "rice":        {"ph_opt": 6.0, "temp_opt": 25,  "humidity_opt": 82,  "rainfall_opt": 1500, "season": "Kharif", "zone": "Tropical"},
    "watermelon":  {"ph_opt": 6.5, "temp_opt": 30,  "humidity_opt": 60,  "rainfall_opt": 600,  "season": "Zaid",   "zone": "Semi-Arid"},
}

# ── Feature descriptors ───────────────────────────────────────────────────────
_FEATURE_META: list[dict] = [
    {
        "name": "ph", "label": "Soil pH", "unit": "pH",
        "min": 0.0, "max": 14.0, "typical_min": 4.5, "typical_max": 8.5,
        "hint": "Measure with a soil pH kit or send a sample to your nearest KVK.",
    },
    {
        "name": "temperature", "label": "Temperature", "unit": "°C",
        "min": -10.0, "max": 60.0, "typical_min": 10.0, "typical_max": 45.0,
        "hint": "Mean annual air temperature for your location.",
    },
    {
        "name": "humidity", "label": "Relative Humidity", "unit": "%",
        "min": 0.0, "max": 100.0, "typical_min": 20.0, "typical_max": 95.0,
        "hint": "Average relative humidity; use weather station or local IMD data.",
    },
    {
        "name": "rainfall", "label": "Annual Rainfall", "unit": "mm",
        "min": 0.0, "max": 5000.0, "typical_min": 200.0, "typical_max": 3000.0,
        "hint": "Total annual rainfall for your district (district rainfall atlas or IMD).",
    },
]

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "AgriSense-AI Crop Recommendation Service",
    description = (
        "ML-powered crop recommendation engine.\n\n"
        "**Model**: RandomForest classifier inside a sklearn Pipeline (StandardScaler → RF).\n"
        "**Crops**: 22 · **Features**: 4 (ph, temperature, humidity, rainfall)\n\n"
        "Use `POST /predict` for a single recommendation or `POST /predict/batch` "
        "for multiple samples in one call."
    ),
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ── Shared schemas ─────────────────────────────────────────────────────────────

class SoilWeatherInput(BaseModel):
    """Four soil/weather parameters fed into the crop recommender."""
    ph:          float = Field(..., ge=0.0,   le=14.0,   description="Soil pH (0–14)")
    temperature: float = Field(..., ge=-10.0, le=60.0,   description="Temperature (°C)")
    humidity:    float = Field(..., ge=0.0,   le=100.0,  description="Relative humidity (%)")
    rainfall:    float = Field(..., ge=0.0,   le=5000.0, description="Annual rainfall (mm/yr)")

    @field_validator("ph", "temperature", "humidity", "rainfall", mode="before")
    @classmethod
    def _coerce_numeric(cls, v):
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ValueError(f"Expected a numeric value, got {v!r}")


class CropPrediction(BaseModel):
    rank:       int
    crop:       str
    confidence: float = Field(..., description="Confidence score 0–100 (%)")


# ── Internal helper ───────────────────────────────────────────────────────────

def _infer(features: list[list[float]], top_n: int) -> list[list[CropPrediction]]:
    """
    Run predict_proba on a batch of feature vectors.
    Returns a list (one per sample) of ranked CropPrediction lists.
    """
    try:
        probas: np.ndarray = _pipeline.predict_proba(features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}")

    results = []
    for sample_probas in probas:
        top = sorted(enumerate(sample_probas), key=lambda x: -x[1])[:top_n]
        results.append([
            CropPrediction(
                rank=rank + 1,
                crop=_CLASSES[str(idx)],
                confidence=round(float(prob) * 100, 2),
            )
            for rank, (idx, prob) in enumerate(top)
        ])
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# META ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/health",
    tags=["meta"],
    summary="Liveness probe",
)
def health():
    """Returns service status, model file name, and number of crop classes."""
    return {
        "status":  "ok",
        "model":   MODEL_PATH.name,
        "version": app.version,
        "classes": len(_CLASSES),
        "features": len(_FEATURES),
    }


@app.get(
    "/model/info",
    tags=["meta"],
    summary="Pipeline metadata",
)
def model_info():
    """
    Returns full introspection of the loaded sklearn Pipeline:
    step names, estimator types, feature list, and all crop class labels.
    """
    steps = []
    if hasattr(_pipeline, "steps"):
        for name, estimator in _pipeline.steps:
            steps.append({
                "name": name,
                "type": type(estimator).__name__,
                "params": {
                    k: v for k, v in estimator.get_params().items()
                    if not hasattr(v, "__iter__") or isinstance(v, (str, bool, int, float))
                },
            })

    return {
        "model_file":  MODEL_PATH.name,
        "pipeline_type": type(_pipeline).__name__,
        "steps":       steps,
        "features":    _FEATURES,
        "n_features":  len(_FEATURES),
        "classes":     list(_CLASSES.values()),
        "n_classes":   len(_CLASSES),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CROPS CATALOGUE ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

class CropDetail(BaseModel):
    crop:         str
    ph_opt:       float = Field(..., description="Optimal soil pH")
    temp_opt:     float = Field(..., description="Optimal temperature (°C)")
    humidity_opt: float = Field(..., description="Optimal humidity (%)")
    rainfall_opt: float = Field(..., description="Optimal annual rainfall (mm)")
    season:       str   = Field(..., description="Kharif / Rabi / Zaid / Annual")
    zone:         str   = Field(..., description="Agro-climatic zone")


@app.get(
    "/crops",
    response_model=List[CropDetail],
    tags=["crops"],
    summary="List all supported crops",
)
def list_crops(
    zone:   Optional[str] = Query(None, description="Filter by zone (e.g. Tropical)"),
    season: Optional[str] = Query(None, description="Filter by season (e.g. Kharif)"),
):
    """
    Returns all 22 crops the model can recommend, enriched with their
    agro-climatic optima, growing season, and climatic zone.
    Optionally filter by `zone` or `season`.
    """
    crops = [
        CropDetail(crop=name, **info)
        for name, info in sorted(_CROP_CATALOGUE.items())
    ]
    if zone:
        crops = [c for c in crops if c.zone.lower() == zone.lower()]
    if season:
        crops = [c for c in crops if c.season.lower() == season.lower()]
    return crops


@app.get(
    "/crops/{crop}",
    response_model=CropDetail,
    tags=["crops"],
    summary="Get detail for a single crop",
)
def get_crop(
    crop: str = FPath(..., description="Crop name (e.g. rice, maize)"),
):
    """
    Returns the agro-climatic profile for a specific crop:
    optimal pH, temperature, humidity, rainfall, growing season, and zone.
    """
    key = crop.strip().lower()
    if key not in _CROP_CATALOGUE:
        raise HTTPException(
            status_code=404,
            detail=f"Crop '{crop}' not found. Supported: {sorted(_CROP_CATALOGUE)}",
        )
    return CropDetail(crop=key, **_CROP_CATALOGUE[key])


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURES ROUTE
# ═══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/features",
    tags=["meta"],
    summary="Describe all input features",
)
def list_features():
    """
    Returns metadata for each of the 4 model input features:
    name, label, unit, valid range, typical agronomic range, and a user hint.
    """
    return {"features": _FEATURE_META, "count": len(_FEATURE_META)}


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

class PredictRequest(SoilWeatherInput):
    top_n: int = Field(
        default=3, ge=1, le=22,
        description="Number of top crop recommendations to return (1–22)",
    )


class PredictResponse(BaseModel):
    top_crops:  List[CropPrediction]
    top_n:      int
    input_echo: dict


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["prediction"],
    summary="Recommend top-N crops for one sample",
)
def predict(req: PredictRequest):
    """
    Returns the top-N crop recommendations (default 3, max 22) for a single
    soil/weather sample, ranked by model confidence.

    All four inputs are required. Use `top_n` to control how many crops to return.
    """
    rows   = [[req.ph, req.temperature, req.humidity, req.rainfall]]
    preds  = _infer(rows, req.top_n)

    return PredictResponse(
        top_crops  = preds[0],
        top_n      = req.top_n,
        input_echo = req.model_dump(exclude={"top_n"}),
    )


# ── Batch prediction ───────────────────────────────────────────────────────────

class BatchItem(SoilWeatherInput):
    id: Optional[str] = Field(None, description="Optional caller-supplied identifier")


class BatchRequest(BaseModel):
    samples: List[BatchItem] = Field(
        ..., min_length=1, max_length=100,
        description="List of 1–100 soil/weather samples",
    )
    top_n: int = Field(
        default=3, ge=1, le=22,
        description="Number of top crops to return per sample",
    )


class BatchResultItem(BaseModel):
    index:     int
    id:        Optional[str]
    top_crops: List[CropPrediction]
    input:     dict


class BatchResponse(BaseModel):
    results: List[BatchResultItem]
    top_n:   int
    count:   int


@app.post(
    "/predict/batch",
    response_model=BatchResponse,
    tags=["prediction"],
    summary="Recommend crops for multiple samples in one call",
)
def predict_batch(req: BatchRequest):
    """
    Accepts up to **100 soil/weather samples** in a single request and returns
    top-N crop recommendations for each.

    Useful for bulk field-zone analysis or comparing multiple farm plots.
    """
    rows  = [[s.ph, s.temperature, s.humidity, s.rainfall] for s in req.samples]
    preds = _infer(rows, req.top_n)

    results = [
        BatchResultItem(
            index     = i,
            id        = req.samples[i].id,
            top_crops = preds[i],
            input     = req.samples[i].model_dump(exclude={"id"}),
        )
        for i in range(len(req.samples))
    ]

    return BatchResponse(results=results, top_n=req.top_n, count=len(results))


# ── Compare prediction ─────────────────────────────────────────────────────────

class CompareRequest(SoilWeatherInput):
    crops: List[str] = Field(
        ..., min_length=2, max_length=22,
        description="Crop names to compare (at least 2)",
    )


class CropCompareEntry(BaseModel):
    crop:       str
    confidence: float = Field(..., description="Model confidence 0–100 (%)")
    rank:       int   = Field(..., description="Rank among ALL 22 crops (1 = best)")
    viable:     bool  = Field(..., description="True if confidence ≥ 5 %")


class CompareResponse(BaseModel):
    comparison:   List[CropCompareEntry]
    best_overall: str
    input_echo:   dict


@app.post(
    "/predict/compare",
    response_model=CompareResponse,
    tags=["prediction"],
    summary="Compare confidence scores for specific crops",
)
def predict_compare(req: CompareRequest):
    """
    Given a set of soil/weather conditions, returns the model confidence for
    each of the **specified crops** so you can directly compare them side by side.

    Also reports the global rank of each crop across all 22 classes and flags
    whether the crop is `viable` (confidence ≥ 5%).

    Useful for answering: *"Between wheat and maize, which suits my field better?"*
    """
    # Normalise requested crop names
    requested = [c.strip().lower() for c in req.crops]
    unknown   = [c for c in requested if c not in _CROP_CATALOGUE]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown crops: {unknown}. Supported: {sorted(_CROP_CATALOGUE)}",
        )

    # Full probability distribution across all 22 classes
    rows   = [[req.ph, req.temperature, req.humidity, req.rainfall]]
    probas = _pipeline.predict_proba(rows)[0]

    # Build global rank lookup (1 = highest prob)
    ranked = sorted(range(len(probas)), key=lambda i: -probas[i])
    global_rank = {_CLASSES[str(i)]: pos + 1 for pos, i in enumerate(ranked)}

    # Build crop → class-index lookup
    crop_to_idx = {v: int(k) for k, v in _CLASSES.items()}

    comparison = sorted(
        [
            CropCompareEntry(
                crop       = crop,
                confidence = round(float(probas[crop_to_idx[crop]]) * 100, 2),
                rank       = global_rank[crop],
                viable     = probas[crop_to_idx[crop]] >= 0.05,
            )
            for crop in requested
        ],
        key=lambda e: -e.confidence,
    )

    return CompareResponse(
        comparison   = comparison,
        best_overall = comparison[0].crop,
        input_echo   = req.model_dump(exclude={"crops"}),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FRONTEND STATIC FILES
# ═══════════════════════════════════════════════════════════════════════════════

if FRONTEND_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR), html=True),
        name="static",
    )

    @app.get("/", include_in_schema=False)
    def serve_index():
        index = FRONTEND_DIR / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return {"message": "Frontend not found – place index.html in frontend:/"}


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "predict:app",
        host      = "0.0.0.0",
        port      = 8000,
        reload    = True,
        log_level = "info",
    )
