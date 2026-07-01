"""
app.py
──────
AgriSense-AI — Unified FastAPI application.

Merges all 7 AgriSense services into a single FastAPI app served by a
single Uvicorn process on a single port.

Start with:
    uvicorn app:app --host 0.0.0.0 --port 8000

In production (Render):
    cd backend && uvicorn app:app --host 0.0.0.0 --port $PORT

Services mounted
  predict   — ML crop recommendation (routes: /predict, /crops, /features, /model/info)
  profit    — Profit estimation        (routes: /profit, /profit/crops, /states, /commodities)
  weather   — Open-Meteo forecast      (routes: /weather, /weather/summary, /weather/geocode)
  risk      — Risk engine              (routes: /risk, /risk/crops)
  advisor   — Gemini AI advisor        (routes: /advise, /advise/prompt)
  planner   — Crop operations planner  (routes: /planner/schedule, /planner/crops)
  sensor    — IoT sensor feed          (routes: /latest-sensor, /sensor/status)

  meta      — Unified health probe     (route: /health)
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Load .env FIRST, before any service module that reads env vars ─────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass  # python-dotenv is in requirements.txt; this should never happen

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Import all service routers ─────────────────────────────────────────────────
from predict import router as predict_router
from profit  import router as profit_router
from weather import router as weather_router
from risk    import router as risk_router
from advisor import router as advisor_router
from planner import router as planner_router
from sensor  import router as sensor_router

# ── Application ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AgriSense-AI",
    description=(
        "Unified AgriSense backend — crop recommendation, profit estimation, "
        "weather integration, risk analysis, AI advisor, operations planner, "
        "and IoT sensor feed — all on a single port."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ── allow all origins (frontend on Vercel, local dev, etc.) ───────────
_CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
_origins = (
    [o.strip() for o in _CORS_ORIGINS.split(",") if o.strip()]
    if _CORS_ORIGINS != "*"
    else ["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register service routers ───────────────────────────────────────────────────
app.include_router(predict_router)
app.include_router(profit_router)
app.include_router(weather_router)
app.include_router(risk_router)
app.include_router(advisor_router)
app.include_router(planner_router)
app.include_router(sensor_router)


# ── Unified health probe ───────────────────────────────────────────────────────
@app.get("/health", tags=["meta"], summary="Unified liveness probe")
def health():
    """
    Returns 200 OK when the unified backend is running.

    All 7 services are co-located in this process, so a single health
    endpoint is sufficient.  The frontend's System Health tab will show all
    7 services as online when this endpoint returns 200.
    """
    return {
        "status":   "ok",
        "version":  "2.0.0",
        "services": [
            "predict", "profit", "weather",
            "risk", "advisor", "planner", "sensor",
        ],
    }


# ── Root redirect ─────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "AgriSense-AI backend is running. "
                   "Visit /docs for the interactive API reference.",
        "docs": "/docs",
    }
