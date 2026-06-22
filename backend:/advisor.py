"""
advisor.py
──────────
AgriSense-AI Farm Advisor — Gemini-powered advice endpoint.

Reads the system prompt from ``advisor_system_prompt.md`` (same directory),
aggregates the four AgriSense service outputs, and calls the Gemini API to
produce plain-language farmer advice.

Endpoint
  POST /advise  →  { advice: str }

Environment
  GEMINI_API_KEY   – required; your Google AI Studio key

Usage
  GEMINI_API_KEY=<key> python backend:/advisor.py
  # or
  GEMINI_API_KEY=<key> uvicorn backend:.advisor:app --reload --port 8004

Port: 8004
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Optional

# ── Load .env from the workspace root before reading env vars ─────────────────
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass   # python-dotenv not installed; env vars must be set in the shell

from google import genai
from google.genai import types as genai_types
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent   # backend:/

_SYSTEM_PROMPT_PATH = _HERE / "advisor_system_prompt.md"

# ── Load system prompt ─────────────────────────────────────────────────────────
def _load_system_prompt() -> str:
    if not _SYSTEM_PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"System prompt not found at {_SYSTEM_PROMPT_PATH}. "
            "Ensure advisor_system_prompt.md is in the backend directory."
        )
    return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


SYSTEM_PROMPT: str = _load_system_prompt()

# ── Gemini client setup ────────────────────────────────────────────────────────
_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not _API_KEY:
    import warnings
    warnings.warn(
        "GEMINI_API_KEY is not set. POST /advise will fail until it is provided.",
        stacklevel=1,
    )

_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AgriSense-AI Farm Advisor",
    description=(
        "Calls the Gemini API with all four AgriSense pipeline outputs "
        "(crop recommendation, profit estimate, weather, risk) and returns "
        "plain-language advice for the farmer."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ─────────────────────────────────────────────────

class TopCrop(BaseModel):
    rank:       int
    crop:       str
    confidence: float = Field(..., description="Confidence score 0–100 (%)")


class CropRecommendation(BaseModel):
    top_crops: list[TopCrop]


class ProfitEstimate(BaseModel):
    crop:                str
    production_quintals: Optional[float]  = None
    market_price:        Optional[float]  = None   # Rs./quintal
    revenue:             Optional[float]  = None   # Rs.
    cost:                Optional[float]  = None   # Rs.
    profit:              Optional[float]  = None   # Rs.
    profit_margin_pct:   Optional[float]  = None
    break_even_price:    Optional[float]  = None   # Rs./quintal
    price_available:     bool             = False


class WeatherSummary(BaseModel):
    location:      Optional[str]   = None
    forecast_days: Optional[int]   = None
    temperature:   Optional[float] = Field(None, description="Mean temp (°C)")
    humidity:      Optional[float] = Field(None, description="Mean humidity (%)")
    rainfall_forecast_mm: Optional[float] = Field(None, description="Total rain (mm)")
    summary:       Optional[str]   = Field(None, description="Human-readable summary (optional)")


class RiskAssessment(BaseModel):
    crop:            str
    soil_risk:       Optional[float] = None
    disease_risk:    Optional[float] = None
    water_risk:      Optional[float] = None
    weather_risk:    Optional[float] = None
    composite_risk:  float
    risk_level:      str   = Field(..., description="Low / Moderate / High / Critical")


class AdviseRequest(BaseModel):
    crop_recommendation: CropRecommendation
    profit_estimate:     ProfitEstimate
    weather:             WeatherSummary
    risk_assessment:     RiskAssessment
    language:            Optional[str] = Field(
        None,
        description="Optional language override (e.g. 'Hindi', 'Marathi'). "
                    "Defaults to English.",
    )


class AdviseResponse(BaseModel):
    advice:    str
    model:     str
    cached:    bool = False


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_user_message(req: AdviseRequest) -> str:
    """Serialise the four inputs into a clean JSON block for Gemini."""
    payload = req.model_dump(exclude={"language"})
    msg = textwrap.dedent(f"""\
        Here is the AgriSense data for this farmer. Please generate advice now.

        ```json
        {json.dumps(payload, indent=2, ensure_ascii=False)}
        ```
    """)
    if req.language:
        msg += f"\nPlease respond in {req.language}."
    return msg


def _call_gemini(system_prompt: str, user_message: str) -> str:
    """Send a system-prompted message to Gemini and return the text."""
    if not _API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured on the server. "
                   "Set the environment variable and restart.",
        )

    client = genai.Client(api_key=_API_KEY)

    try:
        response = client.models.generate_content(
            model=_MODEL_NAME,
            contents=user_message,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.4,
                max_output_tokens=512,
            ),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API error: {exc}",
        )

    # Safely extract text
    try:
        text = response.text
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Gemini returned an empty or blocked response.",
        )

    return text.strip()


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
def health():
    return {
        "status":  "ok",
        "model":   _MODEL_NAME,
        "api_key_set": bool(_API_KEY),
        "system_prompt_loaded": bool(SYSTEM_PROMPT),
        "system_prompt_chars":  len(SYSTEM_PROMPT),
    }


@app.post("/advise", response_model=AdviseResponse, tags=["advisor"])
def advise(req: AdviseRequest):
    """
    Aggregate crop recommendation, profit, weather, and risk data,
    then call Gemini to produce simple, farmer-friendly advice.

    All four input blocks mirror the response schemas of the other
    AgriSense services so this endpoint can be called directly after
    chaining /predict → /profit → /weather/summary → /risk.
    """
    user_msg = _build_user_message(req)
    advice   = _call_gemini(SYSTEM_PROMPT, user_msg)

    return AdviseResponse(
        advice=advice,
        model=_MODEL_NAME,
    )


@app.get("/advise/prompt", tags=["meta"])
def get_system_prompt():
    """Return the active system prompt (useful for debugging/inspection)."""
    return {"system_prompt": SYSTEM_PROMPT, "chars": len(SYSTEM_PROMPT)}


# ── Example endpoint (no API key needed) ──────────────────────────────────────

@app.get("/advise/example-input", tags=["meta"])
def example_input():
    """Return a fully-formed example request body for POST /advise."""
    return {
        "crop_recommendation": {
            "top_crops": [
                {"rank": 1, "crop": "wheat",    "confidence": 91.5},
                {"rank": 2, "crop": "maize",    "confidence": 6.2},
                {"rank": 3, "crop": "chickpea", "confidence": 2.3},
            ]
        },
        "profit_estimate": {
            "crop":                "wheat",
            "production_quintals": 50,
            "market_price":        2150,
            "revenue":             107500,
            "cost":                60000,
            "profit":              47500,
            "profit_margin_pct":   44.2,
            "break_even_price":    1200,
            "price_available":     True,
        },
        "weather": {
            "location":            "Ludhiana, Punjab",
            "forecast_days":       7,
            "temperature":         22.4,
            "humidity":            65.0,
            "rainfall_forecast_mm": 0.0,
            "summary": "Clear and dry for the next 7 days. No rainfall expected.",
        },
        "risk_assessment": {
            "crop":           "wheat",
            "soil_risk":      12.0,
            "disease_risk":   18.5,
            "water_risk":     22.0,
            "weather_risk":   15.0,
            "composite_risk": 17.0,
            "risk_level":     "Low",
        },
    }


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "advisor:app",
        host="0.0.0.0",
        port=8004,
        reload=True,
        log_level="info",
    )
