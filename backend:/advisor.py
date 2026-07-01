"""
advisor.py
──────────
AgriSense-AI Farm Advisor — Gemini-powered advice with full offline fallback.

POST /advise  →  { advice: str, model: str, cached: bool }

Behaviour
---------
1. If a valid GEMINI_API_KEY is configured, advice is generated via Gemini.
2. If the key is absent, invalid, quota-exceeded, or the network is down,
   the service falls back to a rich, rule-based local advisor that uses all
   four AgriSense pipeline outputs to produce detailed agronomic advice.
3. HTTP 503 is NEVER returned.  The endpoint always responds with 200 + advice.

Environment (optional)
  GEMINI_API_KEY   – Google AI Studio key (demo works without it)
  GEMINI_MODEL     – model name override (default: gemini-2.0-flash)

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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Optional Gemini import — never crash if the library is unavailable ─────────
try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent   # backend:/
_SYSTEM_PROMPT_PATH = _HERE / "advisor_system_prompt.md"

# ── Load system prompt (used by Gemini path; not required for fallback) ────────
def _load_system_prompt() -> str:
    if _SYSTEM_PROMPT_PATH.exists():
        return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    return ""   # fallback: empty string is fine for Gemini too

SYSTEM_PROMPT: str = _load_system_prompt()

# ── Gemini client setup ────────────────────────────────────────────────────────
_API_KEY    = os.getenv("GEMINI_API_KEY", "").strip()
_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

if not _API_KEY:
    import warnings
    warnings.warn(
        "GEMINI_API_KEY is not set. POST /advise will use the built-in "
        "local advisor (full agronomic advice, no internet required).",
        stacklevel=1,
    )

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AgriSense-AI Farm Advisor",
    description=(
        "Produces farmer-friendly agronomic advice from crop recommendation, "
        "profit estimate, weather, and risk data.  Gemini API is used when "
        "available; a rich local engine provides full advice otherwise."
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


# ── Request / Response schemas ─────────────────────────────────────────────────

class TopCrop(BaseModel):
    rank:       int
    crop:       str
    confidence: float = Field(..., description="Confidence score 0–100 (%)")


class CropRecommendation(BaseModel):
    top_crops: list[TopCrop]


class ProfitEstimate(BaseModel):
    crop:                str
    production_quintals: Optional[float] = None
    market_price:        Optional[float] = None   # Rs./quintal
    revenue:             Optional[float] = None   # Rs.
    cost:                Optional[float] = None   # Rs.
    profit:              Optional[float] = None   # Rs.
    profit_margin_pct:   Optional[float] = None
    break_even_price:    Optional[float] = None   # Rs./quintal
    price_available:     bool            = False


class WeatherSummary(BaseModel):
    location:             Optional[str]   = None
    forecast_days:        Optional[int]   = None
    temperature:          Optional[float] = Field(None, description="Mean temp (°C)")
    humidity:             Optional[float] = Field(None, description="Mean humidity (%)")
    rainfall_forecast_mm: Optional[float] = Field(None, description="Total rain (mm)")
    summary:              Optional[str]   = Field(None, description="Human-readable summary")


class RiskAssessment(BaseModel):
    crop:           str
    soil_risk:      Optional[float] = None
    disease_risk:   Optional[float] = None
    water_risk:     Optional[float] = None
    weather_risk:   Optional[float] = None
    composite_risk: float
    risk_level:     str = Field(..., description="Low / Moderate / High / Critical")


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
    advice:  str
    model:   str
    cached:  bool = False


# ── Gemini path ────────────────────────────────────────────────────────────────

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
    """Send a system-prompted message to Gemini and return the text.

    Raises RuntimeError on any failure so the caller can fall back gracefully.
    """
    if not _GENAI_AVAILABLE:
        raise RuntimeError("google-genai library not installed.")
    if not _API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set.")

    client = genai.Client(api_key=_API_KEY)
    response = client.models.generate_content(
        model=_MODEL_NAME,
        contents=user_message,
        config=genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.4,
            max_output_tokens=600,
        ),
    )
    text = response.text   # raises AttributeError if blocked/empty
    if not text or not text.strip():
        raise RuntimeError("Gemini returned an empty response.")
    return text.strip()


# ── Local agronomic knowledge base ────────────────────────────────────────────

# Crop-specific agronomic profiles used by the local fallback engine
_CROP_PROFILES: dict[str, dict] = {
    "wheat": {
        "optimal_temp": (15, 25), "optimal_humidity": (50, 70),
        "water_need": "moderate", "season": "Rabi (Oct–Mar)",
        "disease_watch": "rust (brown/yellow) and powdery mildew",
        "sow_tip": "Best sown Oct–Nov. Ensure seed treatment before sowing.",
        "harvest_tip": "Harvest when grain moisture drops below 14%.",
        "soil_tip": "Prefers well-drained loamy soil with pH 6.0–7.5.",
        "fertilizer": "Apply 120 kg N, 60 kg P, 40 kg K per hectare.",
    },
    "rice": {
        "optimal_temp": (20, 35), "optimal_humidity": (70, 90),
        "water_need": "high", "season": "Kharif (Jun–Nov)",
        "disease_watch": "blast, brown spot, and bacterial blight",
        "sow_tip": "Transplant seedlings at 20–25 days age. Maintain 5 cm standing water.",
        "harvest_tip": "Harvest when 80% of grains turn golden-yellow.",
        "soil_tip": "Clay or clay-loam soils retain water best for paddy.",
        "fertilizer": "Apply 100 kg N, 50 kg P, 50 kg K per hectare in splits.",
    },
    "maize": {
        "optimal_temp": (18, 32), "optimal_humidity": (50, 80),
        "water_need": "moderate", "season": "Kharif (Jun–Sep) or Rabi",
        "disease_watch": "fall armyworm, northern leaf blight",
        "sow_tip": "Sow at 5 cm depth, 60 × 20 cm spacing. Ensure good germination moisture.",
        "harvest_tip": "Harvest when husks dry and kernels are hard (milk-line gone).",
        "soil_tip": "Well-drained sandy loam to loam, pH 5.8–7.0.",
        "fertilizer": "Apply 150 kg N, 75 kg P, 40 kg K per hectare.",
    },
    "cotton": {
        "optimal_temp": (21, 35), "optimal_humidity": (50, 80),
        "water_need": "moderate-high", "season": "Kharif (Apr–Nov)",
        "disease_watch": "bollworm, whitefly, and leaf curl virus",
        "sow_tip": "Sow when soil temperature is above 18°C. Use Bt-cotton varieties.",
        "harvest_tip": "Pick bolls when fully open; avoid harvesting after heavy rains.",
        "soil_tip": "Deep black cotton soil or well-drained alluvial soil, pH 6.0–8.0.",
        "fertilizer": "Apply 120 kg N, 60 kg P, 60 kg K per hectare.",
    },
    "sugarcane": {
        "optimal_temp": (20, 35), "optimal_humidity": (60, 90),
        "water_need": "very high", "season": "Year-round (planted Oct–Mar)",
        "disease_watch": "red rot, smut, and ratoon stunting disease",
        "sow_tip": "Use healthy setts with at least 3 buds. Plant in furrows 30 cm deep.",
        "harvest_tip": "Harvest at 10–12 months for optimal sucrose content.",
        "soil_tip": "Deep, well-drained loamy soil. Avoid waterlogging.",
        "fertilizer": "Apply 250 kg N, 100 kg P, 100 kg K per hectare.",
    },
    "chickpea": {
        "optimal_temp": (15, 25), "optimal_humidity": (40, 65),
        "water_need": "low", "season": "Rabi (Oct–Mar)",
        "disease_watch": "ascochyta blight and fusarium wilt",
        "sow_tip": "Sow in well-prepared seedbed at 30 cm row spacing. Inoculate with Rhizobium.",
        "harvest_tip": "Harvest when 90% of pods turn brown and seeds rattle.",
        "soil_tip": "Sandy loam to clay loam, pH 6.0–8.0. Avoid waterlogging.",
        "fertilizer": "Apply 20 kg N, 40 kg P per hectare (low N — fixes its own).",
    },
    "soybean": {
        "optimal_temp": (20, 30), "optimal_humidity": (60, 80),
        "water_need": "moderate", "season": "Kharif (Jun–Oct)",
        "disease_watch": "stem fly, semilooper, and yellow mosaic virus",
        "sow_tip": "Sow at 3–4 cm depth, 45 × 5 cm spacing after first good monsoon rains.",
        "harvest_tip": "Harvest when 95% of pods are mature and leaves have dropped.",
        "soil_tip": "Well-drained loamy soil, pH 6.0–7.5.",
        "fertilizer": "Apply 30 kg N, 60 kg P, 40 kg K per hectare.",
    },
    "mustard": {
        "optimal_temp": (10, 25), "optimal_humidity": (50, 70),
        "water_need": "low-moderate", "season": "Rabi (Oct–Feb)",
        "disease_watch": "Alternaria blight and white rust",
        "sow_tip": "Sow in rows 30–45 cm apart at 1–1.5 cm depth in well-prepared field.",
        "harvest_tip": "Harvest when 75% of pods turn golden-yellow.",
        "soil_tip": "Well-drained sandy loam, pH 6.0–7.5.",
        "fertilizer": "Apply 80 kg N, 40 kg P, 40 kg K per hectare.",
    },
    "groundnut": {
        "optimal_temp": (22, 35), "optimal_humidity": (50, 80),
        "water_need": "moderate", "season": "Kharif (Jun–Oct)",
        "disease_watch": "tikka leaf spot, stem rot, and aflatoxin contamination",
        "sow_tip": "Sow shelled seeds at 5 cm depth, 30 × 10 cm spacing.",
        "harvest_tip": "Harvest when inner pod wall shows dark discolouration.",
        "soil_tip": "Light sandy loam, well-drained, pH 6.0–7.0.",
        "fertilizer": "Apply 20 kg N, 60 kg P, 40 kg K per hectare.",
    },
}

_DEFAULT_PROFILE = {
    "optimal_temp": (18, 30), "optimal_humidity": (50, 75),
    "water_need": "moderate", "season": "Consult local agriculture office",
    "disease_watch": "common fungal and pest attacks",
    "sow_tip": "Follow recommended sowing calendar for your region.",
    "harvest_tip": "Harvest at physiological maturity to avoid losses.",
    "soil_tip": "Maintain soil pH 6.0–7.5 for most crops.",
    "fertilizer": "Follow soil test based fertilizer recommendations.",
}


def _get_profile(crop: str) -> dict:
    return _CROP_PROFILES.get(crop.lower().strip(), _DEFAULT_PROFILE)


def _risk_scores(req: AdviseRequest) -> dict[str, float]:
    ra = req.risk_assessment
    return {
        "Soil":    ra.soil_risk    or 0.0,
        "Disease": ra.disease_risk or 0.0,
        "Water":   ra.water_risk   or 0.0,
        "Weather": ra.weather_risk or 0.0,
    }


def _highest_risk_factor(scores: dict[str, float]) -> tuple[str, float]:
    return max(scores.items(), key=lambda kv: kv[1])


def _weather_advisory(req: AdviseRequest, profile: dict) -> str:
    w = req.weather
    temp     = w.temperature          or 25.0
    humidity = w.humidity             or 65.0
    rain_mm  = w.rainfall_forecast_mm or 0.0
    days     = w.forecast_days        or 7
    loc      = w.location             or "your region"
    summary  = w.summary              or ""

    t_lo, t_hi = profile["optimal_temp"]
    h_lo, h_hi = profile["optimal_humidity"]

    lines: list[str] = []

    if summary:
        lines.append(f"Weather forecast for {loc} ({days}-day): {summary}")
    else:
        lines.append(
            f"Weather forecast for {loc} ({days}-day): "
            f"Mean temp {temp:.1f}°C, humidity {humidity:.0f}%, "
            f"expected rainfall {rain_mm:.0f} mm."
        )

    # Temperature advice
    if temp < t_lo:
        lines.append(
            f"Temperatures ({temp:.1f}°C) are below the optimal range "
            f"({t_lo}–{t_hi}°C). Protect young seedlings from cold stress — "
            "consider light irrigation before frost nights."
        )
    elif temp > t_hi:
        lines.append(
            f"Temperatures ({temp:.1f}°C) are above the optimal range "
            f"({t_lo}–{t_hi}°C). Irrigate during early morning or evening "
            "to minimise heat stress on the crop."
        )
    else:
        lines.append(
            f"Temperatures are in the ideal range for this crop "
            f"({t_lo}–{t_hi}°C). Good conditions for active growth."
        )

    # Humidity / disease risk
    if humidity > h_hi:
        lines.append(
            f"High humidity ({humidity:.0f}%) increases the risk of "
            f"fungal infections. Scout fields regularly for "
            f"{profile['disease_watch']} and ensure good field drainage."
        )
    elif humidity < h_lo:
        lines.append(
            f"Low humidity ({humidity:.0f}%) may cause moisture stress. "
            "Consider supplemental irrigation at critical growth stages."
        )

    # Rainfall
    if rain_mm > 50:
        lines.append(
            f"Heavy rainfall expected ({rain_mm:.0f} mm). Delay spraying "
            "of fertilisers or pesticides. Ensure drainage channels are clear."
        )
    elif rain_mm == 0.0:
        lines.append(
            "No rain forecast. Plan irrigation schedule and conserve soil moisture."
        )

    return " ".join(lines)


def _profit_section(req: AdviseRequest) -> str:
    pe = req.profit_estimate
    crop = pe.crop.capitalize()

    if not pe.price_available or pe.profit is None:
        return (
            f"Profit data for {crop} is not available at this time. "
            "Check your local mandi (APMC) for current market prices before selling. "
            "Keep input costs (seeds, fertiliser, labour) recorded to calculate your break-even price."
        )

    lines: list[str] = []
    profit = pe.profit
    margin = pe.profit_margin_pct or 0.0
    bep    = pe.break_even_price  or 0.0
    rev    = pe.revenue           or 0.0
    cost   = pe.cost              or 0.0
    yld    = pe.production_quintals or 0.0
    mkt    = pe.market_price      or 0.0

    if profit > 0:
        lines.append(
            f"You could earn ₹{profit:,.0f} net profit on this crop "
            f"(revenue ₹{rev:,.0f} minus cost ₹{cost:,.0f}), "
            f"a margin of {margin:.1f}%."
        )
    else:
        lines.append(
            f"Current projections show a loss of ₹{abs(profit):,.0f}. "
            "Consider reducing input costs or shifting to an alternative crop."
        )

    if yld > 0 and mkt > 0:
        lines.append(
            f"Expected yield: {yld:.1f} quintals at ₹{mkt:,.0f}/quintal."
        )

    if bep > 0:
        if mkt > 0 and mkt > bep:
            lines.append(
                f"Your break-even price is ₹{bep:,.0f}/quintal — "
                f"current market (₹{mkt:,.0f}) is above this, so you are in a profitable zone. "
                "Do not sell below ₹{:.0f}/quintal to avoid losses.".format(bep)
            )
        else:
            lines.append(
                f"Your break-even price is ₹{bep:,.0f}/quintal. "
                "Ensure you sell above this price to cover all costs."
            )

    return " ".join(lines)


def _risk_section(req: AdviseRequest, profile: dict, scores: dict[str, float]) -> str:
    ra         = req.risk_assessment
    composite  = ra.composite_risk
    level      = ra.risk_level
    top_factor, top_score = _highest_risk_factor(scores)

    # Plain-language risk mapping
    if composite <= 25:
        overall_msg = "Conditions look good overall — risk is low."
    elif composite <= 50:
        overall_msg = "There is moderate risk. Take some precautions before proceeding."
    elif composite <= 75:
        overall_msg = "Risk is high. Act carefully and monitor the crop closely."
    else:
        overall_msg = "Risk is very high. Consider consulting your local agriculture officer before proceeding."

    # Factor-specific advice
    factor_advice: dict[str, str] = {
        "Soil": (
            f"Soil risk is elevated ({top_score:.1f}/100). "
            f"{profile['soil_tip']} Test soil pH and correct deficiencies before planting."
        ),
        "Disease": (
            f"Disease risk is elevated ({top_score:.1f}/100). "
            f"Watch out for {profile['disease_watch']}. "
            "Apply recommended fungicide/insecticide prophylactically if symptoms appear."
        ),
        "Water": (
            f"Water risk is elevated ({top_score:.1f}/100). "
            f"This crop has {profile['water_need']} water needs. "
            "Schedule irrigation at critical stages (germination, flowering, grain fill)."
        ),
        "Weather": (
            f"Weather risk is elevated ({top_score:.1f}/100). "
            "Monitor daily forecasts. Be ready to cover sensitive crops or adjust harvest timing."
        ),
    }

    specific = factor_advice.get(top_factor, f"Monitor {top_factor.lower()} conditions closely.")
    return f"Risk Level: {level} (composite score {composite:.1f}/100). {overall_msg} Biggest concern: {specific}"


def _action_items(
    req: AdviseRequest,
    profile: dict,
    scores: dict[str, float],
) -> list[str]:
    """Generate exactly 3 prioritised action items based on the data."""
    actions: list[tuple[float, str]] = []

    composite = req.risk_assessment.composite_risk
    pe        = req.profit_estimate
    w         = req.weather
    top_factor, top_score = _highest_risk_factor(scores)

    # --- Action derived from highest risk factor ---
    if top_factor == "Disease" and top_score > 20:
        actions.append((top_score,
            f"Scout your {req.risk_assessment.crop} field today for signs of "
            f"{profile['disease_watch']}. Apply a recommended fungicide or "
            "insecticide immediately if symptoms are detected."
        ))
    elif top_factor == "Water" and top_score > 20:
        actions.append((top_score,
            f"Check soil moisture levels now. "
            f"This crop needs {profile['water_need']} water — "
            "irrigate if topsoil is dry to 5 cm depth."
        ))
    elif top_factor == "Soil" and top_score > 20:
        actions.append((top_score,
            "Get a soil test done this week if you haven't already. "
            f"{profile['soil_tip']} Apply soil amendments before the next irrigation."
        ))
    elif top_factor == "Weather" and top_score > 20:
        actions.append((top_score,
            "Monitor the weather forecast daily for the next 7 days. "
            "Adjust irrigation and spraying schedule to avoid rainfall windows."
        ))

    # --- Weather-triggered action ---
    rain = w.rainfall_forecast_mm or 0.0
    temp = w.temperature          or 25.0
    t_lo, t_hi = profile["optimal_temp"]

    if rain > 50:
        actions.append((50,
            "Heavy rain is forecast — clear drainage channels and check "
            "for waterlogging around plant roots within 24 hours."
        ))
    elif rain == 0.0:
        actions.append((40,
            f"No rain forecast. {profile['sow_tip']} "
            "Plan irrigation for the next 3–5 days to maintain soil moisture."
        ))
    if temp > t_hi:
        actions.append((45,
            f"Heat stress warning ({temp:.1f}°C is above optimal {t_hi}°C). "
            "Irrigate in early morning or evening to cool the root zone."
        ))

    # --- Profit-triggered action ---
    if pe.profit is not None and pe.profit < 0:
        actions.append((60,
            "Current profit projection is negative. Review your input cost plan — "
            "explore whether seed, fertiliser, or labour costs can be reduced, "
            "or consider shifting to the 2nd-ranked crop recommendation."
        ))
    elif pe.break_even_price and pe.market_price and pe.market_price < pe.break_even_price:
        actions.append((55,
            f"Market price (₹{pe.market_price:,.0f}/q) is below your break-even "
            f"(₹{pe.break_even_price:,.0f}/q). Hold stock if you have storage; "
            "sell only when price recovers."
        ))

    # --- Fertiliser / sowing general action ---
    actions.append((30, f"{profile['fertilizer']} Split nitrogen applications for better uptake."))

    # --- Harvest / monitoring action ---
    if composite > 50:
        actions.append((35,
            f"Risk is high. Visit your field every 2–3 days. "
            f"{profile['harvest_tip']}"
        ))
    else:
        actions.append((25, profile["sow_tip"]))

    # Sort by priority (highest first), return top 3
    actions.sort(key=lambda x: -x[0])
    return [f"{i+1}. {a}" for i, (_, a) in enumerate(actions[:3])]


# ── Local fallback engine (always available, no network required) ──────────────

def _generate_local_advice(req: AdviseRequest) -> str:
    """
    Produce rich, data-driven agronomic advice without any external API.

    Uses all four AgriSense pipeline outputs:
      - crop_recommendation  → best crop context & confidence
      - profit_estimate      → financial section
      - weather              → weather advisory
      - risk_assessment      → risk summary & action items
    """
    crop        = req.profit_estimate.crop
    profile     = _get_profile(crop)
    scores      = _risk_scores(req)
    top_crops   = req.crop_recommendation.top_crops
    best        = top_crops[0] if top_crops else None
    crop_label  = crop.capitalize()

    # ── Section 1: Crop Recommendation ────────────────────────────────────────
    if best:
        conf = best.confidence
        alts = ", ".join(
            f"{t.crop.capitalize()} ({t.confidence:.1f}%)"
            for t in top_crops[1:3]
        )
        if conf >= 80:
            crop_section = (
                f"Based on your soil and climate data, {crop_label} is the "
                f"strongly recommended crop with {conf:.1f}% confidence. "
                f"Soil conditions and seasonal parameters align well with this crop. "
                + (f"Alternative options: {alts}." if alts else "")
            )
        elif conf >= 60:
            crop_section = (
                f"{crop_label} is a good fit for your conditions "
                f"(confidence {conf:.1f}%). "
                f"Conditions are favourable but not ideal — follow agronomic best practices closely. "
                + (f"Consider {alts} as alternatives if input costs are a concern." if alts else "")
            )
        else:
            crop_section = (
                f"The model recommends {crop_label} (confidence {conf:.1f}%), but confidence is below 60%. "
                "Conditions may not be optimal. "
                "Consult your local Krishi Vigyan Kendra (KVK) or agriculture officer before committing. "
                + (f"Other options worth exploring: {alts}." if alts else "")
            )
    else:
        crop_section = (
            f"Growing {crop_label} based on your soil inputs. "
            "Please confirm with a local agronomist that conditions are suitable."
        )

    # ── Section 2: Profit Outlook ──────────────────────────────────────────────
    profit_section = _profit_section(req)

    # ── Section 3: Weather Advisory ───────────────────────────────────────────
    weather_section = _weather_advisory(req, profile)

    # ── Section 4: Risk Summary ────────────────────────────────────────────────
    risk_section = _risk_section(req, profile, scores)

    # ── Section 5: Action Items ────────────────────────────────────────────────
    actions = _action_items(req, profile, scores)
    action_block = "\n".join(actions)

    advice = f"""\
Crop Recommendation: {crop_section}

Profit Outlook: {profit_section}

Weather Advisory: {weather_section}

Risk Summary: {risk_section}

Your Top 3 Action Items Today:
{action_block}"""

    return advice.strip()


# ── Endpoint ───────────────────────────────────────────────────────────────────

@app.post("/advise", response_model=AdviseResponse, tags=["advisor"])
def advise(req: AdviseRequest) -> AdviseResponse:
    """
    Produce farmer-friendly agronomic advice from the four AgriSense inputs.

    Always returns HTTP 200 with a useful advisor response:
    - Gemini API is used when a valid GEMINI_API_KEY is configured.
    - If the key is absent, invalid, quota-exceeded, or the network is down,
      the built-in local agronomic engine generates detailed advice instead.
    """
    # ── Attempt Gemini ─────────────────────────────────────────────────────────
    if _API_KEY and _GENAI_AVAILABLE:
        try:
            user_msg = _build_user_message(req)
            advice   = _call_gemini(SYSTEM_PROMPT, user_msg)
            return AdviseResponse(advice=advice, model=_MODEL_NAME, cached=False)
        except Exception as exc:
            # Log the failure and fall through to local engine
            import logging
            logging.getLogger("advisor").warning(
                "Gemini API unavailable (%s). Falling back to local advisor.", exc
            )

    # ── Local fallback — always succeeds ──────────────────────────────────────
    advice = _generate_local_advice(req)
    return AdviseResponse(
        advice=advice,
        model="AgriSense Local Advisor (offline)",
        cached=False,
    )


# ── Health & debug endpoints ───────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
def health():
    return {
        "status":               "ok",
        "gemini_model":         _MODEL_NAME,
        "gemini_api_key_set":   bool(_API_KEY),
        "gemini_lib_available": _GENAI_AVAILABLE,
        "local_advisor":        "always_available",
        "system_prompt_loaded": bool(SYSTEM_PROMPT),
        "system_prompt_chars":  len(SYSTEM_PROMPT),
    }


@app.get("/advise/prompt", tags=["meta"])
def get_system_prompt():
    """Return the active system prompt (useful for debugging/inspection)."""
    return {"system_prompt": SYSTEM_PROMPT, "chars": len(SYSTEM_PROMPT)}


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
