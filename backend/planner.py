"""
planner.py
──────────
AgriSense-AI — Crop lifecycle planner router.

Calculates a calendar of operations (sowing, irrigation, fertilizing, harvest)
given a crop name and sowing date, using the crop's lifecycle JSON.

Endpoints
  POST /planner/schedule  → returns full operational calendar with dates
  GET  /planner/crops     → returns list of supported crops
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent          # backend/
_WORKSPACE = _HERE.parent                        # AgriSense/
_LIFECYCLE_DIR = _HERE / "lifecycle"             # backend/lifecycle/

# ── Load and verify supported crops at startup ────────────────────────────────
_SUPPORTED_CROPS: List[str] = []
if _LIFECYCLE_DIR.exists():
    _SUPPORTED_CROPS = sorted([p.stem for p in _LIFECYCLE_DIR.glob("*.json")])

print("=" * 60, flush=True)
print("  AgriSense-AI Crop Planner Service - Startup", flush=True)
print(f"  Lifecycle directory: {_LIFECYCLE_DIR}", flush=True)
print(f"  Discovered {len(_SUPPORTED_CROPS)} crops: {', '.join(_SUPPORTED_CROPS)}", flush=True)
print("=" * 60, flush=True)

# ── Crop optimal month ranges (for India-centric conditions) ──────────────────
# maps crop_id -> set of optimal months (1-12)
_OPTIMAL_MONTHS: Dict[str, set] = {
    "wheat": {11, 12},     # November – December
    "rice": {6, 7},        # June – July
    "cotton": {5, 6},      # May – June
    "soybean": {6, 7},     # June – July
    "apple": {12, 1},      # December – January
    "banana": {6, 7, 9, 10}, # June, July, September, October
    "blackgram": {6, 7, 10}, # June, July, October
    "chickpea": {10, 11},  # October – November
    "coconut": {6, 7, 8, 9}, # June – September
    "coffee": {6, 7, 8},   # June – August
    "grapes": {10, 11},    # October – November
    "jute": {3, 4, 5},     # March – May
    "kidneybeans": {10, 11}, # October – November
    "lentil": {10, 11},    # October – November
    "maize": {6, 7, 10, 11}, # June, July, October, November
    "mango": {7, 8},       # July – August
    "mothbeans": {6, 7},   # June – July
    "mungbean": {3, 4, 6, 7}, # March, April, June, July
    "muskmelon": {2, 3},   # February – March
    "orange": {6, 7, 8},   # June – August
    "papaya": {6, 7, 2, 3}, # June, July, February, March
    "pigeonpeas": {6, 7},  # June – July
    "pomegranate": {6, 7, 1, 2}, # June, July, January, February
    "watermelon": {1, 2, 3} # January – March
}

# ── APIRouter ─────────────────────────────────────────────────────────────────
router = APIRouter()

# ── Schemas ───────────────────────────────────────────────────────────────────

class ScheduleRequest(BaseModel):
    crop: str = Field(..., description="Crop name (e.g., wheat, rice, cotton, soybean)")
    sowing_date: date = Field(..., description="Target sowing date (YYYY-MM-DD)")

    @field_validator("crop", mode="before")
    @classmethod
    def _normalise_crop(cls, v: str) -> str:
        return v.strip().lower()


class ScheduleEvent(BaseModel):
    event_type: str = Field(..., description="Type of event: sowing, irrigation, fertilizer, harvest")
    stage_name: str = Field(..., description="Name of the stage or operation")
    planned_date: str = Field(..., description="Calculated date of operation (YYYY-MM-DD)")
    days_after_sowing: int = Field(..., description="Days relative to sowing date")
    importance: str = Field(..., description="Importance level: Critical or Moderate")
    details: Dict[str, Any] = Field(..., description="Stage-specific operational parameters (seed rate, depth, NPK, etc.)")
    description: str = Field(..., description="Agronomic instructions or notes")


class ScheduleResponse(BaseModel):
    crop: str
    display_name: str
    season: str
    sowing_date: str
    estimated_harvest_date: str
    planting_window_status: str = Field(..., description="Status: optimal, early_suboptimal, late_suboptimal, outside_window")
    window_message: str
    schedule: List[ScheduleEvent]


# ── Sowing Window Validator Helper ────────────────────────────────────────────

def validate_planting_window(crop_id: str, sowing: date, window_text: str) -> tuple[str, str]:
    if crop_id not in _OPTIMAL_MONTHS:
        return "optimal", "Validation data not available. Proceeding with standard timing."

    opt_months = _OPTIMAL_MONTHS[crop_id]
    sow_month = sowing.month

    if sow_month in opt_months:
        return "optimal", f"Sowing date is within the optimal planting window ({window_text})."

    # Check if early (1 month before the earliest month in range)
    min_opt = min(opt_months)
    early_month = 12 if min_opt == 1 else min_opt - 1
    if sow_month == early_month:
        return "early_suboptimal", f"Sowing date is slightly early. The optimal window is {window_text}. Early sowing may face germinating temperature issues."

    # Check if late (1 month after the latest month in range)
    max_opt = max(opt_months)
    late_month = 1 if max_opt == 12 else max_opt + 1
    if sow_month == late_month:
        return "late_suboptimal", f"Sowing date is slightly late. The optimal window is {window_text}. Late sowing may reduce yields or increase pest vulnerability."

    return "outside_window", f"Sowing date is outside the recommended planting window ({window_text}). Consider adjusting your planting schedule."


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/planner/crops", tags=["planner"])
def get_supported_crops():
    """Returns a list of crops that have lifecycle data files available."""
    if not _LIFECYCLE_DIR.exists():
        return {"crops": []}

    crops = []
    for path in _LIFECYCLE_DIR.glob("*.json"):
        crops.append(path.stem)
    return {"crops": sorted(crops)}


@router.post("/planner/schedule", response_model=ScheduleResponse, tags=["planner"])
def generate_schedule(req: ScheduleRequest):
    json_path = _LIFECYCLE_DIR / f"{req.crop}.json"
    if not json_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Lifecycle plan for crop '{req.crop}' not found. Supported crops: {get_supported_crops()['crops']}"
        )

    try:
        with open(json_path, "r") as f:
            lifecycle = json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading lifecycle file: {str(e)}"
        )

    stages = lifecycle.get("stages", {})
    sowing_info = stages.get("sowing", {})
    irrigation_info = stages.get("irrigation", [])
    fertilizer_info = stages.get("fertilizer", [])
    harvest_info = stages.get("harvest", {})

    # 1. Sowing Window status validation
    window_text = sowing_info.get("planting_window", "N/A")
    window_status, window_msg = validate_planting_window(req.crop, req.sowing_date, window_text)

    # 2. Build chronological schedule
    schedule: List[ScheduleEvent] = []

    # Sowing Event
    sow_desc = (
        f"Sow seeds using '{sowing_info.get('method', 'standard method')}' "
        f"at a depth of {sowing_info.get('sowing_depth_cm', {}).get('min', 3)}-{sowing_info.get('sowing_depth_cm', {}).get('max', 5)} cm. "
        f"Maintain spacing of {sowing_info.get('row_spacing_cm', 'N/A')} cm (row) x {sowing_info.get('plant_spacing_cm', 'N/A')} cm (plant). "
        f"{sowing_info.get('soil_conditions', '')}"
    )
    schedule.append(
        ScheduleEvent(
            event_type="sowing",
            stage_name="Sowing & Establishment",
            planned_date=req.sowing_date.isoformat(),
            days_after_sowing=0,
            importance="Critical",
            details={
                "method": sowing_info.get("method"),
                "seed_rate_kg_ha": sowing_info.get("seed_rate", {}).get("value_range_kg_ha"),
                "sowing_depth_cm": sowing_info.get("sowing_depth_cm"),
                "row_spacing_cm": sowing_info.get("row_spacing_cm"),
                "plant_spacing_cm": sowing_info.get("plant_spacing_cm"),
            },
            description=sow_desc
        )
    )

    # Irrigation Events
    for irr in irrigation_info:
        das = irr.get("days_after_sowing", 0)
        event_date = req.sowing_date + timedelta(days=das)
        irr_desc = f"Apply irrigation ({irr.get('water_depth_mm', 0)} mm water depth). {irr.get('notes', '')}"
        schedule.append(
            ScheduleEvent(
                event_type="irrigation",
                stage_name=irr.get("stage_name", "Irrigation"),
                planned_date=event_date.isoformat(),
                days_after_sowing=das,
                importance=irr.get("importance", "Moderate"),
                details={
                    "water_depth_mm": irr.get("water_depth_mm")
                },
                description=irr_desc
            )
        )

    # Fertilizer Events
    for fert in fertilizer_info:
        das = fert.get("days_after_sowing", 0)
        event_date = req.sowing_date + timedelta(days=das)
        nutrients = fert.get("nutrients_kg_ha", {})
        fert_desc = (
            f"Apply fertilizer via '{fert.get('method', 'broadcasting')}'. "
            f"Nutrients (kg/ha): N={nutrients.get('N', 0)}, P={nutrients.get('P', 0)}, K={nutrients.get('K', 0)}"
        )
        if "S" in nutrients:
            fert_desc += f", S={nutrients.get('S')}"
        fert_desc += f". {fert.get('notes', '')}"

        schedule.append(
            ScheduleEvent(
                event_type="fertilizer",
                stage_name=fert.get("stage_name", "Fertilizer Application"),
                planned_date=event_date.isoformat(),
                days_after_sowing=das,
                importance="Critical" if das == 0 else "Moderate",
                details={
                    "nutrients_kg_ha": nutrients,
                    "method": fert.get("method")
                },
                description=fert_desc
            )
        )

    # Harvest Event
    harvest_das = harvest_info.get("days_after_sowing", 120)
    harvest_date = req.sowing_date + timedelta(days=harvest_das)
    indicators = ", ".join(harvest_info.get("maturity_indicators", []))
    harv_desc = (
        f"Harvest crop using '{harvest_info.get('harvest_method', 'manual')}' method. "
        f"Maturity indicators: {indicators}. "
        f"Target grain moisture: {harvest_info.get('optimal_grain_moisture_pct')}%."
    )
    schedule.append(
        ScheduleEvent(
            event_type="harvest",
            stage_name="Harvesting",
            planned_date=harvest_date.isoformat(),
            days_after_sowing=harvest_das,
            importance="Critical",
            details={
                "expected_yield_range_t_ha": harvest_info.get("expected_yield_range_t_ha"),
                "optimal_grain_moisture_pct": harvest_info.get("optimal_grain_moisture_pct"),
                "harvest_method": harvest_info.get("harvest_method")
            },
            description=harv_desc
        )
    )

    # Sort schedule chronologically by days_after_sowing (and event type to group sowing first at day 0)
    schedule.sort(key=lambda x: (x.days_after_sowing, 0 if x.event_type == "sowing" else 1))

    return ScheduleResponse(
        crop=req.crop,
        display_name=lifecycle.get("display_name", req.crop.capitalize()),
        season=lifecycle.get("season", "N/A"),
        sowing_date=req.sowing_date.isoformat(),
        estimated_harvest_date=harvest_date.isoformat(),
        planting_window_status=window_status,
        window_message=window_msg,
        schedule=schedule
    )
