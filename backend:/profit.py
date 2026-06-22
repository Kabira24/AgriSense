"""
profit.py
─────────
FastAPI profit estimation service for AgriSense-AI.

Routes
──────
  Meta
    GET  /health                Liveness probe
    GET  /commodities           All Agmarknet commodities in the price index
    GET  /commodities/{name}    Prices for one commodity across all states

  Crops catalogue
    GET  /crops                 All crops with cost, commodity, price-availability
    GET  /crops/{crop}          Single crop cost + live price card
    GET  /states                States available in the Agmarknet price dataset

  Profit estimation
    POST /profit                Full profit estimate for one crop/field
    POST /profit/batch          Profit estimates for multiple crop/field combos
    POST /profit/compare        Side-by-side profit comparison for several crops
    POST /profit/breakeven      Break-even price only (lightweight, no revenue)

Usage
  python backend:/profit.py
  uvicorn backend:.profit:app --reload --port 8001
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Path as FPath, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_BACKEND   = Path(__file__).parent
_WORKSPACE = _BACKEND.parent
sys.path.insert(0, str(_BACKEND))

from profit_data import CROP_TO_COMMODITY, COST_PER_HECTARE
from price_index  import PriceIndex


def _resolve(base: Path, *candidates: str) -> Path:
    for name in candidates:
        p = base / name
        if p.exists():
            return p
    raise FileNotFoundError(f"None of {candidates} found under {base}")


_DS_DIR   = _resolve(_WORKSPACE, "datasets:", "datasets")
_CSV_PATH = (
    _DS_DIR
    / "agmarknet-india-commodity-prices-2024-2025"
    / "agmarknet_india_historical_prices_2024_2025.csv"
)

# ── Load price index once at startup ──────────────────────────────────────────
print("Building price index…", flush=True)
_PRICE_IDX = PriceIndex(_CSV_PATH)
print(
    f"  Indexed {len(_PRICE_IDX.commodities)} commodities "
    f"across {len(_PRICE_IDX.states)} states.",
    flush=True,
)

TONNES_TO_QUINTALS = 10  # 1 tonne = 10 quintals

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AgriSense-AI Profit Estimation Service",
    description=(
        "Estimates crop revenue, cultivation cost, and net profit "
        "using Agmarknet 2024-25 price data (CACP Cost-A2+FL basis).\n\n"
        "**Price coverage**: 7 of 22 crops have live Agmarknet prices. "
        "All 22 crops have cultivation cost estimates.\n\n"
        "Use `POST /profit/compare` to rank multiple crops by profitability "
        "for your field and state."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _lookup_price(crop: str, state: Optional[str]) -> Optional[dict]:
    """Return Agmarknet price record for a crop, or None if unavailable."""
    commodity = CROP_TO_COMMODITY.get(crop)
    if not commodity:
        return None
    return _PRICE_IDX.lookup(commodity, state=state)


def _price_source(crop: str, state: Optional[str], record: dict) -> str:
    commodity = CROP_TO_COMMODITY[crop]
    if (
        state
        and state in _PRICE_IDX._index.get(commodity, {})
    ):
        return state
    return "national median"


def _calc(
    crop: str,
    yield_t_ha: float,
    area_ha: float,
    state: Optional[str],
) -> dict:
    """
    Core profit calculation. Returns a plain dict that maps to ProfitResult.
    Raises HTTPException(422) for unsupported crops.
    """
    crop = crop.strip().lower()

    cost_per_ha = COST_PER_HECTARE.get(crop)
    if cost_per_ha is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown crop '{crop}'. Supported: {sorted(COST_PER_HECTARE)}",
        )

    production_q = yield_t_ha * area_ha * TONNES_TO_QUINTALS
    total_cost   = cost_per_ha * area_ha
    break_even   = round(total_cost / production_q, 2) if production_q > 0 else None

    price_record = _lookup_price(crop, state)
    price_src    = _price_source(crop, state, price_record) if price_record else None

    revenue = revenue_min = revenue_max = None
    profit  = profit_margin = None

    if price_record:
        modal   = price_record["modal"]
        mn      = price_record.get("min") or modal
        mx      = price_record.get("max") or modal
        revenue     = round(production_q * modal, 2)
        revenue_min = round(production_q * mn,    2)
        revenue_max = round(production_q * mx,    2)
        profit      = round(revenue - total_cost,  2)
        profit_margin = round((profit / revenue) * 100, 2) if revenue > 0 else None

    return {
        "crop":                crop,
        "area_ha":             area_ha,
        "yield_t_ha":          yield_t_ha,
        "production_quintals": round(production_q, 2),
        "market_price_modal":  price_record["modal"]       if price_record else None,
        "market_price_min":    price_record.get("min")     if price_record else None,
        "market_price_max":    price_record.get("max")     if price_record else None,
        "price_source":        price_src,
        "price_date":          price_record.get("date")    if price_record else None,
        "price_available":     price_record is not None,
        "revenue":             revenue,
        "revenue_min":         revenue_min,
        "revenue_max":         revenue_max,
        "cost":                round(total_cost, 2),
        "profit":              profit,
        "profit_margin_pct":   profit_margin,
        "break_even_price":    break_even,
        "commodity_matched":   CROP_TO_COMMODITY.get(crop),
        "cost_per_ha":         cost_per_ha,
    }


# ── Schemas ────────────────────────────────────────────────────────────────────

class ProfitResult(BaseModel):
    """Full profit breakdown for one crop / field combination."""
    crop:                str
    area_ha:             float
    yield_t_ha:          float
    production_quintals: float

    # Prices (Rs./quintal)
    market_price_modal:  Optional[float] = None
    market_price_min:    Optional[float] = None
    market_price_max:    Optional[float] = None
    price_source:        Optional[str]   = None
    price_date:          Optional[str]   = None
    price_available:     bool

    # Financials (Rs.)
    revenue:             Optional[float] = None
    revenue_min:         Optional[float] = None
    revenue_max:         Optional[float] = None
    cost:                float
    profit:              Optional[float] = None
    profit_margin_pct:   Optional[float] = None
    break_even_price:    Optional[float] = Field(None, description="Rs./quintal")

    # Metadata
    commodity_matched:   Optional[str]   = None
    cost_per_ha:         float


class ProfitRequest(BaseModel):
    crop:       str   = Field(..., description="Crop name (from recommender)")
    yield_t_ha: float = Field(..., gt=0, le=100,      description="Yield in tonnes/hectare")
    area_ha:    float = Field(..., gt=0, le=100_000,  description="Cultivated area in hectares")
    state:      Optional[str] = Field(None,           description="Indian state (optional)")

    @field_validator("crop", mode="before")
    @classmethod
    def _norm_crop(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("state", mode="before")
    @classmethod
    def _norm_state(cls, v):
        return v.strip().title() if v else None


# ═══════════════════════════════════════════════════════════════════════════════
# META ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/health",
    tags=["meta"],
    summary="Liveness probe",
)
def health():
    """Returns service status and price index coverage statistics."""
    with_price = sum(1 for v in CROP_TO_COMMODITY.values() if v is not None)
    return {
        "status":              "ok",
        "version":             app.version,
        "commodities_indexed": len(_PRICE_IDX.commodities),
        "states_indexed":      len(_PRICE_IDX.states),
        "crops_supported":     len(COST_PER_HECTARE),
        "crops_with_prices":   with_price,
        "price_source":        "Agmarknet 2024-25",
        "cost_source":         "CACP Cost-A2+FL 2023-24",
    }


@app.get(
    "/states",
    tags=["meta"],
    summary="List states in the price dataset",
)
def list_states():
    """Returns all Indian states present in the Agmarknet price index."""
    return {"states": _PRICE_IDX.states, "count": len(_PRICE_IDX.states)}


# ═══════════════════════════════════════════════════════════════════════════════
# COMMODITIES ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/commodities",
    tags=["commodities"],
    summary="List all Agmarknet commodities in the price index",
)
def list_commodities():
    """
    Returns all commodity names indexed from the Agmarknet 2024-25 dataset,
    plus the set of states that have price data for each commodity.
    """
    result = []
    for commodity in _PRICE_IDX.commodities:
        state_data = _PRICE_IDX._index.get(commodity, {})
        states_covered = [k for k in state_data if k != "__national__"]
        national = state_data.get("__national__", {})
        result.append({
            "commodity":       commodity,
            "states_covered":  sorted(states_covered),
            "national_modal":  national.get("modal"),
            "national_min":    national.get("min"),
            "national_max":    national.get("max"),
        })
    return {"commodities": result, "count": len(result)}


@app.get(
    "/commodities/{commodity_name}",
    tags=["commodities"],
    summary="Prices for one commodity across all states",
)
def get_commodity(
    commodity_name: str = FPath(..., description="Agmarknet commodity name"),
):
    """
    Returns the latest modal, min, and max price (Rs./quintal) for the
    requested commodity for every state available in the price index,
    plus the national median.
    """
    # Case-insensitive fuzzy match
    match = next(
        (c for c in _PRICE_IDX.commodities
         if c.lower() == commodity_name.strip().lower()),
        None,
    )
    if match is None:
        raise HTTPException(
            status_code=404,
            detail=f"Commodity '{commodity_name}' not found in price index. "
                   f"Available: {_PRICE_IDX.commodities}",
        )

    state_data = _PRICE_IDX._index[match]
    prices = {
        state: {k: v for k, v in record.items()}
        for state, record in state_data.items()
    }
    return {
        "commodity": match,
        "prices_by_state": prices,
        "states_available": sorted(k for k in prices if k != "__national__"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CROPS CATALOGUE ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

class CropCostCard(BaseModel):
    crop:              str
    cost_per_ha:       float  = Field(..., description="Cultivation cost (Rs./ha)")
    commodity_mapped:  Optional[str]
    price_available:   bool
    national_modal:    Optional[float] = Field(None, description="National median price (Rs./quintal)")


@app.get(
    "/crops",
    response_model=list[CropCostCard],
    tags=["crops"],
    summary="All crops with cost and price-availability",
)
def list_crops(
    price_available: Optional[bool] = Query(
        None, description="Filter: true = only crops with live prices"
    ),
):
    """
    Returns all 22 supported crops with:
    - cultivation cost (Rs./ha, CACP basis)
    - Agmarknet commodity mapping
    - whether live market prices are available
    - national median price if available

    Use `?price_available=true` to filter to the 7 crops with live price data.
    """
    cards = []
    for crop in sorted(COST_PER_HECTARE):
        commodity = CROP_TO_COMMODITY.get(crop)
        record    = _PRICE_IDX.lookup(commodity) if commodity else None
        card = CropCostCard(
            crop             = crop,
            cost_per_ha      = COST_PER_HECTARE[crop],
            commodity_mapped = commodity,
            price_available  = record is not None,
            national_modal   = record["modal"] if record else None,
        )
        if price_available is None or card.price_available == price_available:
            cards.append(card)
    return cards


@app.get(
    "/crops/{crop}",
    response_model=CropCostCard,
    tags=["crops"],
    summary="Cost and price card for one crop",
)
def get_crop(
    crop:  str           = FPath(..., description="Crop name (e.g. rice, maize)"),
    state: Optional[str] = Query(None, description="Indian state for localised price"),
):
    """
    Returns the cultivation cost and latest market price (national or
    state-specific) for a single crop.
    """
    key = crop.strip().lower()
    if key not in COST_PER_HECTARE:
        raise HTTPException(
            status_code=404,
            detail=f"Crop '{crop}' not found. Supported: {sorted(COST_PER_HECTARE)}",
        )
    commodity = CROP_TO_COMMODITY.get(key)
    record    = _PRICE_IDX.lookup(commodity, state=state) if commodity else None
    return CropCostCard(
        crop             = key,
        cost_per_ha      = COST_PER_HECTARE[key],
        commodity_mapped = commodity,
        price_available  = record is not None,
        national_modal   = record["modal"] if record else None,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PROFIT ESTIMATION ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/profit",
    response_model=ProfitResult,
    tags=["profit"],
    summary="Full profit estimate for one crop / field",
)
def estimate_profit(req: ProfitRequest):
    """
    Estimate revenue, cultivation cost, and net profit for a single
    crop on a given field area.

    - `yield_t_ha`: expected yield in tonnes per hectare
    - `area_ha`: cultivated area in hectares
    - `state`: optional — uses national median price if omitted

    Returns modal, min, and max revenue based on Agmarknet prices,
    plus break-even price per quintal.
    """
    return ProfitResult(**_calc(req.crop, req.yield_t_ha, req.area_ha, req.state))


# ── Batch ──────────────────────────────────────────────────────────────────────

class BatchProfitItem(BaseModel):
    id:         Optional[str] = Field(None, description="Caller-supplied identifier")
    crop:       str
    yield_t_ha: float = Field(..., gt=0, le=100)
    area_ha:    float = Field(..., gt=0, le=100_000)
    state:      Optional[str] = None

    @field_validator("crop", mode="before")
    @classmethod
    def _norm(cls, v): return v.strip().lower()

    @field_validator("state", mode="before")
    @classmethod
    def _norm_state(cls, v): return v.strip().title() if v else None


class BatchProfitRequest(BaseModel):
    items: list[BatchProfitItem] = Field(
        ..., min_length=1, max_length=50,
        description="List of 1–50 crop/field combinations",
    )


class BatchProfitResultItem(BaseModel):
    index:  int
    id:     Optional[str]
    result: Optional[ProfitResult] = None
    error:  Optional[str]          = None


class BatchProfitResponse(BaseModel):
    results: list[BatchProfitResultItem]
    count:   int
    errors:  int


@app.post(
    "/profit/batch",
    response_model=BatchProfitResponse,
    tags=["profit"],
    summary="Profit estimates for up to 50 crop/field combos",
)
def estimate_profit_batch(req: BatchProfitRequest):
    """
    Accepts up to **50 crop/field combinations** in a single request.

    Each item is processed independently. Items that fail validation
    (e.g. unsupported crop) are returned with an `error` field rather
    than failing the whole request.

    Useful for analysing multiple farm plots or comparing several crops
    across different field sizes.
    """
    results = []
    error_count = 0

    for i, item in enumerate(req.items):
        try:
            data = _calc(item.crop, item.yield_t_ha, item.area_ha, item.state)
            results.append(BatchProfitResultItem(
                index=i, id=item.id, result=ProfitResult(**data),
            ))
        except HTTPException as exc:
            error_count += 1
            results.append(BatchProfitResultItem(
                index=i, id=item.id, error=exc.detail,
            ))

    return BatchProfitResponse(
        results=results,
        count=len(results),
        errors=error_count,
    )


# ── Compare ────────────────────────────────────────────────────────────────────

class CompareRequest(BaseModel):
    crops:      list[str] = Field(
        ..., min_length=2, max_length=22,
        description="Crops to compare (at least 2)",
    )
    yield_t_ha: float = Field(..., gt=0, le=100,     description="Yield in tonnes/ha")
    area_ha:    float = Field(..., gt=0, le=100_000, description="Area in hectares")
    state:      Optional[str] = None

    @field_validator("crops", mode="before")
    @classmethod
    def _norm_crops(cls, v):
        return [c.strip().lower() for c in v]

    @field_validator("state", mode="before")
    @classmethod
    def _norm_state(cls, v): return v.strip().title() if v else None


class CompareEntry(BaseModel):
    rank:              int
    crop:              str
    profit:            Optional[float] = Field(None, description="Net profit (Rs.)")
    revenue:           Optional[float] = None
    cost:              float
    profit_margin_pct: Optional[float] = None
    break_even_price:  Optional[float] = None
    price_available:   bool


class CompareResponse(BaseModel):
    comparison:    list[CompareEntry]
    most_profit:   Optional[str] = Field(None, description="Crop with highest profit")
    area_ha:       float
    yield_t_ha:    float
    state:         Optional[str]


@app.post(
    "/profit/compare",
    response_model=CompareResponse,
    tags=["profit"],
    summary="Side-by-side profitability ranking for multiple crops",
)
def compare_profit(req: CompareRequest):
    """
    Given identical field conditions (area, yield, state), calculates and
    **ranks** each requested crop by net profit (descending).

    Crops without live Agmarknet prices are ranked last since profit cannot
    be computed, but their break-even price is still returned.

    Useful for answering: *"Between rice, maize, and pigeonpeas, which is
    most profitable on my 2-hectare field in Maharashtra?"*
    """
    unknown = [c for c in req.crops if c not in COST_PER_HECTARE]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown crops: {unknown}. Supported: {sorted(COST_PER_HECTARE)}",
        )

    rows = []
    for crop in req.crops:
        try:
            data = _calc(crop, req.yield_t_ha, req.area_ha, req.state)
            rows.append(data)
        except HTTPException:
            pass

    # Sort: crops with profit first (desc), then break-even only, then no data
    def _sort_key(d):
        if d["profit"] is not None:
            return (0, -(d["profit"]))
        if d["break_even_price"] is not None:
            return (1, d["break_even_price"])
        return (2, 0)

    rows.sort(key=_sort_key)

    comparison = [
        CompareEntry(
            rank              = i + 1,
            crop              = d["crop"],
            profit            = d["profit"],
            revenue           = d["revenue"],
            cost              = d["cost"],
            profit_margin_pct = d["profit_margin_pct"],
            break_even_price  = d["break_even_price"],
            price_available   = d["price_available"],
        )
        for i, d in enumerate(rows)
    ]

    best = next((e.crop for e in comparison if e.profit is not None), None)

    return CompareResponse(
        comparison  = comparison,
        most_profit = best,
        area_ha     = req.area_ha,
        yield_t_ha  = req.yield_t_ha,
        state       = req.state,
    )


# ── Break-even only ────────────────────────────────────────────────────────────

class BreakevenRequest(BaseModel):
    crop:       str   = Field(..., description="Crop name")
    yield_t_ha: float = Field(..., gt=0, le=100)
    area_ha:    float = Field(..., gt=0, le=100_000)

    @field_validator("crop", mode="before")
    @classmethod
    def _norm(cls, v): return v.strip().lower()


class BreakevenResponse(BaseModel):
    crop:              str
    area_ha:           float
    yield_t_ha:        float
    production_quintals: float
    total_cost:        float
    cost_per_ha:       float
    break_even_price:  float = Field(..., description="Min sell price to cover costs (Rs./quintal)")
    national_modal:    Optional[float] = Field(None, description="Current national modal price for reference")
    margin_at_modal:   Optional[float] = Field(None, description="Profit if sold at national modal (Rs.)")


@app.post(
    "/profit/breakeven",
    response_model=BreakevenResponse,
    tags=["profit"],
    summary="Break-even price only — lightweight, no revenue calculation",
)
def breakeven(req: BreakevenRequest):
    """
    Returns only the **break-even price per quintal** — the minimum sell
    price needed to cover cultivation costs.

    Lighter than `POST /profit` — useful for quick sanity checks or
    when you don't need full revenue/profit figures.

    Also returns the current national modal price for reference so the
    farmer can immediately see how far above break-even the market price sits.
    """
    crop = req.crop
    cost_per_ha = COST_PER_HECTARE.get(crop)
    if cost_per_ha is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown crop '{crop}'. Supported: {sorted(COST_PER_HECTARE)}",
        )

    production_q = req.yield_t_ha * req.area_ha * TONNES_TO_QUINTALS
    total_cost   = cost_per_ha * req.area_ha
    break_even   = round(total_cost / production_q, 2) if production_q > 0 else None

    # National modal for reference
    record       = _lookup_price(crop, state=None)
    national_modal = record["modal"] if record else None
    margin = (
        round((national_modal - break_even) * production_q, 2)
        if national_modal and break_even else None
    )

    return BreakevenResponse(
        crop                = crop,
        area_ha             = req.area_ha,
        yield_t_ha          = req.yield_t_ha,
        production_quintals = round(production_q, 2),
        total_cost          = round(total_cost, 2),
        cost_per_ha         = cost_per_ha,
        break_even_price    = break_even,
        national_modal      = national_modal,
        margin_at_modal     = margin,
    )


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "profit:app",
        host      = "0.0.0.0",
        port      = 8001,
        reload    = False,
        log_level = "info",
    )
