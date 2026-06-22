"""
price_index.py
──────────────
Builds an in-memory price index from the Agmarknet CSV at startup.

Index structure:
    { commodity → { state → { "modal": float, "min": float, "max": float,
                               "date": str, "market": str } } }

For each (commodity, state) pair we keep only the LATEST entry by date.
A national fallback (key="__national__") stores the median across all states.

Usage (import once, reuse):
    from price_index import PriceIndex
    idx = PriceIndex(CSV_PATH)
    result = idx.lookup("Maize", state="Gujarat")
"""

from __future__ import annotations

import csv
import statistics
from datetime import datetime
from pathlib import Path
from typing import Optional


_DATE_FMT = "%d %b %Y"   # "05 Apr 2025"


def _parse_date(s: str) -> datetime:
    try:
        return datetime.strptime(s.strip(), _DATE_FMT)
    except ValueError:
        return datetime.min


def _parse_price(s: str) -> Optional[float]:
    try:
        v = float(s.strip())
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None


class PriceIndex:
    """Lazy-loaded price lookup for Agmarknet commodity data."""

    def __init__(self, csv_path: str | Path) -> None:
        self._path = Path(csv_path)
        self._index: dict[str, dict[str, dict]] = {}
        self._build()

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        # Per-commodity, per-state: keep the most recent row
        latest: dict[str, dict[str, dict]] = {}

        with open(self._path, encoding="utf-8", errors="ignore") as f:
            for row in csv.DictReader(f):
                commodity = row["Commodity"].strip()
                state     = row["State"].strip()
                date      = _parse_date(row["Price Date"])
                modal     = _parse_price(row["Modal Price (Rs./Quintal)"])
                mn        = _parse_price(row["Min Price (Rs./Quintal)"])
                mx        = _parse_price(row["Max Price (Rs./Quintal)"])
                market    = row["Market Name"].strip()

                if modal is None:
                    continue

                entry = dict(modal=modal, min=mn, max=mx,
                             date=row["Price Date"].strip(), market=market,
                             _dt=date)

                latest.setdefault(commodity, {})
                existing = latest[commodity].get(state)
                if existing is None or date > existing["_dt"]:
                    latest[commodity][state] = entry

        # Build final index (drop _dt sentinel) + national median
        for commodity, states in latest.items():
            self._index[commodity] = {}
            modals = []
            for state, entry in states.items():
                clean = {k: v for k, v in entry.items() if k != "_dt"}
                self._index[commodity][state] = clean
                modals.append(entry["modal"])

            # National fallback = median modal price across all states
            if modals:
                nat_modal = statistics.median(modals)
                self._index[commodity]["__national__"] = dict(
                    modal=nat_modal,
                    min=min(e["min"] or e["modal"] for e in states.values()),
                    max=max(e["max"] or e["modal"] for e in states.values()),
                    date="national median",
                    market="national median",
                )

    # ── Lookup ────────────────────────────────────────────────────────────────

    def lookup(
        self,
        commodity: str,
        state: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Return price dict for the given commodity.
        Falls back: requested state → national median → None.
        """
        states = self._index.get(commodity)
        if states is None:
            return None
        if state and state in states:
            return states[state]
        return states.get("__national__")

    @property
    def commodities(self) -> list[str]:
        return sorted(self._index.keys())

    @property
    def states(self) -> list[str]:
        st = set()
        for s in self._index.values():
            st.update(k for k in s if k != "__national__")
        return sorted(st)
