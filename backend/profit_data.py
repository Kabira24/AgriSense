"""
profit_data.py
──────────────
Static lookup tables used by the profit estimation service.

1. CROP_TO_COMMODITY  — maps recommender crop names → Agmarknet commodity names
2. COST_PER_HECTARE   — estimated cultivation cost in Rs./ha (CACP 2023-24 averages)

Crops in the recommender that have NO matching Agmarknet commodity are still
given a cost estimate; revenue will be flagged as "price not available".
"""

# ── 1. Name mapping: recommender label → Agmarknet commodity ─────────────────
# Only crops that exist in the Agmarknet dataset are mapped.
CROP_TO_COMMODITY: dict[str, str] = {
    "apple":       "Apple",
    "banana":      "Banana",
    "blackgram":   None,                              # not in Agmarknet
    "chickpea":    None,                              # not in Agmarknet (chana)
    "coconut":     None,                              # not in Agmarknet
    "coffee":      None,                              # not in Agmarknet
    "cotton":      "Cotton",
    "grapes":      None,                              # not in Agmarknet
    "jute":        None,                              # not in Agmarknet
    "kidneybeans": None,                              # not in Agmarknet
    "lentil":      "Lentil (Masur)(Whole)",
    "maize":       "Maize",
    "mango":       "Mango",
    "mothbeans":   None,                              # not in Agmarknet
    "mungbean":    "Green Gram (Moong)(Whole)",
    "muskmelon":   None,                              # not in Agmarknet
    "orange":      None,                              # not in Agmarknet
    "papaya":      None,                              # not in Agmarknet
    "pigeonpeas":  "Arhar (Tur/Red Gram)(Whole)",
    "pomegranate": None,                              # not in Agmarknet
    "rice":        None,                              # not in Agmarknet (paddy)
    "watermelon":  None,                              # not in Agmarknet
}

# ── 2. Cultivation cost (Rs. per hectare) ────────────────────────────────────
# Source: CACP Cost of Cultivation reports 2023-24 (Cost-A2+FL basis).
# Crops not in CACP reports use conservative regional averages.
COST_PER_HECTARE: dict[str, float] = {
    "apple":       180_000,
    "banana":       90_000,
    "blackgram":    25_000,
    "chickpea":     22_000,
    "coconut":      55_000,
    "coffee":      120_000,
    "cotton":       55_000,
    "grapes":      200_000,
    "jute":         32_000,
    "kidneybeans":  28_000,
    "lentil":       22_000,
    "maize":        30_000,
    "mango":        80_000,
    "mothbeans":    20_000,
    "mungbean":     24_000,
    "muskmelon":    35_000,
    "orange":       90_000,
    "papaya":       60_000,
    "pigeonpeas":   28_000,
    "pomegranate": 120_000,
    "rice":         45_000,
    "watermelon":   40_000,
}
