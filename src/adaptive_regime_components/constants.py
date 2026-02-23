"""Shared constants for adaptive regime detection."""

from __future__ import annotations

from typing import Dict, List

TRENDING = "TRENDING"
CHOPPY = "CHOPPY"
MIXED = "MIXED"

REGIME_LABELS: List[str] = [TRENDING, CHOPPY, MIXED]

MICRO_REGIMES = [
    "TRENDING_UP",
    "TRENDING_DOWN",
    "BREAKOUT",
    "ABSORPTION",
    "CHOPPY",
    "MIXED",
    "TRANSITION",
    "UNKNOWN",
]


def default_probabilities() -> Dict[str, float]:
    return {TRENDING: 0.33, CHOPPY: 0.33, MIXED: 0.34}
