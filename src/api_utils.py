"""
Utility helpers for strategy API layer.
"""
from __future__ import annotations

from typing import Iterable, List

from .strategies.base_strategy import Regime


def get_regime_description(regime: str) -> str:
    descriptions = {
        "TRENDING": "Strong directional movement. Pullback and Momentum strategies preferred.",
        "CHOPPY": "Range-bound, low trend efficiency. Mean Reversion and VWAP Magnet strategies preferred.",
        "MIXED": "Uncertain direction. Rotation and conservative strategies preferred.",
    }
    return descriptions.get(regime, "Unknown regime")


def coerce_regimes(reg_list: Iterable[object]) -> List[Regime]:
    """Accept list of Regime enums or strings; return list[Regime]."""
    normalized: List[Regime] = []
    for raw in reg_list:
        if isinstance(raw, Regime):
            normalized.append(raw)
            continue
        try:
            normalized.append(Regime[str(raw).upper()])
        except Exception:
            continue
    return normalized


def normalize_strategy_name(strategy_name: str) -> str:
    """Normalize strategy labels to internal snake_case endpoint key."""
    return strategy_name.lower().replace(" ", "_")

