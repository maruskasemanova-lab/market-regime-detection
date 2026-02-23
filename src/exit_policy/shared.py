from __future__ import annotations

from datetime import time
from typing import Any


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def safe_intrabar_quote(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def bars_held(pos: Any, current_bar_index: int) -> int:
    entry_index = pos.entry_bar_index if pos.entry_bar_index is not None else current_bar_index
    return max(0, int(current_bar_index) - int(entry_index))


def is_midday_window(bar_time: time) -> bool:
    """Return True for lower-conviction midday window."""
    midday_start = time(10, 30)
    midday_end = time(14, 0)
    return midday_start <= bar_time < midday_end


__all__ = [
    "bars_held",
    "is_midday_window",
    "safe_div",
    "safe_intrabar_quote",
    "to_float",
]
