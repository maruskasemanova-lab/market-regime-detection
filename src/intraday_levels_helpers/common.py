"""Shared helpers for intraday levels tracking."""

from __future__ import annotations

from typing import Any, Dict, List


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def trim_sequence(values: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    if len(values) <= limit:
        return values
    return values[-limit:]


__all__ = ["to_float", "trim_sequence"]
