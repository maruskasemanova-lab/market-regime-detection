"""Compatibility wrappers for intraday level entry-quality runtime helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .day_trading_models import TradingSession
from .strategies.base_strategy import Signal
from .day_trading_runtime.entry_quality_gate import (
    _compact_level_payload as _compact_level_payload_impl,
    _event_direction as _event_direction_impl,
    _safe_level_distance_pct as _safe_level_distance_pct_impl,
    _to_optional_float as _to_optional_float_impl,
    runtime_evaluate_intraday_levels_entry_quality as _runtime_evaluate_intraday_levels_entry_quality_impl,
)


def _to_optional_float(value: Any) -> Optional[float]:
    return _to_optional_float_impl(value)


def _safe_level_distance_pct(price: float, level_price: float) -> Optional[float]:
    return _safe_level_distance_pct_impl(price, level_price)


def _event_direction(event: Optional[Dict[str, Any]]) -> int:
    return _event_direction_impl(event)


def _compact_level_payload(level: Dict[str, Any], current_price: float) -> Dict[str, Any]:
    return _compact_level_payload_impl(level, current_price)


def runtime_evaluate_intraday_levels_entry_quality(
    self,
    *,
    session: TradingSession,
    signal: Signal,
    current_price: float,
    current_bar_index: int,
) -> Dict[str, Any]:
    return _runtime_evaluate_intraday_levels_entry_quality_impl(
        self=self,
        session=session,
        signal=signal,
        current_price=current_price,
        current_bar_index=current_bar_index,
    )
