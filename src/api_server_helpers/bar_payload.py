"""Helpers for API bar payload mapping and serialization safety."""

from __future__ import annotations

from datetime import datetime
import math
from typing import Any, Dict


_BASE_BAR_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
)

_L2_BAR_FIELDS = (
    "l2_delta",
    "l2_buy_volume",
    "l2_sell_volume",
    "l2_volume",
    "l2_imbalance",
    "l2_bid_depth_total",
    "l2_ask_depth_total",
    "l2_book_pressure",
    "l2_book_pressure_change",
    "l2_iceberg_buy_count",
    "l2_iceberg_sell_count",
    "l2_iceberg_bias",
    "l2_quality",
)

_INTRABAR_FIELDS = (
    "intrabar_quotes_1s",
)

_TCBBO_BAR_FIELDS = (
    "tcbbo_net_premium",
    "tcbbo_cumulative_net_premium",
    "tcbbo_call_buy_premium",
    "tcbbo_put_buy_premium",
    "tcbbo_call_sell_premium",
    "tcbbo_put_sell_premium",
    "tcbbo_sweep_count",
    "tcbbo_sweep_premium",
    "tcbbo_trade_count",
    "tcbbo_has_data",
)


def parse_bar_timestamp(timestamp_value: str) -> datetime:
    """Parse API timestamp value used by session endpoints."""
    return datetime.fromisoformat(str(timestamp_value).replace("Z", "+00:00"))


def build_day_trading_bar_payload(
    bar: Any,
    *,
    include_l2_quality_flags: bool = False,
) -> Dict[str, Any]:
    """Build manager-facing bar payload from a BarInput-like object."""
    payload: Dict[str, Any] = {}
    for key in (*_BASE_BAR_FIELDS, *_L2_BAR_FIELDS, *_INTRABAR_FIELDS, *_TCBBO_BAR_FIELDS):
        payload[key] = getattr(bar, key, None)

    if include_l2_quality_flags:
        payload["l2_quality_flags"] = getattr(bar, "l2_quality_flags", None)

    return payload


def sanitize_non_finite_numbers(value: Any) -> Any:
    """Recursively replace NaN/Inf with None for JSON serialization safety."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: sanitize_non_finite_numbers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_non_finite_numbers(item) for item in value]
    return value


__all__ = [
    "build_day_trading_bar_payload",
    "parse_bar_timestamp",
    "sanitize_non_finite_numbers",
]
