"""Helpers for API bar payload mapping and serialization safety."""

from __future__ import annotations

from datetime import datetime
import math
from typing import Any, Dict, Mapping


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


def parse_bar_timestamp(timestamp_value: Any) -> datetime:
    """Parse API timestamp value used by session endpoints."""
    if isinstance(timestamp_value, datetime):
        return timestamp_value
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


def build_day_trading_bar_payload_from_mapping(
    bar: Mapping[str, Any],
    *,
    include_l2_quality_flags: bool = False,
) -> Dict[str, Any]:
    """Build manager-facing bar payload from a dict-like batch row."""
    payload: Dict[str, Any] = {}
    for key in (*_BASE_BAR_FIELDS, *_L2_BAR_FIELDS, *_INTRABAR_FIELDS, *_TCBBO_BAR_FIELDS):
        payload[key] = bar.get(key)

    if payload.get("l2_book_pressure_change") is None and "l2_book_pressure_delta" in bar:
        payload["l2_book_pressure_change"] = bar.get("l2_book_pressure_delta")

    if include_l2_quality_flags:
        payload["l2_quality_flags"] = bar.get("l2_quality_flags")

    return payload


def sanitize_non_finite_numbers(value: Any) -> Any:
    """Recursively replace NaN/Inf with None for JSON serialization safety."""
    import math
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if type(value).__name__ in ('float32', 'float64', 'float16'):
        import numpy as np
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, dict):
        sanitized_dict = {}
        for key, item in value.items():
            safe_k = sanitize_non_finite_numbers(key)
            if safe_k is None:
                if isinstance(key, float):
                    import math
                    if math.isnan(key):
                        safe_k = "NaN"
                    elif key > 0:
                        safe_k = "Infinity"
                    else:
                        safe_k = "-Infinity"
                elif type(key).__name__ in ('float32', 'float64', 'float16'):
                    import numpy as np
                    if np.isnan(key):
                        safe_k = "NaN"
                    elif key > 0:
                        safe_k = "Infinity"
                    else:
                        safe_k = "-Infinity"
                else:
                    safe_k = "NaN"
            sanitized_dict[safe_k] = sanitize_non_finite_numbers(item)
        return sanitized_dict
    if isinstance(value, (list, tuple)):
        return [sanitize_non_finite_numbers(item) for item in value]
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        return sanitize_non_finite_numbers(value.to_dict())
    return value


__all__ = [
    "build_day_trading_bar_payload",
    "build_day_trading_bar_payload_from_mapping",
    "parse_bar_timestamp",
    "sanitize_non_finite_numbers",
]
