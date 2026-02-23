from __future__ import annotations

from typing import Any, Dict, List, Optional

from .shared import safe_intrabar_quote as _safe_intrabar_quote
from .shared import to_float as _to_float


def row_float(row: Any, key: str, default: float = 0.0) -> float:
    if isinstance(row, dict):
        return _to_float(row.get(key), default)
    return _to_float(getattr(row, key, None), default)


def true_range(current_high: float, current_low: float, prev_close: Optional[float]) -> float:
    hl = max(0.0, current_high - current_low)
    if prev_close is None:
        return hl
    hc = abs(current_high - prev_close)
    lc = abs(current_low - prev_close)
    return max(hl, hc, lc)


def rolling_atr_pct(bars: List[Any], window: int = 14) -> float:
    if not bars or len(bars) < 2:
        return 0.0
    tr_values: List[float] = []
    prev_close: Optional[float] = None
    for bar in bars[-(window + 1) :]:
        high = row_float(bar, "high", 0.0)
        low = row_float(bar, "low", 0.0)
        close = row_float(bar, "close", 0.0)
        if high <= 0.0 or low <= 0.0 or close <= 0.0:
            prev_close = close if close > 0 else prev_close
            continue
        tr_values.append(true_range(high, low, prev_close))
        prev_close = close
    if not tr_values:
        return 0.0
    atr = sum(tr_values[-window:]) / max(1, min(window, len(tr_values)))
    ref_close = row_float(bars[-1], "close", 0.0)
    if ref_close <= 0.0:
        return 0.0
    return (atr / ref_close) * 100.0


def aggregate_recent_5m_bars(bars: List[Any], max_groups: int = 24) -> List[Dict[str, float]]:
    if not bars:
        return []
    recent = bars[-(max_groups * 5) :]
    grouped: List[Dict[str, float]] = []
    for idx in range(0, len(recent), 5):
        chunk = recent[idx : idx + 5]
        if len(chunk) < 5:
            continue
        opens = [_to_float(getattr(item, "open", None), 0.0) for item in chunk]
        highs = [_to_float(getattr(item, "high", None), 0.0) for item in chunk]
        lows = [_to_float(getattr(item, "low", None), 0.0) for item in chunk]
        closes = [_to_float(getattr(item, "close", None), 0.0) for item in chunk]
        if not opens or not highs or not lows or not closes:
            continue
        grouped.append(
            {
                "open": opens[0],
                "high": max(highs),
                "low": min(lows),
                "close": closes[-1],
                "l2_delta": sum(_to_float(getattr(item, "l2_delta", None), 0.0) for item in chunk),
                "l2_imbalance": sum(_to_float(getattr(item, "l2_imbalance", None), 0.0) for item in chunk)
                / float(len(chunk)),
                "l2_book_pressure": sum(
                    _to_float(getattr(item, "l2_book_pressure", None), 0.0) for item in chunk
                )
                / float(len(chunk)),
            }
        )
    return grouped


def rolling_l2_bias_5m(agg_5m_bars: List[Dict[str, float]]) -> float:
    if not agg_5m_bars:
        return 0.0
    sample = agg_5m_bars[-6:]
    bias_sum = 0.0
    for row in sample:
        delta = _to_float(row.get("l2_delta"), 0.0)
        imbalance = _to_float(row.get("l2_imbalance"), 0.0)
        book = _to_float(row.get("l2_book_pressure"), 0.0)
        delta_component = max(-1.0, min(1.0, delta / 10000.0))
        bias_sum += (0.40 * delta_component) + (0.35 * imbalance) + (0.25 * book)
    return bias_sum / float(len(sample))


def infer_5m_regime(agg_5m_bars: List[Dict[str, float]]) -> str:
    if len(agg_5m_bars) < 3:
        return "unknown"
    sample = agg_5m_bars[-6:]
    first_close = _to_float(sample[0].get("close"), 0.0)
    last_close = _to_float(sample[-1].get("close"), 0.0)
    if first_close <= 0.0 or last_close <= 0.0:
        return "unknown"
    drift_pct = ((last_close - first_close) / first_close) * 100.0
    tr_values: List[float] = []
    prev_close: Optional[float] = None
    for row in sample:
        tr_values.append(
            true_range(
                _to_float(row.get("high"), 0.0),
                _to_float(row.get("low"), 0.0),
                prev_close,
            )
        )
        prev_close = _to_float(row.get("close"), 0.0)
    atr = sum(tr_values) / float(len(tr_values)) if tr_values else 0.0
    atr_pct = (atr / last_close) * 100.0 if last_close > 0 else 0.0
    if abs(drift_pct) >= max(0.25, atr_pct * 0.6):
        return "trending"
    return "choppy"


def intrabar_spread_bps(bar: Any) -> Optional[float]:
    rows = getattr(bar, "intrabar_quotes_1s", None)
    if not isinstance(rows, list) or not rows:
        return None
    spreads: List[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        bid = _safe_intrabar_quote(row.get("bid"))
        ask = _safe_intrabar_quote(row.get("ask"))
        if bid <= 0.0 or ask <= 0.0:
            continue
        mid = (bid + ask) / 2.0
        if mid <= 0.0:
            continue
        spreads.append(((ask - bid) / mid) * 10000.0)
    if not spreads:
        return None
    return sum(spreads) / float(len(spreads))


__all__ = [
    "aggregate_recent_5m_bars",
    "infer_5m_regime",
    "intrabar_spread_bps",
    "rolling_atr_pct",
    "rolling_l2_bias_5m",
]
