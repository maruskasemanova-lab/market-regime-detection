"""Intrabar checkpoint trace helpers for runtime trading-bar processing."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from ..day_trading_models import BarData, TradingSession
from ..day_trading_runtime.intrabar_slice import (
    runtime_evaluate_intrabar_slice as _runtime_evaluate_intrabar_slice_impl,
)
from ..day_trading_runtime_intrabar import (
    calculate_intrabar_1s_snapshot as _calculate_intrabar_1s_snapshot_impl,
)
from ..day_trading_runtime_sweep import to_optional_float as _to_optional_float_impl



def _quote_midpoint(row: Dict[str, Any]) -> Optional[float]:
    try:
        bid = float(row.get("bid", 0.0) or 0.0)
        ask = float(row.get("ask", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if bid > 0.0 and ask > 0.0:
        return (bid + ask) / 2.0
    if ask > 0.0:
        return ask
    if bid > 0.0:
        return bid
    return None


def _resolve_intrabar_checkpoint_meta(
    *,
    session: TradingSession,
    bar: BarData,
    timestamp: datetime,
) -> tuple[int, List[tuple[BarData, datetime, int]]]:
    raw_step = getattr(getattr(session, "config", None), "intrabar_eval_step_seconds", 5)
    try:
        step = int(raw_step)
    except (TypeError, ValueError):
        step = 5
    step = max(1, min(60, step))

    raw_quotes = getattr(bar, "intrabar_quotes_1s", None)
    checkpoints_meta: List[tuple[BarData, datetime, int]] = []

    if isinstance(raw_quotes, list) and raw_quotes:
        import copy

        normalized_by_second: Dict[int, Dict[str, Any]] = {}
        for item in raw_quotes:
            if not isinstance(item, dict):
                continue
            try:
                sec = int(float(item.get("s", 0) or 0))
            except (TypeError, ValueError):
                continue
            if 0 <= sec <= 59:
                normalized_by_second[sec] = dict(item, s=sec)

        ordered_seconds = sorted(normalized_by_second.keys())
        prefix_quotes: List[Dict[str, Any]] = []
        for idx, sec in enumerate(ordered_seconds):
            prefix_quotes.append(normalized_by_second[sec])
            is_boundary = (sec % step == 0) or (sec == 59) or (idx == len(ordered_seconds) - 1)
            if not is_boundary:
                continue

            cp_bar = copy.copy(bar)
            cp_quotes = [dict(row) for row in prefix_quotes]
            cp_bar.intrabar_quotes_1s = cp_quotes
            cp_bar.l2_delta = None
            cp_bar.l2_buy_volume = None
            cp_bar.l2_sell_volume = None
            cp_bar.l2_volume = None
            cp_bar.l2_imbalance = None
            cp_bar.l2_bid_depth_total = None
            cp_bar.l2_ask_depth_total = None
            cp_bar.l2_book_pressure = None
            cp_bar.l2_book_pressure_change = None
            cp_bar.l2_iceberg_buy_count = None
            cp_bar.l2_iceberg_sell_count = None
            cp_bar.l2_iceberg_bias = None
            cp_bar.l2_quality_flags = None
            cp_bar.l2_quality = None

            mids: List[float] = []
            for row in cp_quotes:
                mid = _quote_midpoint(row)
                if mid is not None and mid > 0.0:
                    mids.append(mid)

            if mids:
                base_open = float(bar.open) if float(bar.open) > 0.0 else mids[0]
                cp_open = float(base_open)
                cp_close = float(mids[-1])
                cp_high = float(max([cp_open] + mids))
                cp_low = float(min([cp_open] + mids))
                cp_vwap: Optional[float] = float(sum(mids) / len(mids))
            else:
                cp_open = float(bar.open)
                cp_high = float(bar.high)
                cp_low = float(bar.low)
                cp_close = float(bar.close)
                cp_vwap = _to_optional_float_impl(getattr(bar, "vwap", None))

            elapsed_ratio = min(1.0, max(1.0 / 60.0, float(sec + 1) / 60.0))
            cp_volume = float(bar.volume or 0.0) * elapsed_ratio

            cp_bar.open = cp_open
            cp_bar.high = cp_high
            cp_bar.low = cp_low
            cp_bar.close = cp_close
            cp_bar.volume = cp_volume
            cp_bar.vwap = cp_vwap

            cp_ts = timestamp.replace(second=sec, microsecond=0)
            checkpoints_meta.append((cp_bar, cp_ts, sec))

    if not checkpoints_meta:
        fallback_sec = max(0, min(int(getattr(timestamp, "second", 0) or 0), 59))
        checkpoints_meta.append((bar, timestamp.replace(microsecond=0), fallback_sec))

    return step, checkpoints_meta
