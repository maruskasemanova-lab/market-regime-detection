"""Intrabar and micro-confirmation helpers for runtime processing."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .day_trading_models import BarData, TradingSession
from .strategies.base_strategy import Signal, SignalType


def signal_direction(signal: Signal) -> int:
    if signal.signal_type == SignalType.BUY:
        return 1
    if signal.signal_type == SignalType.SELL:
        return -1
    return 0


def micro_confirmation_snapshot(
    *,
    session: TradingSession,
    signal: Signal,
    current_bar_index: int,
    signal_bar_index: int,
    required_bars: int,
    mode: str = "consecutive_close",
    volume_delta_min_pct: float = 0.60,
) -> Dict[str, Any]:
    """
    Evaluate micro confirmation for pending entries.

    Modes:
    - "consecutive_close": require N consecutive bars closing in signal direction
    - "volume_delta": require 1 closed bar with directional volume delta >= threshold
    - "disabled": bypass confirmation entirely

    Uses only bars closed strictly before current bar open:
    - closed bar window: [signal_bar_index + 1, current_bar_index - 1]
    """
    direction = signal_direction(signal)
    last_closed_index = current_bar_index - 1
    confirmation_start = max(0, int(signal_bar_index) + 1)
    available_count = max(0, last_closed_index - confirmation_start + 1)

    base_snapshot: Dict[str, Any] = {
        "enabled": True,
        "mode": mode,
        "signal_bar_index": int(signal_bar_index),
        "last_closed_bar_index": int(last_closed_index),
        "available_bars": int(available_count),
    }

    if mode == "disabled":
        base_snapshot.update(
            {
                "required_bars": 0,
                "ready": True,
                "passed": True,
                "reason": "micro_confirmation_disabled",
                "confirmation_window_start": 0,
                "confirmation_window_end": 0,
                "close_indexes": [],
                "closes": [],
            }
        )
        return base_snapshot

    if mode == "volume_delta":
        ready = available_count >= 1
        if not ready:
            base_snapshot.update(
                {
                    "required_bars": 1,
                    "ready": False,
                    "passed": False,
                    "reason": "awaiting_micro_confirmation",
                    "confirmation_window_start": int(confirmation_start),
                    "confirmation_window_end": int(last_closed_index),
                    "close_indexes": [],
                    "closes": [],
                    "volume_delta_min_pct": volume_delta_min_pct,
                }
            )
            return base_snapshot

        bar = session.bars[last_closed_index] if last_closed_index < len(session.bars) else None
        buy_vol = float(getattr(bar, "l2_buy_volume", 0) or 0) if bar else 0.0
        sell_vol = float(getattr(bar, "l2_sell_volume", 0) or 0) if bar else 0.0
        total_vol = buy_vol + sell_vol
        if total_vol > 0:
            directional_ratio = (buy_vol / total_vol) if direction > 0 else (sell_vol / total_vol)
        else:
            bar_close = float(getattr(bar, "close", 0) or 0) if bar else 0.0
            bar_open = float(getattr(bar, "open", 0) or 0) if bar else 0.0
            close_in_direction = (bar_close > bar_open) if direction > 0 else (bar_close < bar_open)
            directional_ratio = 1.0 if close_in_direction else 0.0

        passed = directional_ratio >= volume_delta_min_pct
        reason = "micro_confirmation_passed" if passed else "micro_confirmation_failed"

        base_snapshot.update(
            {
                "required_bars": 1,
                "ready": True,
                "passed": bool(passed),
                "reason": reason,
                "confirmation_window_start": int(last_closed_index),
                "confirmation_window_end": int(last_closed_index),
                "close_indexes": [last_closed_index],
                "closes": [float(getattr(bar, "close", 0) or 0)] if bar else [],
                "volume_delta_min_pct": volume_delta_min_pct,
                "volume_delta_actual": round(directional_ratio, 4),
                "l2_buy_volume": round(buy_vol, 2),
                "l2_sell_volume": round(sell_vol, 2),
            }
        )
        return base_snapshot

    ready = available_count >= required_bars
    window_end = last_closed_index
    window_start = max(confirmation_start, window_end - required_bars + 1)
    closes: List[float] = []
    close_indexes: List[int] = []
    for idx in range(window_start, window_end + 1):
        if idx < 0 or idx >= len(session.bars):
            continue
        closes.append(float(getattr(session.bars[idx], "close", 0.0) or 0.0))
        close_indexes.append(idx)

    passed = False
    if ready and len(closes) >= required_bars and direction != 0:
        trend_up = True
        trend_down = True
        prev_close = None
        for value in closes:
            if prev_close is not None:
                if value <= prev_close:
                    trend_up = False
                if value >= prev_close:
                    trend_down = False
            prev_close = value
        passed = trend_up if direction > 0 else trend_down

    if not ready:
        reason = "awaiting_micro_confirmation"
    elif passed:
        reason = "micro_confirmation_passed"
    else:
        reason = "micro_confirmation_failed"

    base_snapshot.update(
        {
            "required_bars": int(required_bars),
            "ready": bool(ready),
            "passed": bool(passed),
            "reason": reason,
            "confirmation_window_start": int(window_start),
            "confirmation_window_end": int(window_end),
            "close_indexes": close_indexes,
            "closes": closes,
        }
    )
    return base_snapshot


def calculate_intrabar_1s_snapshot(
    bar: Optional[BarData],
    *,
    intrabar_window_seconds: int = 5,
) -> Dict[str, Any]:
    """
    Summarize the current bar's optional 1-second quote path.

    Uses only quote rows attached to the currently processed minute,
    preserving no-lookahead behavior.
    """
    intrabar_window_seconds = max(1, min(60, int(intrabar_window_seconds)))
    default_snapshot: Dict[str, Any] = {
        "has_intrabar_coverage": False,
        "coverage_points": 0,
        "mid_move_pct": 0.0,
        "push_ratio": 0.0,
        "directional_consistency": 0.0,
        "micro_volatility_bps": 0.0,
        "spread_bps_avg": 0.0,
        "window_eval_seconds": intrabar_window_seconds,
        "window_long_move_pct": 0.0,
        "window_long_push_ratio": 0.0,
        "window_long_directional_consistency": 0.0,
        "window_short_move_pct": 0.0,
        "window_short_push_ratio": 0.0,
        "window_short_directional_consistency": 0.0,
        "window_max_coverage_points": 0,
    }
    if bar is None:
        return dict(default_snapshot)

    raw_quotes = getattr(bar, "intrabar_quotes_1s", None)
    if not isinstance(raw_quotes, list) or not raw_quotes:
        return dict(default_snapshot)

    rows: List[tuple[int, float, float]] = []
    for item in raw_quotes:
        if not isinstance(item, dict):
            continue
        try:
            second = int(float(item.get("s", 0) or 0))
            bid = float(item.get("bid", 0.0) or 0.0)
            ask = float(item.get("ask", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if second < 0 or second > 59:
            continue
        if bid <= 0.0 and ask <= 0.0:
            continue
        if bid > 0.0 and ask > 0.0:
            mid = (bid + ask) / 2.0
            spread_bps = ((ask - bid) / mid * 10000.0) if mid > 0 else 0.0
        else:
            mid = ask if ask > 0.0 else bid
            spread_bps = 0.0
        if mid <= 0.0:
            continue
        rows.append((second, mid, spread_bps))

    if len(rows) < 2:
        snapshot = dict(default_snapshot)
        snapshot["has_intrabar_coverage"] = len(rows) > 0
        snapshot["coverage_points"] = len(rows)
        return snapshot

    rows.sort(key=lambda entry: entry[0])
    mids = [entry[1] for entry in rows]
    spreads = [entry[2] for entry in rows]
    first_mid = mids[0]
    last_mid = mids[-1]

    abs_moves: List[float] = []
    signed_steps: List[float] = []
    for idx in range(1, len(mids)):
        step = mids[idx] - mids[idx - 1]
        signed_steps.append(step)
        abs_moves.append(abs(step))

    total_abs_move = sum(abs_moves)
    net_move = last_mid - first_mid
    push_hits = sum(1 for step in signed_steps if step > 0)
    pull_hits = sum(1 for step in signed_steps if step < 0)
    directional_base = push_hits + pull_hits

    push_ratio = (
        (push_hits - pull_hits) / directional_base if directional_base > 0 else 0.0
    )
    directional_consistency = (
        abs(net_move) / total_abs_move if total_abs_move > 0.0 else 0.0
    )
    mid_move_pct = ((net_move / first_mid) * 100.0) if first_mid > 0.0 else 0.0

    ret_bps: List[float] = []
    for idx in range(1, len(mids)):
        prev_mid = mids[idx - 1]
        if prev_mid <= 0.0:
            continue
        ret_bps.append(((mids[idx] - prev_mid) / prev_mid) * 10000.0)
    if ret_bps:
        mean_ret = sum(ret_bps) / len(ret_bps)
        variance = sum((ret - mean_ret) ** 2 for ret in ret_bps) / len(ret_bps)
        micro_volatility_bps = math.sqrt(variance)
    else:
        micro_volatility_bps = 0.0

    window_long_move_pct = 0.0
    window_long_push_ratio = 0.0
    window_long_directional_consistency = 0.0
    window_short_move_pct = 0.0
    window_short_push_ratio = 0.0
    window_short_directional_consistency = 0.0
    window_max_coverage_points = 0
    window_buckets: Dict[int, List[tuple[int, float]]] = {}
    for second, mid, _spread in rows:
        bucket_id = int(second // intrabar_window_seconds)
        window_buckets.setdefault(bucket_id, []).append((second, mid))

    for bucket_rows in window_buckets.values():
        if len(bucket_rows) < 2:
            continue
        bucket_rows.sort(key=lambda entry: entry[0])
        bucket_mids = [entry[1] for entry in bucket_rows]
        bucket_steps: List[float] = []
        bucket_abs_steps: List[float] = []
        for idx in range(1, len(bucket_mids)):
            step = bucket_mids[idx] - bucket_mids[idx - 1]
            bucket_steps.append(step)
            bucket_abs_steps.append(abs(step))
        if not bucket_steps:
            continue

        first_bucket_mid = bucket_mids[0]
        if first_bucket_mid <= 0.0:
            continue
        net_bucket_move = bucket_mids[-1] - first_bucket_mid
        total_bucket_abs_move = sum(bucket_abs_steps)
        bucket_push_hits = sum(1 for step in bucket_steps if step > 0)
        bucket_pull_hits = sum(1 for step in bucket_steps if step < 0)
        bucket_directional_base = bucket_push_hits + bucket_pull_hits
        bucket_push_ratio = (
            (bucket_push_hits - bucket_pull_hits) / bucket_directional_base
            if bucket_directional_base > 0
            else 0.0
        )
        bucket_directional_consistency = (
            abs(net_bucket_move) / total_bucket_abs_move
            if total_bucket_abs_move > 0.0
            else 0.0
        )
        bucket_move_pct = (net_bucket_move / first_bucket_mid) * 100.0
        bucket_points = len(bucket_rows)
        window_max_coverage_points = max(window_max_coverage_points, bucket_points)

        if bucket_move_pct > window_long_move_pct:
            window_long_move_pct = bucket_move_pct
            window_long_push_ratio = bucket_push_ratio
            window_long_directional_consistency = bucket_directional_consistency
        if bucket_move_pct < window_short_move_pct:
            window_short_move_pct = bucket_move_pct
            window_short_push_ratio = bucket_push_ratio
            window_short_directional_consistency = bucket_directional_consistency

    return {
        "has_intrabar_coverage": True,
        "coverage_points": len(rows),
        "mid_move_pct": mid_move_pct,
        "push_ratio": push_ratio,
        "directional_consistency": directional_consistency,
        "micro_volatility_bps": micro_volatility_bps,
        "spread_bps_avg": (sum(spreads) / len(spreads)) if spreads else 0.0,
        "window_eval_seconds": intrabar_window_seconds,
        "window_long_move_pct": window_long_move_pct,
        "window_long_push_ratio": window_long_push_ratio,
        "window_long_directional_consistency": window_long_directional_consistency,
        "window_short_move_pct": window_short_move_pct,
        "window_short_push_ratio": window_short_push_ratio,
        "window_short_directional_consistency": window_short_directional_consistency,
        "window_max_coverage_points": window_max_coverage_points,
    }


def intrabar_confirmation_snapshot(
    *,
    session: TradingSession,
    signal: Signal,
    current_bar_index: int,
    signal_bar_index: int,
    window_seconds: int,
    min_coverage_points: int,
    min_move_pct: float,
    min_push_ratio: float,
    max_spread_bps: float,
) -> Dict[str, Any]:
    direction = signal_direction(signal)
    eval_bar_index = current_bar_index - 1
    if eval_bar_index < 0 or eval_bar_index >= len(session.bars):
        return {
            "enabled": True,
            "ready": False,
            "passed": False,
            "reason": "awaiting_closed_bar",
            "signal_bar_index": int(signal_bar_index),
            "evaluation_bar_index": int(eval_bar_index),
        }

    eval_bar = session.bars[eval_bar_index]
    snapshot = calculate_intrabar_1s_snapshot(
        eval_bar,
        intrabar_window_seconds=window_seconds,
    )
    has_coverage = bool(snapshot.get("has_intrabar_coverage", False))
    coverage_points = int(
        snapshot.get("window_max_coverage_points") or snapshot.get("coverage_points", 0) or 0
    )
    spread_bps_avg = float(snapshot.get("spread_bps_avg", 0.0) or 0.0)

    coverage_ok = bool(has_coverage and coverage_points >= max(0, int(min_coverage_points)))
    spread_ok = spread_bps_avg <= max(0.0, float(max_spread_bps))

    if direction > 0:
        move_value = float(
            max(
                snapshot.get("window_long_move_pct", 0.0) or 0.0,
                snapshot.get("mid_move_pct", 0.0) or 0.0,
            )
        )
        push_value = float(
            max(
                snapshot.get("window_long_push_ratio", 0.0) or 0.0,
                snapshot.get("push_ratio", 0.0) or 0.0,
            )
        )
    elif direction < 0:
        move_value = abs(
            float(
                min(
                    snapshot.get("window_short_move_pct", 0.0) or 0.0,
                    snapshot.get("mid_move_pct", 0.0) or 0.0,
                )
            )
        )
        push_value = abs(
            float(
                min(
                    snapshot.get("window_short_push_ratio", 0.0) or 0.0,
                    snapshot.get("push_ratio", 0.0) or 0.0,
                )
            )
        )
    else:
        move_value = 0.0
        push_value = 0.0

    move_ok = move_value >= max(0.0, float(min_move_pct))
    push_ok = push_value >= max(0.0, float(min_push_ratio))
    ready = has_coverage
    passed = bool(ready and direction != 0 and coverage_ok and move_ok and push_ok and spread_ok)

    if not ready:
        reason = "awaiting_intrabar_coverage"
    elif passed:
        reason = "intrabar_confirmation_passed"
    elif not coverage_ok:
        reason = "intrabar_coverage_below_threshold"
    elif not move_ok:
        reason = "intrabar_move_below_threshold"
    elif not push_ok:
        reason = "intrabar_push_below_threshold"
    elif not spread_ok:
        reason = "intrabar_spread_above_threshold"
    else:
        reason = "intrabar_confirmation_failed"

    return {
        "enabled": True,
        "ready": bool(ready),
        "passed": bool(passed),
        "reason": reason,
        "signal_bar_index": int(signal_bar_index),
        "evaluation_bar_index": int(eval_bar_index),
        "window_seconds": int(window_seconds),
        "coverage_points": int(coverage_points),
        "move_pct_abs": float(move_value),
        "push_ratio_abs": float(push_value),
        "spread_bps_avg": float(spread_bps_avg),
        "thresholds": {
            "min_coverage_points": int(max(0, int(min_coverage_points))),
            "min_move_pct": float(max(0.0, float(min_move_pct))),
            "min_push_ratio": float(max(0.0, float(min_push_ratio))),
            "max_spread_bps": float(max(0.0, float(max_spread_bps))),
        },
    }
