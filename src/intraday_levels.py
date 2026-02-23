"""Intraday support/resistance and volume-profile tracker (session-scoped)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional



from .intraday_levels_helpers.common import to_float as _to_float
from .intraday_levels_helpers.snapshot import (
    build_intraday_levels_snapshot,
    intraday_levels_indicator_payload,
)
from .intraday_levels_helpers.state import ensure_intraday_levels_state

def _level_tolerance(level_price: float, pct: float) -> float:
    return max(0.01, abs(level_price) * max(0.0, pct) / 100.0)


def _trim_sequence(values: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    if len(values) <= limit:
        return values
    return values[-limit:]


def _append_event(state: Dict[str, Any], event: Dict[str, Any]) -> None:
    state["recent_events"].append(event)
    max_events = int(state["config"].get("max_recent_events", 40))
    state["recent_events"] = _trim_sequence(state["recent_events"], max_events)


def _append_swing_point(
    state: Dict[str, Any],
    *,
    bucket: str,
    price: float,
    bar_index: int,
    timestamp_iso: str,
) -> None:
    points = state[bucket]
    points.append(
        {
            "price": round(price, 4),
            "bar_index": int(bar_index),
            "timestamp": timestamp_iso,
        }
    )
    max_points = int(state["config"].get("max_swing_points", 30))
    state[bucket] = _trim_sequence(points, max_points)


def _register_level_from_swing(
    state: Dict[str, Any],
    *,
    kind: str,
    price: float,
    bar_index: int,
    timestamp_iso: str,
    source: str,
) -> None:
    if price <= 0.0:
        return

    merge_pct = float(state["config"].get("merge_tolerance_pct", 0.12))
    level_to_merge: Optional[Dict[str, Any]] = None
    smallest_distance = float("inf")
    for level in state["levels"]:
        if level.get("kind") != kind or bool(level.get("broken", False)):
            continue
        existing_price = _to_float(level.get("price"), 0.0)
        tol = _level_tolerance(existing_price or price, merge_pct)
        distance = abs(existing_price - price)
        if distance <= tol and distance < smallest_distance:
            smallest_distance = distance
            level_to_merge = level

    if level_to_merge is not None:
        samples = int(level_to_merge.get("swing_samples", 1))
        merged_price = (
            (_to_float(level_to_merge.get("price"), price) * samples + price) / (samples + 1)
        )
        level_to_merge["price"] = round(merged_price, 4)
        level_to_merge["swing_samples"] = samples + 1
        level_to_merge["last_swing_bar_index"] = int(bar_index)
        level_to_merge["last_swing_timestamp"] = timestamp_iso
        return

    level_id = int(state.get("next_level_id", 1))
    state["next_level_id"] = level_id + 1
    state["levels"].append(
        {
            "id": level_id,
            "kind": kind,
            "price": round(price, 4),
            "source": source,
            "created_bar_index": int(bar_index),
            "created_timestamp": timestamp_iso,
            "last_swing_bar_index": int(bar_index),
            "last_swing_timestamp": timestamp_iso,
            "swing_samples": 1,
            "tests": 0,
            "last_test_bar_index": -1,
            "last_event": None,
            "last_event_bar_index": -1,
            "broken": False,
            "broken_bar_index": -1,
        }
    )


def _detect_and_register_swings(
    state: Dict[str, Any],
    *,
    bars: List[Any],
    current_bar_index: int,
) -> None:
    left = int(state["config"].get("swing_left_bars", 2))
    right = int(state["config"].get("swing_right_bars", 2))
    pivot_index = current_bar_index - right
    if pivot_index < 0 or pivot_index < left:
        return

    start = pivot_index - left
    end = pivot_index + right
    if start < 0 or end >= len(bars):
        return

    window = bars[start:end + 1]
    if len(window) != (left + right + 1):
        return

    pivot_bar = bars[pivot_index]
    pivot_high = _to_float(getattr(pivot_bar, "high", 0.0), 0.0)
    pivot_low = _to_float(getattr(pivot_bar, "low", 0.0), 0.0)
    highs = [_to_float(getattr(bar, "high", 0.0), 0.0) for bar in window]
    lows = [_to_float(getattr(bar, "low", 0.0), 0.0) for bar in window]
    timestamp_iso = (
        getattr(getattr(pivot_bar, "timestamp", None), "isoformat", lambda: "")() or ""
    )

    if pivot_high > 0.0 and highs.count(max(highs)) == 1 and pivot_high == max(highs):
        _append_swing_point(
            state,
            bucket="swing_highs",
            price=pivot_high,
            bar_index=pivot_index,
            timestamp_iso=timestamp_iso,
        )
        _register_level_from_swing(
            state,
            kind="resistance",
            price=pivot_high,
            bar_index=pivot_index,
            timestamp_iso=timestamp_iso,
            source="swing_high",
        )

    if pivot_low > 0.0 and lows.count(min(lows)) == 1 and pivot_low == min(lows):
        _append_swing_point(
            state,
            bucket="swing_lows",
            price=pivot_low,
            bar_index=pivot_index,
            timestamp_iso=timestamp_iso,
        )
        _register_level_from_swing(
            state,
            kind="support",
            price=pivot_low,
            bar_index=pivot_index,
            timestamp_iso=timestamp_iso,
            source="swing_low",
        )


def _detect_and_register_spike_level(
    state: Dict[str, Any],
    *,
    bar: Any,
    current_bar_index: int,
) -> None:
    cfg = state["config"]
    if not bool(cfg.get("spike_detection_enabled", True)):
        return

    bar_high = _to_float(getattr(bar, "high", 0.0), 0.0)
    bar_low = _to_float(getattr(bar, "low", 0.0), 0.0)
    bar_open = _to_float(getattr(bar, "open", 0.0), 0.0)
    bar_close = _to_float(getattr(bar, "close", 0.0), 0.0)
    full_range = max(0.0, bar_high - bar_low)
    if full_range <= 0.0:
        return

    upper_wick = max(0.0, bar_high - max(bar_open, bar_close))
    lower_wick = max(0.0, min(bar_open, bar_close) - bar_low)
    max_wick = max(upper_wick, lower_wick)
    wick_ratio = max_wick / full_range if full_range > 0.0 else 0.0
    min_ratio = min(0.95, max(0.4, _to_float(cfg.get("spike_min_wick_ratio"), 0.60)))
    if wick_ratio < min_ratio:
        return

    timestamp_iso = getattr(getattr(bar, "timestamp", None), "isoformat", lambda: "")() or ""
    if upper_wick >= lower_wick and bar_high > 0.0:
        _register_level_from_swing(
            state,
            kind="resistance",
            price=bar_high,
            bar_index=current_bar_index,
            timestamp_iso=timestamp_iso,
            source="spike_high",
        )
    elif bar_low > 0.0:
        _register_level_from_swing(
            state,
            kind="support",
            price=bar_low,
            bar_index=current_bar_index,
            timestamp_iso=timestamp_iso,
            source="spike_low",
        )


def _update_gap_context(
    state: Dict[str, Any],
    *,
    bars: List[Any],
    bar: Any,
    current_bar_index: int,
) -> None:
    gap_ctx = (
        dict(state.get("gap_context", {}))
        if isinstance(state.get("gap_context"), dict)
        else {}
    )
    cfg = state["config"]
    enabled = bool(cfg.get("gap_analysis_enabled", True))
    gap_ctx["enabled"] = enabled
    if not enabled:
        state["gap_context"] = gap_ctx
        return

    if bool(gap_ctx.get("initialized", False)):
        state["gap_context"] = gap_ctx
        return
    if current_bar_index != 0:
        state["gap_context"] = gap_ctx
        return

    open_price = _to_float(getattr(bar, "open", 0.0), 0.0)
    prev_close = None
    memory_profiles = (
        list(state.get("memory_profiles", []))
        if isinstance(state.get("memory_profiles"), list)
        else []
    )
    if memory_profiles:
        last_profile = memory_profiles[-1]
        if isinstance(last_profile, dict):
            prev_close = _to_float(last_profile.get("close_price"), 0.0)

    if (prev_close is None or prev_close <= 0.0) and len(bars) > 1:
        prev_close = _to_float(getattr(bars[0], "close", 0.0), 0.0)
    prev_close = prev_close if prev_close and prev_close > 0.0 else None

    gap_pct = (
        ((open_price - prev_close) / prev_close * 100.0)
        if (open_price > 0.0 and prev_close is not None and prev_close > 0.0)
        else 0.0
    )
    gap_direction = "up" if gap_pct > 0.0 else "down" if gap_pct < 0.0 else "none"
    abs_gap = abs(gap_pct)
    min_gap = max(0.0, _to_float(cfg.get("gap_min_pct"), 0.30))
    momentum_threshold = max(0.1, _to_float(cfg.get("gap_momentum_threshold_pct"), 2.0))

    favor_side = "none"
    momentum_bias_side = "none"
    gap_fill_target = prev_close
    if abs_gap >= min_gap:
        if gap_pct > 0.0:
            favor_side = "short"
        elif gap_pct < 0.0:
            favor_side = "long"
        if abs_gap >= momentum_threshold:
            momentum_bias_side = "long" if gap_pct > 0.0 else "short"

        if prev_close is not None and prev_close > 0.0:
            timestamp_iso = getattr(getattr(bar, "timestamp", None), "isoformat", lambda: "")() or ""
            if prev_close < open_price:
                _register_level_from_swing(
                    state,
                    kind="support",
                    price=prev_close,
                    bar_index=current_bar_index,
                    timestamp_iso=timestamp_iso,
                    source="gap_fill_target",
                )
            elif prev_close > open_price:
                _register_level_from_swing(
                    state,
                    kind="resistance",
                    price=prev_close,
                    bar_index=current_bar_index,
                    timestamp_iso=timestamp_iso,
                    source="gap_fill_target",
                )

    gap_ctx.update(
        {
            "initialized": True,
            "open_price": round(open_price, 4) if open_price > 0.0 else None,
            "prev_close": round(prev_close, 4) if prev_close else None,
            "gap_pct": round(gap_pct, 4),
            "gap_direction": gap_direction,
            "gap_fill_target": round(gap_fill_target, 4) if gap_fill_target else None,
            "favor_side": favor_side,
            "momentum_bias_side": momentum_bias_side,
        }
    )
    state["gap_context"] = gap_ctx


def _true_range(current_bar: Any, prev_close: Optional[float]) -> float:
    high = _to_float(getattr(current_bar, "high", 0.0), 0.0)
    low = _to_float(getattr(current_bar, "low", 0.0), 0.0)
    if high <= 0.0 and low <= 0.0:
        return 0.0
    hl = max(0.0, high - low)
    if prev_close is None or prev_close <= 0.0:
        return hl
    hc = abs(high - prev_close)
    lc = abs(low - prev_close)
    return max(hl, hc, lc)


def _update_market_activity(
    state: Dict[str, Any],
    *,
    bars: List[Any],
    current_bar_index: int,
) -> None:
    cfg = state["config"]
    lookback = max(5, int(cfg.get("rvol_lookback_bars", 20)))
    volumes: List[float] = []
    start = max(0, current_bar_index - lookback)
    for idx in range(start, current_bar_index):
        volumes.append(max(0.0, _to_float(getattr(bars[idx], "volume", 0.0), 0.0)))
    current_volume = max(
        0.0,
        _to_float(getattr(bars[current_bar_index], "volume", 0.0), 0.0),
    )
    avg_volume = (sum(volumes) / len(volumes)) if volumes else None
    rvol = (
        (current_volume / avg_volume)
        if (avg_volume is not None and avg_volume > 0.0)
        else None
    )

    market_activity = (
        dict(state.get("market_activity", {}))
        if isinstance(state.get("market_activity"), dict)
        else {}
    )
    market_activity["rvol"] = round(float(rvol), 6) if rvol is not None else None
    market_activity["avg_volume_lookback"] = (
        round(float(avg_volume), 4) if avg_volume is not None else None
    )
    market_activity["rvol_lookback_bars"] = lookback
    state["market_activity"] = market_activity

    adaptive = (
        dict(state.get("adaptive_window", {}))
        if isinstance(state.get("adaptive_window"), dict)
        else {}
    )
    adaptive["enabled"] = bool(cfg.get("adaptive_window_enabled", True))
    if not adaptive.get("enabled", True):
        state["adaptive_window"] = adaptive
        return

    min_bars = max(1, int(cfg.get("adaptive_window_min_bars", 6)))
    if bool(adaptive.get("ready", False)):
        state["adaptive_window"] = adaptive
        return
    if current_bar_index < (min_bars - 1):
        adaptive["rvol"] = round(float(rvol), 6) if rvol is not None else None
        adaptive["atr_ratio"] = None
        state["adaptive_window"] = adaptive
        return

    tr_values: List[float] = []
    for idx in range(0, current_bar_index + 1):
        prev_close = (
            _to_float(getattr(bars[idx - 1], "close", 0.0), 0.0)
            if idx > 0
            else None
        )
        tr = _true_range(bars[idx], prev_close)
        if tr > 0.0:
            tr_values.append(tr)
    atr_5 = (sum(tr_values[-5:]) / min(5, len(tr_values))) if tr_values else 0.0
    atr_20 = (sum(tr_values[-20:]) / min(20, len(tr_values))) if tr_values else 0.0
    atr_ratio = (atr_5 / atr_20) if atr_20 > 0.0 else None

    rvol_threshold = max(0.0, _to_float(cfg.get("adaptive_window_rvol_threshold"), 1.0))
    atr_ratio_max = max(0.1, _to_float(cfg.get("adaptive_window_atr_ratio_max"), 1.5))
    ready = bool(
        (rvol is not None and rvol >= rvol_threshold)
        and (atr_ratio is not None and atr_ratio <= atr_ratio_max)
    )
    adaptive["rvol"] = round(float(rvol), 6) if rvol is not None else None
    adaptive["atr_ratio"] = round(float(atr_ratio), 6) if atr_ratio is not None else None
    if ready:
        adaptive["ready"] = True
        adaptive["ready_bar_index"] = int(current_bar_index)
    state["adaptive_window"] = adaptive


def _update_volume_profile(
    state: Dict[str, Any],
    *,
    bar: Any,
) -> None:
    volume = max(0.0, _to_float(getattr(bar, "volume", 0.0), 0.0))
    if volume <= 0.0:
        return

    cfg = state["config"]
    bin_size = _to_float(state.get("volume_profile_bin_size"), 0.0)
    if bin_size <= 0.0:
        reference_price = max(
            0.01,
            _to_float(getattr(bar, "close", 0.0), 0.01),
        )
        bin_size_pct = max(0.01, _to_float(cfg.get("volume_profile_bin_size_pct"), 0.05))
        bin_size = max(0.01, (reference_price * bin_size_pct) / 100.0)
        state["volume_profile_bin_size"] = bin_size

    bar_vwap = _to_float(getattr(bar, "vwap", None), 0.0)
    if bar_vwap > 0.0:
        profile_price = bar_vwap
    else:
        high = _to_float(getattr(bar, "high", 0.0), 0.0)
        low = _to_float(getattr(bar, "low", 0.0), 0.0)
        close = _to_float(getattr(bar, "close", 0.0), 0.0)
        profile_price = (high + low + close) / 3.0 if close > 0.0 else close

    if profile_price <= 0.0:
        return

    bin_index = int(round(profile_price / bin_size))
    bins = state["volume_profile_bins"]
    key = str(bin_index)
    bins[key] = _to_float(bins.get(key), 0.0) + volume

    total_volume = sum(_to_float(v, 0.0) for v in bins.values())
    if total_volume <= 0.0:
        return

    int_bins = {int(k): _to_float(v, 0.0) for k, v in bins.items()}
    poc_index = max(int_bins, key=lambda idx: int_bins[idx])
    poc_volume = int_bins[poc_index]
    target_coverage = max(0.5, min(0.95, _to_float(cfg.get("value_area_pct"), 0.70)))

    selected = {poc_index}
    cumulative_volume = poc_volume
    left_idx = poc_index - 1
    right_idx = poc_index + 1
    while (cumulative_volume / total_volume) < target_coverage:
        left_vol = int_bins.get(left_idx, 0.0)
        right_vol = int_bins.get(right_idx, 0.0)
        if left_vol <= 0.0 and right_vol <= 0.0:
            break
        if right_vol > left_vol:
            selected.add(right_idx)
            cumulative_volume += right_vol
            right_idx += 1
        else:
            selected.add(left_idx)
            cumulative_volume += left_vol
            left_idx -= 1

    value_area_low = min(selected) * bin_size if selected else poc_index * bin_size
    value_area_high = max(selected) * bin_size if selected else poc_index * bin_size

    state["volume_profile"] = {
        "bin_size": round(bin_size, 6),
        "total_volume": round(total_volume, 2),
        "poc_price": round(poc_index * bin_size, 4),
        "poc_volume": round(poc_volume, 2),
        "value_area_low": round(value_area_low, 4),
        "value_area_high": round(value_area_high, 4),
        "value_area_coverage": round(
            (cumulative_volume / total_volume) if total_volume > 0.0 else 0.0,
            4,
        ),
        "bins_count": len(int_bins),
    }


def _evaluate_level_interactions(
    state: Dict[str, Any],
    *,
    bars: List[Any],
    current_bar_index: int,
) -> None:
    if not state["levels"]:
        return

    cfg = state["config"]
    bar = bars[current_bar_index]
    timestamp_iso = getattr(getattr(bar, "timestamp", None), "isoformat", lambda: "")() or ""

    bar_open = _to_float(getattr(bar, "open", 0.0), 0.0)
    bar_high = _to_float(getattr(bar, "high", 0.0), 0.0)
    bar_low = _to_float(getattr(bar, "low", 0.0), 0.0)
    bar_close = _to_float(getattr(bar, "close", 0.0), 0.0)
    bar_volume = max(0.0, _to_float(getattr(bar, "volume", 0.0), 0.0))

    lookback = max(2, int(cfg.get("breakout_volume_lookback", 20)))
    volume_window = [
        max(0.0, _to_float(getattr(prev_bar, "volume", 0.0), 0.0))
        for prev_bar in bars[max(0, current_bar_index - lookback):current_bar_index]
    ]
    base_volume = (sum(volume_window) / len(volume_window)) if volume_window else 0.0
    breakout_multiplier = max(1.0, _to_float(cfg.get("breakout_volume_multiplier"), 1.2))
    min_retest_bars = max(1, int(cfg.get("min_retest_bars", 2)))

    for level in state["levels"]:
        if bool(level.get("broken", False)):
            continue
        level_price = _to_float(level.get("price"), 0.0)
        if level_price <= 0.0:
            continue

        kind = str(level.get("kind", "")).lower()
        test_tol = _level_tolerance(level_price, _to_float(cfg.get("test_tolerance_pct"), 0.08))
        break_tol = _level_tolerance(level_price, _to_float(cfg.get("break_tolerance_pct"), 0.05))
        last_test = int(level.get("last_test_bar_index", -1))
        is_touched = (bar_low <= (level_price + test_tol)) and (bar_high >= (level_price - test_tol))

        if is_touched and (current_bar_index - last_test) >= min_retest_bars:
            level["tests"] = int(level.get("tests", 0)) + 1
            level["last_test_bar_index"] = int(current_bar_index)

            bounce_direction: Optional[str] = None
            if kind == "support" and bar_close > (level_price + break_tol) and bar_close >= bar_open:
                bounce_direction = "bullish"
            elif kind == "resistance" and bar_close < (level_price - break_tol) and bar_close <= bar_open:
                bounce_direction = "bearish"

            if bounce_direction:
                level["last_event"] = "bounce"
                level["last_event_bar_index"] = int(current_bar_index)
                _append_event(
                    state,
                    {
                        "event_type": "bounce",
                        "direction": bounce_direction,
                        "level_id": int(level.get("id", -1)),
                        "level_kind": kind,
                        "price": round(level_price, 4),
                        "bar_index": int(current_bar_index),
                        "timestamp": timestamp_iso,
                        "volume_confirmed": None,
                        "tests": int(level.get("tests", 0)),
                    },
                )

        break_direction: Optional[str] = None
        if kind == "support" and bar_close < (level_price - break_tol):
            break_direction = "bearish"
        elif kind == "resistance" and bar_close > (level_price + break_tol):
            break_direction = "bullish"

        if not break_direction:
            continue

        volume_confirmed = base_volume > 0.0 and bar_volume >= (base_volume * breakout_multiplier)
        if not volume_confirmed:
            continue

        level["broken"] = True
        level["broken_bar_index"] = int(current_bar_index)
        level["last_event"] = "break"
        level["last_event_bar_index"] = int(current_bar_index)
        _append_event(
            state,
            {
                "event_type": "break",
                "direction": break_direction,
                "level_id": int(level.get("id", -1)),
                "level_kind": kind,
                "price": round(level_price, 4),
                "bar_index": int(current_bar_index),
                "timestamp": timestamp_iso,
                "volume_confirmed": True,
                "tests": int(level.get("tests", 0)),
            },
        )


def _prune_levels(state: Dict[str, Any]) -> None:
    max_levels = max(4, int(state["config"].get("max_levels", 24)))
    levels = list(state["levels"])
    if len(levels) <= max_levels:
        return

    levels.sort(
        key=lambda item: (
            bool(item.get("broken", False)),
            -int(item.get("tests", 0)),
            -int(item.get("last_swing_bar_index", -1)),
            -int(item.get("created_bar_index", -1)),
        )
    )
    state["levels"] = levels[:max_levels]


def _update_opening_range(
    state: Dict[str, Any],
    *,
    bar: Any,
    current_bar_index: int,
) -> None:
    cfg = state["config"]
    opening = state.get("opening_range", {}) if isinstance(state.get("opening_range"), dict) else {}
    opening_enabled = bool(cfg.get("opening_range_enabled", True))
    opening["enabled"] = opening_enabled
    if not opening_enabled:
        state["opening_range"] = opening
        return

    bars_target = max(5, int(cfg.get("opening_range_minutes", 30)))
    opening["bars_target"] = bars_target
    opening["bars_collected"] = max(int(opening.get("bars_collected", 0)), current_bar_index + 1)

    bar_high = _to_float(getattr(bar, "high", 0.0), 0.0)
    bar_low = _to_float(getattr(bar, "low", 0.0), 0.0)
    if bar_high > 0.0:
        prev_high = _to_float(opening.get("high"), bar_high)
        opening["high"] = round(max(prev_high, bar_high), 4)
    if bar_low > 0.0:
        prev_low = _to_float(opening.get("low"), bar_low)
        opening["low"] = round(min(prev_low, bar_low), 4)

    if (
        not bool(opening.get("complete", False))
        and current_bar_index >= (bars_target - 1)
        and _to_float(opening.get("high"), 0.0) > 0.0
        and _to_float(opening.get("low"), 0.0) > 0.0
    ):
        opening["complete"] = True
        opening["mid"] = round(
            (_to_float(opening.get("high"), 0.0) + _to_float(opening.get("low"), 0.0)) / 2.0,
            4,
        )
        _register_level_from_swing(
            state,
            kind="resistance",
            price=_to_float(opening.get("high"), 0.0),
            bar_index=current_bar_index,
            timestamp_iso=getattr(getattr(bar, "timestamp", None), "isoformat", lambda: "")() or "",
            source="opening_range_high",
        )
        _register_level_from_swing(
            state,
            kind="support",
            price=_to_float(opening.get("low"), 0.0),
            bar_index=current_bar_index,
            timestamp_iso=getattr(getattr(bar, "timestamp", None), "isoformat", lambda: "")() or "",
            source="opening_range_low",
        )
        _append_event(
            state,
            {
                "event_type": "opening_range_complete",
                "direction": None,
                "level_id": None,
                "level_kind": "opening_range",
                "price": opening.get("mid"),
                "bar_index": int(current_bar_index),
                "timestamp": getattr(getattr(bar, "timestamp", None), "isoformat", lambda: "")() or "",
                "volume_confirmed": None,
                "tests": None,
            },
        )

    if bool(opening.get("complete", False)):
        opening_high = _to_float(opening.get("high"), 0.0)
        opening_low = _to_float(opening.get("low"), 0.0)
        if opening_high > 0.0 and opening_low > 0.0:
            break_tol = _level_tolerance(
                max(opening_high, opening_low),
                _to_float(cfg.get("opening_range_break_tolerance_pct"), 0.05),
            )
            close_px = _to_float(getattr(bar, "close", 0.0), 0.0)
            direction = None
            if close_px > opening_high + break_tol:
                direction = "bullish"
            elif close_px < opening_low - break_tol:
                direction = "bearish"
            if direction and not opening.get("breakout_direction"):
                opening["breakout_direction"] = direction
                opening["breakout_bar_index"] = int(current_bar_index)
                opening["breakout_price"] = round(close_px, 4)
                _append_event(
                    state,
                    {
                        "event_type": "opening_range_break",
                        "direction": direction,
                        "level_id": None,
                        "level_kind": "opening_range",
                        "price": round(close_px, 4),
                        "bar_index": int(current_bar_index),
                        "timestamp": getattr(getattr(bar, "timestamp", None), "isoformat", lambda: "")() or "",
                        "volume_confirmed": None,
                        "tests": None,
                    },
                )

    state["opening_range"] = opening


def _update_poc_migration(
    state: Dict[str, Any],
    *,
    current_bar_index: int,
) -> None:
    cfg = state["config"]
    if not bool(cfg.get("poc_migration_enabled", True)):
        return
    poc_price = _to_float(state.get("volume_profile", {}).get("poc_price"), 0.0)
    if poc_price <= 0.0:
        return

    interval_bars = max(1, int(cfg.get("poc_migration_interval_bars", 30)))
    if current_bar_index % interval_bars != 0 and current_bar_index != int(state.get("last_bar_index", -1)):
        return

    history = list(state.get("poc_history", [])) if isinstance(state.get("poc_history"), list) else []
    if history and int(history[-1].get("bar_index", -1)) == int(current_bar_index):
        return
    history.append(
        {
            "bar_index": int(current_bar_index),
            "poc_price": round(poc_price, 4),
        }
    )
    if len(history) > 128:
        history = history[-128:]
    state["poc_history"] = history


def update_intraday_levels_state(
    state: Optional[Dict[str, Any]],
    *,
    bars: List[Any],
    current_bar_index: int,
) -> Dict[str, Any]:
    normalized = ensure_intraday_levels_state(state)
    tracker_enabled = bool(normalized.get("config", {}).get("enabled", True))
    if current_bar_index < 0 or current_bar_index >= len(bars):
        snapshot = build_intraday_levels_snapshot(normalized)
        normalized["snapshot"] = snapshot
        return normalized
    if not tracker_enabled:
        normalized["last_bar_index"] = int(current_bar_index)
        normalized["snapshot"] = build_intraday_levels_snapshot(normalized)
        return normalized

    bar = bars[current_bar_index]
    _update_gap_context(
        normalized,
        bars=bars,
        bar=bar,
        current_bar_index=current_bar_index,
    )
    _update_market_activity(
        normalized,
        bars=bars,
        current_bar_index=current_bar_index,
    )
    _update_volume_profile(normalized, bar=bar)
    _update_opening_range(
        normalized,
        bar=bar,
        current_bar_index=current_bar_index,
    )
    _detect_and_register_swings(
        normalized,
        bars=bars,
        current_bar_index=current_bar_index,
    )
    _detect_and_register_spike_level(
        normalized,
        bar=bar,
        current_bar_index=current_bar_index,
    )
    _evaluate_level_interactions(
        normalized,
        bars=bars,
        current_bar_index=current_bar_index,
    )
    _prune_levels(normalized)
    _update_poc_migration(
        normalized,
        current_bar_index=current_bar_index,
    )

    normalized["last_bar_index"] = int(current_bar_index)
    normalized["snapshot"] = build_intraday_levels_snapshot(normalized)
    return normalized
