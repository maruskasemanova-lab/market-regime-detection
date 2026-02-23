"""Snapshot builders for intraday levels tracker."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .common import to_float, trim_sequence
from .state import ensure_intraday_levels_state

def _poc_migration_payload(state: Dict[str, Any]) -> Dict[str, Any]:
    cfg = state["config"]
    history = list(state.get("poc_history", [])) if isinstance(state.get("poc_history"), list) else []
    if len(history) < 2:
        return {
            "enabled": bool(cfg.get("poc_migration_enabled", True)),
            "history_points": len(history),
            "change_pct": 0.0,
            "regime_bias": "unknown",
        }

    start = history[0]
    end = history[-1]
    start_price = to_float(start.get("poc_price"), 0.0)
    end_price = to_float(end.get("poc_price"), 0.0)
    change_pct = ((end_price - start_price) / start_price * 100.0) if start_price > 0.0 else 0.0
    trend_threshold = max(0.01, to_float(cfg.get("poc_migration_trend_threshold_pct"), 0.20))
    range_threshold = max(0.01, to_float(cfg.get("poc_migration_range_threshold_pct"), 0.10))

    if abs(change_pct) <= range_threshold:
        regime_bias = "range"
    elif change_pct >= trend_threshold:
        regime_bias = "uptrend"
    elif change_pct <= -trend_threshold:
        regime_bias = "downtrend"
    else:
        regime_bias = "drift"

    return {
        "enabled": bool(cfg.get("poc_migration_enabled", True)),
        "history_points": len(history),
        "start_bar_index": int(start.get("bar_index", -1)),
        "end_bar_index": int(end.get("bar_index", -1)),
        "start_poc_price": round(start_price, 4) if start_price > 0.0 else None,
        "current_poc_price": round(end_price, 4) if end_price > 0.0 else None,
        "change_pct": round(change_pct, 4),
        "regime_bias": regime_bias,
    }



def _build_composite_profile(state: Dict[str, Any]) -> Dict[str, Any]:
    cfg = state["config"]
    if not bool(cfg.get("composite_profile_enabled", True)):
        return {"enabled": False}

    rows: List[Dict[str, Any]] = []
    memory_profiles = list(state.get("memory_profiles", [])) if isinstance(state.get("memory_profiles"), list) else []
    max_days = max(1, int(cfg.get("composite_profile_days", 3)))
    for profile in memory_profiles:
        if not isinstance(profile, dict):
            continue
        poc = to_float(profile.get("poc_price"), 0.0)
        if poc <= 0.0:
            continue
        weight = max(0.1, to_float(profile.get("weight"), 1.0))
        rows.append(
            {
                "source": "memory",
                "weight": weight,
                "poc_price": poc,
                "value_area_low": to_float(profile.get("value_area_low"), 0.0),
                "value_area_high": to_float(profile.get("value_area_high"), 0.0),
            }
        )
    rows = rows[-max_days:]

    current_profile = state.get("volume_profile", {}) if isinstance(state.get("volume_profile"), dict) else {}
    current_poc = to_float(current_profile.get("poc_price"), 0.0)
    if current_poc > 0.0:
        rows.append(
            {
                "source": "current_day",
                "weight": max(0.1, to_float(cfg.get("composite_profile_current_day_weight"), 1.0)),
                "poc_price": current_poc,
                "value_area_low": to_float(current_profile.get("value_area_low"), 0.0),
                "value_area_high": to_float(current_profile.get("value_area_high"), 0.0),
            }
        )

    if not rows:
        return {"enabled": True, "sources": 0}

    total_weight = sum(max(0.1, to_float(row.get("weight"), 1.0)) for row in rows)
    if total_weight <= 0.0:
        return {"enabled": True, "sources": 0}

    def _weighted_avg(key: str) -> Optional[float]:
        weighted_sum = 0.0
        weighted_total = 0.0
        for row in rows:
            value = to_float(row.get(key), 0.0)
            if value <= 0.0:
                continue
            weight = max(0.1, to_float(row.get("weight"), 1.0))
            weighted_sum += value * weight
            weighted_total += weight
        if weighted_total <= 0.0:
            return None
        return weighted_sum / weighted_total

    composite_poc = _weighted_avg("poc_price")
    composite_val = _weighted_avg("value_area_low")
    composite_vah = _weighted_avg("value_area_high")
    return {
        "enabled": True,
        "sources": len(rows),
        "memory_sources": sum(1 for row in rows if row.get("source") == "memory"),
        "current_day_included": any(row.get("source") == "current_day" for row in rows),
        "composite_poc_price": round(composite_poc, 4) if composite_poc is not None else None,
        "composite_value_area_low": round(composite_val, 4) if composite_val is not None else None,
        "composite_value_area_high": round(composite_vah, 4) if composite_vah is not None else None,
        "total_weight": round(total_weight, 4),
    }



def _level_confluence_points(level: Dict[str, Any]) -> int:
    source = str(level.get("source", "")).strip().lower()
    if bool(level.get("memory_level", False)):
        age_days = int(to_float(level.get("memory_age_days"), 0.0))
        return 2 if age_days <= 1 else 3
    if source.startswith("opening_range"):
        return 2
    if source.startswith("prior_day"):
        return 2
    if source.startswith("spike_"):
        return 2
    if source == "gap_fill_target":
        return 2
    return 1



def build_intraday_levels_snapshot(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = ensure_intraday_levels_state(state)
    cfg = normalized["config"]

    levels_limit = max(1, int(cfg.get("snapshot_levels_limit", 10)))
    events_limit = max(1, int(cfg.get("snapshot_events_limit", 8)))
    swings_limit = max(1, int(cfg.get("snapshot_swings_limit", 8)))

    levels = sorted(
        normalized["levels"],
        key=lambda item: (
            bool(item.get("broken", False)),
            -int(item.get("tests", 0)),
            abs(to_float(item.get("price"), 0.0)),
        ),
    )
    level_payload = []
    for level in levels[:levels_limit]:
        level_payload.append(
            {
                "id": int(level.get("id", -1)),
                "kind": str(level.get("kind", "")),
                "price": round(to_float(level.get("price"), 0.0), 4),
                "tests": int(level.get("tests", 0)),
                "swing_samples": int(level.get("swing_samples", 1)),
                "broken": bool(level.get("broken", False)),
                "created_bar_index": int(level.get("created_bar_index", -1)),
                "last_test_bar_index": int(level.get("last_test_bar_index", -1)),
                "last_event": level.get("last_event"),
                "last_event_bar_index": int(level.get("last_event_bar_index", -1)),
                "memory_level": bool(level.get("memory_level", False)),
                "memory_origin_date": (
                    str(level.get("memory_origin_date", ""))
                    if level.get("memory_origin_date") is not None
                    else ""
                ),
                "memory_age_days": (
                    int(to_float(level.get("memory_age_days"), 0.0))
                    if level.get("memory_level", False)
                    else None
                ),
                "memory_weight": (
                    round(to_float(level.get("memory_weight"), 0.0), 4)
                    if level.get("memory_level", False)
                    else None
                ),
                "confluence_points": int(_level_confluence_points(level)),
            }
        )

    recent_events = trim_sequence(list(normalized["recent_events"]), events_limit)
    swing_highs = trim_sequence(list(normalized["swing_highs"]), swings_limit)
    swing_lows = trim_sequence(list(normalized["swing_lows"]), swings_limit)
    profile = dict(normalized.get("volume_profile", {}))
    opening_range = (
        dict(normalized.get("opening_range", {}))
        if isinstance(normalized.get("opening_range"), dict)
        else {}
    )
    gap_context = (
        dict(normalized.get("gap_context", {}))
        if isinstance(normalized.get("gap_context"), dict)
        else {}
    )
    market_activity = (
        dict(normalized.get("market_activity", {}))
        if isinstance(normalized.get("market_activity"), dict)
        else {}
    )
    adaptive_window = (
        dict(normalized.get("adaptive_window", {}))
        if isinstance(normalized.get("adaptive_window"), dict)
        else {}
    )
    poc_migration = _poc_migration_payload(normalized)
    composite_profile = _build_composite_profile(normalized)

    active_levels = sum(1 for lvl in normalized["levels"] if not bool(lvl.get("broken", False)))
    tested_levels = sum(1 for lvl in normalized["levels"] if int(lvl.get("tests", 0)) > 0)
    broken_levels = sum(1 for lvl in normalized["levels"] if bool(lvl.get("broken", False)))
    memory_levels = sum(1 for lvl in normalized["levels"] if bool(lvl.get("memory_level", False)))
    bounce_events = sum(1 for evt in normalized["recent_events"] if evt.get("event_type") == "bounce")
    break_events = sum(1 for evt in normalized["recent_events"] if evt.get("event_type") == "break")

    return {
        "enabled": bool(cfg.get("enabled", True)),
        "bars_processed": max(0, int(normalized.get("last_bar_index", -1)) + 1),
        "levels": level_payload,
        "swing_highs": swing_highs,
        "swing_lows": swing_lows,
        "recent_events": recent_events,
        "latest_event": recent_events[-1] if recent_events else None,
        "volume_profile": {
            "bin_size": to_float(profile.get("bin_size"), 0.0),
            "total_volume": to_float(profile.get("total_volume"), 0.0),
            "poc_price": profile.get("poc_price"),
            "poc_volume": to_float(profile.get("poc_volume"), 0.0),
            "value_area_low": profile.get("value_area_low"),
            "value_area_high": profile.get("value_area_high"),
            "value_area_coverage": to_float(profile.get("value_area_coverage"), 0.0),
            "bins_count": int(to_float(profile.get("bins_count"), 0.0)),
            "composite_profile": composite_profile,
        },
        "opening_range": {
            "enabled": bool(opening_range.get("enabled", True)),
            "bars_target": int(to_float(opening_range.get("bars_target"), 0.0)),
            "bars_collected": int(to_float(opening_range.get("bars_collected"), 0.0)),
            "complete": bool(opening_range.get("complete", False)),
            "high": opening_range.get("high"),
            "low": opening_range.get("low"),
            "mid": opening_range.get("mid"),
            "breakout_direction": opening_range.get("breakout_direction"),
            "breakout_bar_index": int(to_float(opening_range.get("breakout_bar_index"), -1.0)),
            "breakout_price": opening_range.get("breakout_price"),
        },
        "gap_context": {
            "enabled": bool(gap_context.get("enabled", True)),
            "initialized": bool(gap_context.get("initialized", False)),
            "open_price": to_float(gap_context.get("open_price"), 0.0)
            if gap_context.get("open_price") is not None
            else None,
            "prev_close": to_float(gap_context.get("prev_close"), 0.0)
            if gap_context.get("prev_close") is not None
            else None,
            "gap_pct": to_float(gap_context.get("gap_pct"), 0.0),
            "gap_direction": str(gap_context.get("gap_direction", "none")),
            "gap_fill_target": to_float(gap_context.get("gap_fill_target"), 0.0)
            if gap_context.get("gap_fill_target") is not None
            else None,
            "favor_side": str(gap_context.get("favor_side", "none")),
            "momentum_bias_side": str(gap_context.get("momentum_bias_side", "none")),
        },
        "market_activity": {
            "rvol": to_float(market_activity.get("rvol"), 0.0)
            if market_activity.get("rvol") is not None
            else None,
            "avg_volume_lookback": to_float(
                market_activity.get("avg_volume_lookback"),
                0.0,
            )
            if market_activity.get("avg_volume_lookback") is not None
            else None,
            "rvol_lookback_bars": int(
                to_float(
                    market_activity.get("rvol_lookback_bars"),
                    cfg.get("rvol_lookback_bars", 20),
                )
            ),
        },
        "adaptive_window": {
            "enabled": bool(adaptive_window.get("enabled", True)),
            "ready": bool(adaptive_window.get("ready", False)),
            "ready_bar_index": int(to_float(adaptive_window.get("ready_bar_index"), -1.0)),
            "rvol": to_float(adaptive_window.get("rvol"), 0.0)
            if adaptive_window.get("rvol") is not None
            else None,
            "atr_ratio": to_float(adaptive_window.get("atr_ratio"), 0.0)
            if adaptive_window.get("atr_ratio") is not None
            else None,
        },
        "poc_migration": poc_migration,
        "stats": {
            "total_levels": len(normalized["levels"]),
            "active_levels": active_levels,
            "tested_levels": tested_levels,
            "broken_levels": broken_levels,
            "memory_levels": memory_levels,
            "bounce_events": bounce_events,
            "break_events": break_events,
        },
    }



def intraday_levels_indicator_payload(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    snapshot = build_intraday_levels_snapshot(state)
    profile = snapshot.get("volume_profile", {})
    composite = (
        profile.get("composite_profile", {})
        if isinstance(profile.get("composite_profile"), dict)
        else {}
    )
    return {
        "enabled": bool(snapshot.get("enabled", True)),
        "levels": list(snapshot.get("levels", [])),
        "latest_event": snapshot.get("latest_event"),
        "stats": dict(snapshot.get("stats", {})),
        "opening_range": (
            dict(snapshot.get("opening_range", {}))
            if isinstance(snapshot.get("opening_range"), dict)
            else {}
        ),
        "poc_migration": (
            dict(snapshot.get("poc_migration", {}))
            if isinstance(snapshot.get("poc_migration"), dict)
            else {}
        ),
        "gap_context": (
            dict(snapshot.get("gap_context", {}))
            if isinstance(snapshot.get("gap_context"), dict)
            else {}
        ),
        "market_activity": (
            dict(snapshot.get("market_activity", {}))
            if isinstance(snapshot.get("market_activity"), dict)
            else {}
        ),
        "adaptive_window": (
            dict(snapshot.get("adaptive_window", {}))
            if isinstance(snapshot.get("adaptive_window"), dict)
            else {}
        ),
        "volume_profile": {
            "poc_price": profile.get("poc_price"),
            "value_area_low": profile.get("value_area_low"),
            "value_area_high": profile.get("value_area_high"),
            "value_area_coverage": profile.get("value_area_coverage"),
            "composite_poc_price": composite.get("composite_poc_price"),
            "composite_value_area_low": composite.get("composite_value_area_low"),
            "composite_value_area_high": composite.get("composite_value_area_high"),
        },
    }




__all__ = [
    "build_intraday_levels_snapshot",
    "intraday_levels_indicator_payload",
]
