"""Input/state extraction helpers for intraday entry-quality gate."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..day_trading_models import TradingSession
from ..intraday_levels import ensure_intraday_levels_state
from ..trading_config import TradingConfig
from ..day_trading_runtime_sweep import (
    safe_level_distance_pct as _safe_level_distance_pct_impl,
    to_optional_float as _to_optional_float_impl,
)


def _to_optional_float(value: Any) -> Optional[float]:
    return _to_optional_float_impl(value)


def _safe_level_distance_pct(price: float, level_price: float) -> Optional[float]:
    return _safe_level_distance_pct_impl(price, level_price)


def _event_direction(event: Optional[Dict[str, Any]]) -> int:
    if not isinstance(event, dict):
        return 0
    direction = str(event.get("direction", "")).strip().lower()
    if direction == "bullish":
        return 1
    if direction == "bearish":
        return -1
    return 0


def _level_confluence_points(level: Dict[str, Any]) -> int:
    if bool(level.get("memory_level", False)):
        age_days = int(_to_optional_float(level.get("memory_age_days")) or 0.0)
        return 2 if age_days <= 1 else 3
    source = str(level.get("source", "")).strip().lower()
    if source.startswith("opening_range"):
        return 2
    if source.startswith("prior_day"):
        return 2
    if source.startswith("spike_"):
        return 2
    if source == "gap_fill_target":
        return 2
    return 1


def extract_gate_state(session: TradingSession) -> Dict[str, Any]:
    state = ensure_intraday_levels_state(getattr(session, "intraday_levels_state", {}))
    intraday_cfg = state.get("config", {}) if isinstance(state.get("config"), dict) else {}
    levels_raw = list(state.get("levels", [])) if isinstance(state.get("levels"), list) else []
    recent_events = list(state.get("recent_events", [])) if isinstance(state.get("recent_events"), list) else []
    volume_profile = state.get("volume_profile", {}) if isinstance(state.get("volume_profile"), dict) else {}
    snapshot = state.get("snapshot", {}) if isinstance(state.get("snapshot"), dict) else {}
    opening_range = (
        snapshot.get("opening_range", {})
        if isinstance(snapshot.get("opening_range"), dict)
        else {}
    )
    gap_context = (
        snapshot.get("gap_context", {})
        if isinstance(snapshot.get("gap_context"), dict)
        else {}
    )
    market_activity = (
        snapshot.get("market_activity", {})
        if isinstance(snapshot.get("market_activity"), dict)
        else {}
    )
    adaptive_window = (
        snapshot.get("adaptive_window", {})
        if isinstance(snapshot.get("adaptive_window"), dict)
        else {}
    )
    poc_migration = (
        snapshot.get("poc_migration", {})
        if isinstance(snapshot.get("poc_migration"), dict)
        else {}
    )
    snapshot_profile = (
        snapshot.get("volume_profile", {})
        if isinstance(snapshot.get("volume_profile"), dict)
        else {}
    )
    composite_profile = (
        snapshot_profile.get("composite_profile", {})
        if isinstance(snapshot_profile.get("composite_profile"), dict)
        else {}
    )
    return {
        "state": state,
        "intraday_cfg": intraday_cfg,
        "levels_raw": levels_raw,
        "recent_events": recent_events,
        "volume_profile": volume_profile,
        "snapshot": snapshot,
        "opening_range": opening_range,
        "gap_context": gap_context,
        "market_activity": market_activity,
        "adaptive_window": adaptive_window,
        "poc_migration": poc_migration,
        "composite_profile": composite_profile,
    }


def resolve_gate_thresholds(
    config: TradingConfig,
    intraday_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    gate_enabled = bool(getattr(config, "intraday_levels_entry_quality_enabled", True))
    tracker_enabled = bool(intraday_cfg.get("enabled", True))
    min_levels_for_context = max(
        1,
        int(getattr(config, "intraday_levels_min_levels_for_context", 2)),
    )
    entry_tolerance_pct = max(
        0.01,
        float(getattr(config, "intraday_levels_entry_tolerance_pct", 0.10)),
    )
    break_cooldown_bars = max(
        1,
        int(getattr(config, "intraday_levels_break_cooldown_bars", 6)),
    )
    rotation_max_tests = max(
        1,
        int(getattr(config, "intraday_levels_rotation_max_tests", 2)),
    )
    rotation_volume_max_ratio = min(
        2.0,
        max(0.1, float(getattr(config, "intraday_levels_rotation_volume_max_ratio", 0.95))),
    )
    recent_bounce_lookback_bars = max(
        1,
        int(getattr(config, "intraday_levels_recent_bounce_lookback_bars", 6)),
    )
    require_recent_bounce_for_mr = bool(
        getattr(config, "intraday_levels_require_recent_bounce_for_mean_reversion", True)
    )
    momentum_break_max_age_bars = max(
        1,
        int(getattr(config, "intraday_levels_momentum_break_max_age_bars", 3)),
    )
    momentum_min_room_pct = max(
        0.01,
        float(getattr(config, "intraday_levels_momentum_min_room_pct", 0.30)),
    )
    momentum_min_broken_ratio = min(
        1.0,
        max(0.0, float(getattr(config, "intraday_levels_momentum_min_broken_ratio", 0.30))),
    )
    min_confluence_score = max(
        1,
        int(getattr(config, "intraday_levels_min_confluence_score", 2)),
    )
    rvol_filter_enabled = bool(
        getattr(config, "intraday_levels_rvol_filter_enabled", True)
    )
    rvol_min_threshold = max(
        0.0,
        float(getattr(config, "intraday_levels_rvol_min_threshold", 0.80)),
    )
    rvol_strong_threshold = max(
        0.1,
        float(getattr(config, "intraday_levels_rvol_strong_threshold", 1.50)),
    )
    adaptive_window_enabled = bool(
        getattr(config, "intraday_levels_adaptive_window_enabled", True)
    )
    gap_analysis_enabled = bool(
        getattr(config, "intraday_levels_gap_analysis_enabled", True)
    )
    gap_min_pct = max(
        0.0,
        float(getattr(config, "intraday_levels_gap_min_pct", 0.30)),
    )
    gap_momentum_threshold_pct = max(
        0.1,
        float(getattr(config, "intraday_levels_gap_momentum_threshold_pct", 2.0)),
    )
    bounce_conflict_buffer = max(
        0,
        int(getattr(config, "intraday_levels_bounce_conflict_buffer_bars", 0)),
    )

    return {
        "gate_enabled": gate_enabled,
        "tracker_enabled": tracker_enabled,
        "min_levels_for_context": min_levels_for_context,
        "entry_tolerance_pct": entry_tolerance_pct,
        "break_cooldown_bars": break_cooldown_bars,
        "rotation_max_tests": rotation_max_tests,
        "rotation_volume_max_ratio": rotation_volume_max_ratio,
        "recent_bounce_lookback_bars": recent_bounce_lookback_bars,
        "require_recent_bounce_for_mr": require_recent_bounce_for_mr,
        "momentum_break_max_age_bars": momentum_break_max_age_bars,
        "momentum_min_room_pct": momentum_min_room_pct,
        "momentum_min_broken_ratio": momentum_min_broken_ratio,
        "min_confluence_score": min_confluence_score,
        "rvol_filter_enabled": rvol_filter_enabled,
        "rvol_min_threshold": rvol_min_threshold,
        "rvol_strong_threshold": rvol_strong_threshold,
        "adaptive_window_enabled": adaptive_window_enabled,
        "gap_analysis_enabled": gap_analysis_enabled,
        "gap_min_pct": gap_min_pct,
        "gap_momentum_threshold_pct": gap_momentum_threshold_pct,
        "bounce_conflict_buffer": bounce_conflict_buffer,
    }


def build_level_context(
    *,
    levels_raw: List[Dict[str, Any]],
    current_price: float,
    entry_tolerance_pct: float,
    current_rvol: Optional[float],
    rvol_strong_threshold: float,
) -> Dict[str, Any]:
    total_levels = len(levels_raw)
    active_levels = [level for level in levels_raw if not bool(level.get("broken", False))]
    tested_levels_count = sum(1 for level in levels_raw if int(level.get("tests", 0)) > 0)
    broken_levels_count = sum(1 for level in levels_raw if bool(level.get("broken", False)))

    near_levels: List[Dict[str, Any]] = []
    for level in active_levels:
        level_price = _to_optional_float(level.get("price"))
        if level_price is None:
            continue
        distance_pct = _safe_level_distance_pct(current_price, level_price)
        if distance_pct is None or distance_pct > entry_tolerance_pct:
            continue
        near_levels.append(
            {
                **level,
                "_distance_pct": distance_pct,
            }
        )
    near_levels.sort(key=lambda level: float(level.get("_distance_pct", 9_999.0)))
    near_tested_levels = [level for level in near_levels if int(level.get("tests", 0)) > 0]
    near_confluence_score = sum(
        _level_confluence_points(level) for level in near_tested_levels[:3]
    )
    rvol_confluence_bonus = int(
        current_rvol is not None and current_rvol >= rvol_strong_threshold
    )
    near_confluence_score += rvol_confluence_bonus
    near_memory_levels_count = sum(1 for level in near_levels if bool(level.get("memory_level", False)))

    next_resistance_above: Optional[Dict[str, Any]] = None
    next_support_below: Optional[Dict[str, Any]] = None
    for level in active_levels:
        level_price = _to_optional_float(level.get("price"))
        if level_price is None:
            continue
        kind = str(level.get("kind", "")).strip().lower()
        if kind == "resistance" and level_price > current_price:
            if next_resistance_above is None or level_price < float(next_resistance_above["price"]):
                next_resistance_above = {"price": level_price, "level": level}
        elif kind == "support" and level_price < current_price:
            if next_support_below is None or level_price > float(next_support_below["price"]):
                next_support_below = {"price": level_price, "level": level}

    broken_ratio = (
        (broken_levels_count / total_levels)
        if total_levels > 0
        else 0.0
    )
    return {
        "total_levels": total_levels,
        "active_levels": active_levels,
        "tested_levels_count": tested_levels_count,
        "broken_levels_count": broken_levels_count,
        "near_levels": near_levels,
        "near_tested_levels": near_tested_levels,
        "near_confluence_score": near_confluence_score,
        "rvol_confluence_bonus": rvol_confluence_bonus,
        "near_memory_levels_count": near_memory_levels_count,
        "next_resistance_above": next_resistance_above,
        "next_support_below": next_support_below,
        "broken_ratio": broken_ratio,
    }


def extract_recent_events_context(
    *,
    recent_events: List[Dict[str, Any]],
    current_bar_index: int,
    break_cooldown_bars: int,
    recent_bounce_lookback_bars: int,
    bounce_conflict_buffer: int,
    direction: int,
) -> Dict[str, Any]:
    latest_event = recent_events[-1] if recent_events else None
    latest_event_bar_index = -1
    if isinstance(latest_event, dict):
        try:
            latest_event_bar_index = int(latest_event.get("bar_index", -1))
        except (TypeError, ValueError):
            latest_event_bar_index = -1
    bars_since_latest_event = (
        int(current_bar_index - latest_event_bar_index)
        if latest_event_bar_index >= 0
        else None
    )

    recent_break_event: Optional[Dict[str, Any]] = None
    for event in reversed(recent_events):
        if not isinstance(event, dict):
            continue
        if str(event.get("event_type", "")).strip().lower() != "break":
            continue
        if not bool(event.get("volume_confirmed", False)):
            continue
        try:
            event_bar_index = int(event.get("bar_index", -1))
        except (TypeError, ValueError):
            continue
        if event_bar_index < 0:
            continue
        age = current_bar_index - event_bar_index
        if age < 0 or age > break_cooldown_bars:
            continue
        recent_break_event = {**event, "_age_bars": int(age)}
        break

    recent_aligned_bounce: Optional[Dict[str, Any]] = None
    for event in reversed(recent_events):
        if not isinstance(event, dict):
            continue
        if str(event.get("event_type", "")).strip().lower() != "bounce":
            continue
        try:
            event_bar_index = int(event.get("bar_index", -1))
        except (TypeError, ValueError):
            continue
        if event_bar_index < 0:
            continue
        age = current_bar_index - event_bar_index
        if age < 0 or age > recent_bounce_lookback_bars:
            continue
        if _event_direction(event) != direction:
            continue
        recent_aligned_bounce = {**event, "_age_bars": int(age)}
        break

    if recent_aligned_bounce is None and bounce_conflict_buffer > 0:
        for event in reversed(recent_events):
            if not isinstance(event, dict):
                continue
            if str(event.get("event_type", "")).strip().lower() != "bounce":
                continue
            try:
                event_bar_index = int(event.get("bar_index", -1))
            except (TypeError, ValueError):
                continue
            if event_bar_index < 0:
                continue
            age = current_bar_index - event_bar_index
            if age < 0 or age > bounce_conflict_buffer:
                continue
            if _event_direction(event) == -direction:
                recent_aligned_bounce = {
                    "_synthetic": True,
                    "_conflict_override": True,
                    "_age_bars": int(age),
                    "_opposing_bounce_direction": _event_direction(event),
                }
                break

    return {
        "latest_event": latest_event,
        "bars_since_latest_event": bars_since_latest_event,
        "recent_break_event": recent_break_event,
        "recent_aligned_bounce": recent_aligned_bounce,
    }


__all__ = [
    "build_level_context",
    "extract_gate_state",
    "extract_recent_events_context",
    "resolve_gate_thresholds",
]
