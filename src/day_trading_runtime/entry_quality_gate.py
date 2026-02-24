"""Intraday level entry-quality gate helpers for runtime processing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..day_trading_models import TradingSession
from ..intraday_levels import ensure_intraday_levels_state
from ..trading_config import TradingConfig
from ..strategies.base_strategy import Signal, SignalType
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


def _compact_level_payload(level: Dict[str, Any], current_price: float) -> Dict[str, Any]:
    level_price = _to_optional_float(level.get("price")) or 0.0
    distance_pct = _safe_level_distance_pct(current_price, level_price)
    source = str(level.get("source", "")).strip().lower()
    memory_level = bool(level.get("memory_level", False))
    memory_age_days = (
        int(_to_optional_float(level.get("memory_age_days")) or 0.0)
        if memory_level
        else None
    )
    if memory_level:
        confluence_points = 2 if (memory_age_days or 0) <= 1 else 3
    elif source.startswith("opening_range"):
        confluence_points = 2
    else:
        confluence_points = 1
    return {
        "id": int(level.get("id", -1)),
        "kind": str(level.get("kind", "")),
        "price": round(level_price, 4),
        "source": str(level.get("source", "")),
        "tests": int(level.get("tests", 0)),
        "broken": bool(level.get("broken", False)),
        "memory_level": memory_level,
        "memory_age_days": memory_age_days,
        "memory_weight": (
            round(float(_to_optional_float(level.get("memory_weight")) or 0.0), 4)
            if memory_level
            else None
        ),
        "confluence_points": int(confluence_points),
        "distance_pct": round(float(distance_pct), 4) if distance_pct is not None else None,
    }

def runtime_evaluate_intraday_levels_entry_quality(
    self,
    *,
    session: TradingSession,
    signal: Signal,
    current_price: float,
    current_bar_index: int,
) -> Dict[str, Any]:
    """
    Evaluate strategy-specific intraday level context before queuing entry.

    This gate is deterministic and uses only current/past session state.
    """
    strategy_key = self._canonical_strategy_key(signal.strategy_name or "")
    direction = 1 if signal.signal_type == SignalType.BUY else -1 if signal.signal_type == SignalType.SELL else 0
    side = "long" if direction > 0 else "short" if direction < 0 else "flat"

    config = session.config if isinstance(getattr(session, "config", None), TradingConfig) else TradingConfig()
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

    total_levels = len(levels_raw)
    active_levels = [lvl for lvl in levels_raw if not bool(lvl.get("broken", False))]
    tested_levels_count = sum(1 for lvl in levels_raw if int(lvl.get("tests", 0)) > 0)
    broken_levels_count = sum(1 for lvl in levels_raw if bool(lvl.get("broken", False)))

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
    near_levels.sort(key=lambda lvl: float(lvl.get("_distance_pct", 9_999.0)))
    near_tested_levels = [lvl for lvl in near_levels if int(lvl.get("tests", 0)) > 0]
    near_confluence_score = sum(
        _level_confluence_points(lvl) for lvl in near_tested_levels[:3]
    )
    current_rvol = _to_optional_float(market_activity.get("rvol"))
    rvol_confluence_bonus = int(
        current_rvol is not None and current_rvol >= rvol_strong_threshold
    )
    near_confluence_score += rvol_confluence_bonus
    near_memory_levels_count = sum(1 for lvl in near_levels if bool(lvl.get("memory_level", False)))

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

    poc_price = _to_optional_float(volume_profile.get("poc_price"))
    composite_poc_price = _to_optional_float(composite_profile.get("composite_poc_price"))
    effective_poc_price = poc_price if poc_price is not None else composite_poc_price
    poc_source = "session" if poc_price is not None else ("composite" if composite_poc_price is not None else "unknown")
    value_area_low = _to_optional_float(volume_profile.get("value_area_low"))
    value_area_high = _to_optional_float(volume_profile.get("value_area_high"))
    composite_value_area_low = _to_optional_float(composite_profile.get("composite_value_area_low"))
    composite_value_area_high = _to_optional_float(composite_profile.get("composite_value_area_high"))
    value_area_source = "unknown"
    active_value_area_low = value_area_low
    active_value_area_high = value_area_high
    if (
        value_area_low is not None
        and value_area_high is not None
        and value_area_low <= value_area_high
    ):
        value_area_source = "session"
    elif (
        composite_value_area_low is not None
        and composite_value_area_high is not None
        and composite_value_area_low <= composite_value_area_high
    ):
        value_area_source = "composite"
        active_value_area_low = composite_value_area_low
        active_value_area_high = composite_value_area_high

    if (
        active_value_area_low is not None
        and active_value_area_high is not None
        and active_value_area_low <= active_value_area_high
    ):
        if current_price < active_value_area_low:
            value_area_position = "below"
        elif current_price > active_value_area_high:
            value_area_position = "above"
        else:
            value_area_position = "inside"
    else:
        value_area_position = "unknown"
    price_outside_value_area = value_area_position in {"below", "above"}

    # Near-edge tolerance: price slightly inside VA (within 0.15% of boundary)
    # counts as "near_edge" for MR/rotation that trade at VA extremes.
    va_near_edge = False
    if (
        value_area_position == "inside"
        and active_value_area_low is not None
        and active_value_area_high is not None
    ):
        va_range = max(1e-9, active_value_area_high - active_value_area_low)
        edge_tolerance = va_range * 0.10  # 10% of VA range
        if (current_price - active_value_area_low) <= edge_tolerance:
            va_near_edge = True
        elif (active_value_area_high - current_price) <= edge_tolerance:
            va_near_edge = True

    poc_on_trade_side = False
    poc_position = "unknown"
    if effective_poc_price is not None:
        if abs(effective_poc_price - current_price) <= max(0.01, current_price * 0.0001):
            poc_position = "at"
        elif effective_poc_price > current_price:
            poc_position = "above"
        else:
            poc_position = "below"
        poc_on_trade_side = (
            (direction > 0 and effective_poc_price > current_price)
            or (direction < 0 and effective_poc_price < current_price)
        )

    opening_breakout_direction = str(opening_range.get("breakout_direction", "")).strip().lower()
    poc_migration_bias = str(poc_migration.get("regime_bias", "unknown")).strip().lower()
    adaptive_window_ready = bool(adaptive_window.get("ready", False))
    gap_pct = _to_optional_float(gap_context.get("gap_pct")) or 0.0
    gap_abs_pct = abs(gap_pct)
    gap_favor_side = str(gap_context.get("favor_side", "none")).strip().lower()
    gap_momentum_bias_side = str(
        gap_context.get("momentum_bias_side", "none")
    ).strip().lower()

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

    # Bounce conflict buffer: if no aligned bounce found, check for opposing bounce
    # within buffer bars and waive the bounce requirement
    bounce_conflict_buffer = max(
        0,
        int(getattr(config, "intraday_levels_bounce_conflict_buffer_bars", 0)),
    )
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
                # Opposing bounce within buffer — waive bounce requirement
                recent_aligned_bounce = {
                    "_synthetic": True,
                    "_conflict_override": True,
                    "_age_bars": int(age),
                    "_opposing_bounce_direction": _event_direction(event),
                }
                break

    room_to_next_opposite_level_pct: Optional[float] = None
    if direction > 0 and next_resistance_above is not None and current_price > 0.0:
        room_to_next_opposite_level_pct = (
            (float(next_resistance_above["price"]) - current_price) / current_price
        ) * 100.0
    elif direction < 0 and next_support_below is not None and current_price > 0.0:
        room_to_next_opposite_level_pct = (
            (current_price - float(next_support_below["price"])) / current_price
        ) * 100.0

    broken_ratio = (
        (broken_levels_count / total_levels)
        if total_levels > 0
        else 0.0
    )

    SOFT_CHECKS: set = {
        "opening_range_breakout_aligned",
        "momentum_poc_migration_aligned",
        "gap_momentum_bias_aligned",
        "gap_fill_bias_aligned",
        "poc_migration_not_opposed",
        "rotation_prefers_non_trending_poc_migration",
        "mean_reversion_prefers_non_trending_poc_migration",
        "poc_on_trade_side",
        "no_untested_obstacle_to_poc",
    }

    checks: Dict[str, bool] = {
        "tracker_enabled": tracker_enabled,
        "entry_quality_enabled": gate_enabled,
        "valid_direction": direction != 0,
        "minimum_levels_context": total_levels >= min_levels_for_context,
    }
    reasons: List[str] = []
    soft_reasons: List[str] = []
    if not tracker_enabled:
        reasons.append("intraday_tracker_disabled")
    if direction == 0:
        reasons.append("unsupported_signal_type_for_level_gate")
    if total_levels < min_levels_for_context:
        reasons.append("insufficient_level_context")
    checks["rvol_filter_enabled"] = bool(rvol_filter_enabled)
    checks["rvol_available"] = current_rvol is not None
    checks["rvol_minimum"] = bool(
        not rvol_filter_enabled
        or (current_rvol is not None and current_rvol >= rvol_min_threshold)
    )
    if rvol_filter_enabled and not checks["rvol_available"]:
        reasons.append("rvol_unavailable")
    if rvol_filter_enabled and checks["rvol_available"] and not checks["rvol_minimum"]:
        reasons.append("rvol_below_min_threshold")

    checks["adaptive_window_enabled"] = bool(adaptive_window_enabled)
    checks["adaptive_window_ready"] = bool(
        (not adaptive_window_enabled) or adaptive_window_ready
    )
    if adaptive_window_enabled and not adaptive_window_ready:
        reasons.append("adaptive_time_window_not_ready")

    target_price_override: Optional[float] = None
    untested_obstacles_to_poc: List[Dict[str, Any]] = []
    if direction != 0:
        if strategy_key == "vwap_magnet":
            checks["near_tested_level"] = len(near_tested_levels) >= 1
            checks["minimum_confluence_score"] = near_confluence_score >= min_confluence_score
            checks["outside_value_area"] = price_outside_value_area
            checks["poc_on_trade_side"] = poc_on_trade_side
            checks["no_recent_volume_break"] = recent_break_event is None
            checks["poc_migration_not_opposed"] = bool(
                poc_migration_bias in {"unknown", "range", "drift"}
                or (poc_migration_bias == "uptrend" and direction > 0)
                or (poc_migration_bias == "downtrend" and direction < 0)
            )
            checks["gap_fill_bias_aligned"] = True
            if gap_analysis_enabled and gap_abs_pct >= gap_min_pct and gap_abs_pct < gap_momentum_threshold_pct:
                checks["gap_fill_bias_aligned"] = bool(
                    gap_favor_side == "none"
                    or (gap_favor_side == "long" and direction > 0)
                    or (gap_favor_side == "short" and direction < 0)
                )

            if poc_on_trade_side and effective_poc_price is not None:
                for level in active_levels:
                    level_price = _to_optional_float(level.get("price"))
                    if level_price is None:
                        continue
                    kind = str(level.get("kind", "")).strip().lower()
                    tests = int(level.get("tests", 0))
                    if tests > 0:
                        continue
                    if direction > 0:
                        if (
                            kind == "resistance"
                            and current_price < level_price < effective_poc_price
                        ):
                            untested_obstacles_to_poc.append(level)
                    else:
                        if (
                            kind == "support"
                            and effective_poc_price < level_price < current_price
                        ):
                            untested_obstacles_to_poc.append(level)
            checks["no_untested_obstacle_to_poc"] = len(untested_obstacles_to_poc) == 0

            if not checks["near_tested_level"]:
                reasons.append("vwap_requires_tested_level_near_entry")
            if not checks["minimum_confluence_score"]:
                reasons.append("vwap_insufficient_confluence_score")
            if not checks["outside_value_area"]:
                reasons.append("vwap_requires_price_outside_value_area")
            if not checks["poc_on_trade_side"]:
                soft_reasons.append("vwap_requires_poc_on_trade_side")
            if not checks["no_recent_volume_break"]:
                reasons.append("vwap_blocked_recent_volume_break")
            if not checks["no_untested_obstacle_to_poc"]:
                soft_reasons.append("vwap_blocked_untested_obstacle_to_poc")
            if not checks["poc_migration_not_opposed"]:
                soft_reasons.append("vwap_poc_migration_opposed")
            if not checks["gap_fill_bias_aligned"]:
                soft_reasons.append("vwap_gap_fill_bias_mismatch")

        elif strategy_key == "rotation":
            candidate_level = near_tested_levels[0] if near_tested_levels else None
            tests_count = int(candidate_level.get("tests", 0)) if candidate_level else 0
            checks["near_tested_level"] = candidate_level is not None
            checks["minimum_confluence_score"] = near_confluence_score >= min_confluence_score
            checks["rotation_level_tests_range"] = bool(
                candidate_level and 1 <= tests_count <= rotation_max_tests
            )
            checks["rotation_level_unbroken"] = bool(
                candidate_level and not bool(candidate_level.get("broken", False))
            )

            volume_ratio = _to_optional_float((signal.metadata or {}).get("volume_ratio"))
            checks["rotation_volume_ratio_available"] = volume_ratio is not None
            checks["rotation_volume_exhaustion"] = bool(
                volume_ratio is not None and volume_ratio <= rotation_volume_max_ratio
            )
            # False breakdown / liquidity sweep: break then bounce = valid rotation setup
            _rot_break_then_bounce = (
                recent_break_event is not None
                and recent_aligned_bounce is not None
                and int(recent_aligned_bounce.get("bar_index", -1))
                > int(recent_break_event.get("bar_index", -2))
            )
            checks["no_recent_volume_break"] = recent_break_event is None or _rot_break_then_bounce
            checks["rotation_prefers_non_trending_poc_migration"] = bool(
                poc_migration_bias in {"unknown", "range", "drift"}
            )
            checks["gap_fill_bias_aligned"] = True
            if gap_analysis_enabled and gap_abs_pct >= gap_min_pct and gap_abs_pct < gap_momentum_threshold_pct:
                checks["gap_fill_bias_aligned"] = bool(
                    gap_favor_side == "none"
                    or (gap_favor_side == "long" and direction > 0)
                    or (gap_favor_side == "short" and direction < 0)
                )

            if not checks["near_tested_level"]:
                reasons.append("rotation_requires_tested_level_near_entry")
            if not checks["minimum_confluence_score"]:
                reasons.append("rotation_insufficient_confluence_score")
            if not checks["rotation_level_tests_range"]:
                reasons.append("rotation_level_overtested")
            if not checks["rotation_level_unbroken"]:
                reasons.append("rotation_level_already_broken")
            if not checks["rotation_volume_ratio_available"]:
                reasons.append("rotation_volume_ratio_missing")
            if checks["rotation_volume_ratio_available"] and not checks["rotation_volume_exhaustion"]:
                reasons.append("rotation_volume_not_exhausted")
            if not checks["no_recent_volume_break"]:
                reasons.append("rotation_blocked_recent_volume_break")
            if not checks["rotation_prefers_non_trending_poc_migration"]:
                soft_reasons.append("rotation_poc_migration_trending")
            if not checks["gap_fill_bias_aligned"]:
                soft_reasons.append("rotation_gap_fill_bias_mismatch")

        elif strategy_key == "mean_reversion":
            checks["near_tested_level"] = len(near_tested_levels) >= 1
            checks["minimum_confluence_score"] = near_confluence_score >= min_confluence_score
            # MR at VA edge (slightly inside) is a valid setup
            checks["outside_value_area"] = price_outside_value_area or va_near_edge
            checks["poc_on_trade_side"] = poc_on_trade_side
            # False breakdown / liquidity sweep: if price broke a level then bounced
            # back, that's the best MR setup. Allow entry when aligned bounce follows break.
            _break_then_bounce = (
                recent_break_event is not None
                and recent_aligned_bounce is not None
                and int(recent_aligned_bounce.get("bar_index", -1))
                > int(recent_break_event.get("bar_index", -2))
            )
            checks["no_recent_volume_break"] = recent_break_event is None or _break_then_bounce
            checks["recent_aligned_bounce"] = (
                True if not require_recent_bounce_for_mr else recent_aligned_bounce is not None
            )
            checks["mean_reversion_prefers_non_trending_poc_migration"] = bool(
                poc_migration_bias in {"unknown", "range", "drift"}
            )
            checks["gap_fill_bias_aligned"] = True
            if gap_analysis_enabled and gap_abs_pct >= gap_min_pct and gap_abs_pct < gap_momentum_threshold_pct:
                checks["gap_fill_bias_aligned"] = bool(
                    gap_favor_side == "none"
                    or (gap_favor_side == "long" and direction > 0)
                    or (gap_favor_side == "short" and direction < 0)
                )

            if not checks["near_tested_level"]:
                reasons.append("mean_reversion_requires_tested_level_near_entry")
            if not checks["minimum_confluence_score"]:
                reasons.append("mean_reversion_insufficient_confluence_score")
            if not checks["outside_value_area"]:
                reasons.append("mean_reversion_requires_price_outside_value_area")
            if not checks["poc_on_trade_side"]:
                soft_reasons.append("mean_reversion_requires_poc_on_trade_side")
            if not checks["no_recent_volume_break"]:
                reasons.append("mean_reversion_blocked_recent_volume_break")
            if not checks["recent_aligned_bounce"]:
                reasons.append("mean_reversion_requires_recent_bounce")
            if not checks["mean_reversion_prefers_non_trending_poc_migration"]:
                soft_reasons.append("mean_reversion_poc_migration_trending")
            if not checks["gap_fill_bias_aligned"]:
                soft_reasons.append("mean_reversion_gap_fill_bias_mismatch")

            if (
                checks["outside_value_area"]
                and checks["poc_on_trade_side"]
                and effective_poc_price is not None
            ):
                target_price_override = float(effective_poc_price)

        elif strategy_key == "momentum":
            latest_break_confirmed = bool(
                isinstance(latest_event, dict)
                and str(latest_event.get("event_type", "")).strip().lower() == "break"
                and bool(latest_event.get("volume_confirmed", False))
            )
            latest_break_age = (
                int(current_bar_index - int(latest_event.get("bar_index", -1)))
                if latest_break_confirmed
                else None
            )
            checks["latest_event_break"] = latest_break_confirmed
            checks["break_age_within_limit"] = bool(
                latest_break_age is not None and 0 <= latest_break_age <= momentum_break_max_age_bars
            )
            checks["break_direction_aligned"] = bool(
                latest_break_confirmed and _event_direction(latest_event) == direction
            )
            checks["outside_value_area"] = value_area_position != "inside"
            checks["broken_ratio_minimum"] = broken_ratio >= momentum_min_broken_ratio
            checks["room_to_next_level"] = bool(
                room_to_next_opposite_level_pct is None
                or room_to_next_opposite_level_pct >= momentum_min_room_pct
            )
            checks["opening_range_breakout_aligned"] = bool(
                not opening_breakout_direction
                or (opening_breakout_direction == "bullish" and direction > 0)
                or (opening_breakout_direction == "bearish" and direction < 0)
            )
            checks["momentum_poc_migration_aligned"] = bool(
                poc_migration_bias in {"unknown", "drift"}
                or (poc_migration_bias == "uptrend" and direction > 0)
                or (poc_migration_bias == "downtrend" and direction < 0)
            )
            checks["gap_momentum_bias_aligned"] = True
            if gap_analysis_enabled and gap_abs_pct >= gap_momentum_threshold_pct:
                checks["gap_momentum_bias_aligned"] = bool(
                    gap_momentum_bias_side == "none"
                    or (gap_momentum_bias_side == "long" and direction > 0)
                    or (gap_momentum_bias_side == "short" and direction < 0)
                )

            if not checks["latest_event_break"]:
                reasons.append("momentum_requires_latest_event_break")
            if not checks["break_age_within_limit"]:
                reasons.append("momentum_break_too_old")
            if not checks["break_direction_aligned"]:
                reasons.append("momentum_break_direction_mismatch")
            if not checks["outside_value_area"]:
                reasons.append("momentum_blocked_inside_value_area")
            if not checks["broken_ratio_minimum"]:
                reasons.append("momentum_broken_ratio_too_low")
            if not checks["room_to_next_level"]:
                reasons.append("momentum_insufficient_room_to_next_level")
            if not checks["opening_range_breakout_aligned"]:
                soft_reasons.append("momentum_opening_range_breakout_mismatch")
            if not checks["momentum_poc_migration_aligned"]:
                soft_reasons.append("momentum_poc_migration_mismatch")
            if not checks["gap_momentum_bias_aligned"]:
                soft_reasons.append("momentum_gap_bias_mismatch")
        elif strategy_key == "pullback":
            checks["strategy_specific_gate_applied"] = True

            # Pullbacks require a tested level nearby or at least some level context
            checks["near_tested_level"] = len(near_tested_levels) >= 1
            checks["minimum_confluence_score"] = near_confluence_score >= 1
            checks["poc_on_trade_side"] = poc_on_trade_side or effective_poc_price is None

            # Relax RVOL requirement for pullbacks (default to 0.5 if not configured)
            pullback_rvol_min = max(
                0.0,
                float(getattr(config, "intraday_levels_pullback_rvol_min_threshold", 0.50))
            )

            # Override the rvol_minimum check
            checks["rvol_minimum"] = bool(
                not rvol_filter_enabled
                or (current_rvol is not None and current_rvol >= pullback_rvol_min)
            )

            if not checks["near_tested_level"]:
                soft_reasons.append("pullback_prefers_tested_level_near_entry")
            if not checks["minimum_confluence_score"]:
                soft_reasons.append("pullback_prefers_confluence")

            # Properly manage the rejection reasons array
            if rvol_filter_enabled and checks["rvol_available"] and not checks["rvol_minimum"]:
                if "rvol_below_min_threshold" not in reasons:
                    reasons.append("rvol_below_pullback_min_threshold")
            elif "rvol_below_min_threshold" in reasons and checks["rvol_minimum"]:
                reasons.remove("rvol_below_min_threshold")
        else:
            checks["strategy_specific_gate_applied"] = False
    if not gate_enabled:
        reasons = []
        soft_reasons = []

    passed = (not reasons) or (not gate_enabled)
    has_soft_warnings = bool(soft_reasons) and passed
    reason = "disabled" if not gate_enabled else ("passed" if passed else reasons[0])

    soft_failed_checks = [k for k, v in checks.items() if not v and k in SOFT_CHECKS]

    near_levels_payload = [
        _compact_level_payload(level, current_price)
        for level in near_levels[:4]
    ]
    context_payload: Dict[str, Any] = {
        "gate": "intraday_levels_entry_quality",
        "gate_enabled": gate_enabled,
        "passed": bool(passed),
        "has_soft_warnings": has_soft_warnings,
        "reason": reason,
        "reasons": list(dict.fromkeys(reasons)),
        "soft_reasons": list(dict.fromkeys(soft_reasons)),
        "soft_failed_checks": soft_failed_checks,
        "strategy_key": strategy_key,
        "side": side,
        "signal_type": signal.signal_type.value if signal.signal_type else None,
        "checks": checks,
        "stats": {
            "total_levels": int(total_levels),
            "active_levels": int(len(active_levels)),
            "tested_levels": int(tested_levels_count),
            "broken_levels": int(broken_levels_count),
            "broken_ratio": round(float(broken_ratio), 4),
            "near_levels_count": int(len(near_levels)),
            "near_tested_levels_count": int(len(near_tested_levels)),
            "near_confluence_score": int(near_confluence_score),
            "rvol_confluence_bonus": int(rvol_confluence_bonus),
            "near_memory_levels_count": int(near_memory_levels_count),
        },
        "near_levels": near_levels_payload,
        "latest_event": dict(latest_event) if isinstance(latest_event, dict) else None,
        "bars_since_latest_event": bars_since_latest_event,
        "recent_break_event": (
            {
                key: value
                for key, value in recent_break_event.items()
                if key != "_age_bars"
            }
            if isinstance(recent_break_event, dict)
            else None
        ),
        "recent_break_event_age_bars": (
            int(recent_break_event["_age_bars"]) if isinstance(recent_break_event, dict) else None
        ),
        "recent_aligned_bounce": (
            {
                key: value
                for key, value in recent_aligned_bounce.items()
                if key != "_age_bars"
            }
            if isinstance(recent_aligned_bounce, dict)
            else None
        ),
        "recent_aligned_bounce_age_bars": (
            int(recent_aligned_bounce["_age_bars"]) if isinstance(recent_aligned_bounce, dict) else None
        ),
        "volume_profile": {
            "poc_price": poc_price,
            "composite_poc_price": composite_poc_price,
            "effective_poc_price": effective_poc_price,
            "poc_source": poc_source,
            "value_area_low": value_area_low,
            "value_area_high": value_area_high,
            "composite_value_area_low": composite_value_area_low,
            "composite_value_area_high": composite_value_area_high,
            "value_area_source": value_area_source,
            "value_area_position": value_area_position,
            "price_outside_value_area": bool(price_outside_value_area),
            "poc_position": poc_position,
            "poc_on_trade_side": bool(poc_on_trade_side),
        },
        "opening_range": {
            "complete": bool(opening_range.get("complete", False)),
            "high": _to_optional_float(opening_range.get("high")),
            "low": _to_optional_float(opening_range.get("low")),
            "mid": _to_optional_float(opening_range.get("mid")),
            "breakout_direction": (
                opening_breakout_direction if opening_breakout_direction else None
            ),
            "breakout_bar_index": (
                int(_to_optional_float(opening_range.get("breakout_bar_index")) or -1)
                if opening_range.get("breakout_bar_index") is not None
                else -1
            ),
        },
        "poc_migration": {
            "regime_bias": poc_migration_bias,
            "change_pct": _to_optional_float(poc_migration.get("change_pct")),
            "history_points": int(_to_optional_float(poc_migration.get("history_points")) or 0),
            "start_poc_price": _to_optional_float(poc_migration.get("start_poc_price")),
            "current_poc_price": _to_optional_float(
                poc_migration.get("current_poc_price")
            ),
        },
        "gap_context": {
            "enabled": bool(gap_context.get("enabled", True)),
            "gap_pct": float(gap_pct),
            "gap_direction": str(gap_context.get("gap_direction", "none")),
            "favor_side": gap_favor_side,
            "momentum_bias_side": gap_momentum_bias_side,
            "gap_fill_target": _to_optional_float(gap_context.get("gap_fill_target")),
            "gap_abs_pct": float(gap_abs_pct),
        },
        "market_activity": {
            "rvol": current_rvol,
            "avg_volume_lookback": _to_optional_float(
                market_activity.get("avg_volume_lookback")
            ),
            "rvol_lookback_bars": int(
                _to_optional_float(market_activity.get("rvol_lookback_bars")) or 0
            ),
            "rvol_filter_enabled": bool(rvol_filter_enabled),
            "rvol_min_threshold": float(rvol_min_threshold),
            "rvol_strong_threshold": float(rvol_strong_threshold),
        },
        "adaptive_window": {
            "enabled": bool(adaptive_window_enabled),
            "ready": bool(adaptive_window_ready),
            "ready_bar_index": int(
                _to_optional_float(adaptive_window.get("ready_bar_index")) or -1
            ),
            "atr_ratio": _to_optional_float(adaptive_window.get("atr_ratio")),
            "rvol": _to_optional_float(adaptive_window.get("rvol")),
        },
        "room_to_next_opposite_level_pct": (
            round(float(room_to_next_opposite_level_pct), 4)
            if room_to_next_opposite_level_pct is not None
            else None
        ),
        "untested_obstacles_to_poc": [
            _compact_level_payload(level, current_price)
            for level in untested_obstacles_to_poc[:3]
        ],
        "config": {
            "min_levels_for_context": int(min_levels_for_context),
            "entry_tolerance_pct": float(entry_tolerance_pct),
            "break_cooldown_bars": int(break_cooldown_bars),
            "rotation_max_tests": int(rotation_max_tests),
            "rotation_volume_max_ratio": float(rotation_volume_max_ratio),
            "recent_bounce_lookback_bars": int(recent_bounce_lookback_bars),
            "require_recent_bounce_for_mean_reversion": bool(
                require_recent_bounce_for_mr
            ),
            "momentum_break_max_age_bars": int(momentum_break_max_age_bars),
            "momentum_min_room_pct": float(momentum_min_room_pct),
            "momentum_min_broken_ratio": float(momentum_min_broken_ratio),
            "min_confluence_score": int(min_confluence_score),
            "memory_enabled": bool(intraday_cfg.get("memory_enabled", True)),
            "memory_min_tests": int(_to_optional_float(intraday_cfg.get("memory_min_tests")) or 2),
            "memory_max_age_days": int(
                _to_optional_float(intraday_cfg.get("memory_max_age_days")) or 5
            ),
            "opening_range_enabled": bool(intraday_cfg.get("opening_range_enabled", True)),
            "opening_range_minutes": int(
                _to_optional_float(intraday_cfg.get("opening_range_minutes")) or 30
            ),
            "poc_migration_enabled": bool(
                intraday_cfg.get("poc_migration_enabled", True)
            ),
            "poc_migration_interval_bars": int(
                _to_optional_float(intraday_cfg.get("poc_migration_interval_bars")) or 30
            ),
            "composite_profile_enabled": bool(
                intraday_cfg.get("composite_profile_enabled", True)
            ),
            "composite_profile_days": int(
                _to_optional_float(intraday_cfg.get("composite_profile_days")) or 3
            ),
            "spike_detection_enabled": bool(
                intraday_cfg.get("spike_detection_enabled", True)
            ),
            "spike_min_wick_ratio": float(
                _to_optional_float(intraday_cfg.get("spike_min_wick_ratio")) or 0.60
            ),
            "prior_day_anchors_enabled": bool(
                intraday_cfg.get("prior_day_anchors_enabled", True)
            ),
            "gap_analysis_enabled": bool(gap_analysis_enabled),
            "gap_min_pct": float(gap_min_pct),
            "gap_momentum_threshold_pct": float(gap_momentum_threshold_pct),
            "rvol_filter_enabled": bool(rvol_filter_enabled),
            "rvol_min_threshold": float(rvol_min_threshold),
            "rvol_strong_threshold": float(rvol_strong_threshold),
            "adaptive_window_enabled": bool(adaptive_window_enabled),
        },
    }
    if target_price_override is not None and target_price_override > 0.0:
        context_payload["target_price_override"] = round(float(target_price_override), 4)
    return context_payload
