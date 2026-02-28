"""Intraday level entry-quality gate helpers for runtime processing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..day_trading_models import TradingSession
from ..trading_config import TradingConfig
from ..strategies.base_strategy import Signal, SignalType
from ..day_trading_runtime_sweep import (
    safe_level_distance_pct as _safe_level_distance_pct_impl,
    to_optional_float as _to_optional_float_impl,
)
from .entry_quality_gate_inputs import (
    build_level_context,
    extract_gate_state,
    extract_recent_events_context,
    resolve_gate_thresholds,
)
from .entry_quality_gate_rules import (
    apply_strategy_specific_entry_quality_checks,
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
    gate_state = extract_gate_state(session)
    state = gate_state["state"]
    intraday_cfg = gate_state["intraday_cfg"]
    levels_raw = gate_state["levels_raw"]
    recent_events = gate_state["recent_events"]
    volume_profile = gate_state["volume_profile"]
    snapshot = gate_state["snapshot"]
    opening_range = gate_state["opening_range"]
    gap_context = gate_state["gap_context"]
    market_activity = gate_state["market_activity"]
    adaptive_window = gate_state["adaptive_window"]
    poc_migration = gate_state["poc_migration"]
    composite_profile = gate_state["composite_profile"]

    thresholds = resolve_gate_thresholds(config, intraday_cfg)
    gate_enabled = bool(thresholds["gate_enabled"])
    tracker_enabled = bool(thresholds["tracker_enabled"])
    min_levels_for_context = int(thresholds["min_levels_for_context"])
    entry_tolerance_pct = float(thresholds["entry_tolerance_pct"])
    break_cooldown_bars = int(thresholds["break_cooldown_bars"])
    rotation_max_tests = int(thresholds["rotation_max_tests"])
    rotation_volume_max_ratio = float(thresholds["rotation_volume_max_ratio"])
    recent_bounce_lookback_bars = int(thresholds["recent_bounce_lookback_bars"])
    require_recent_bounce_for_mr = bool(thresholds["require_recent_bounce_for_mr"])
    momentum_break_max_age_bars = int(thresholds["momentum_break_max_age_bars"])
    momentum_min_room_pct = float(thresholds["momentum_min_room_pct"])
    momentum_min_broken_ratio = float(thresholds["momentum_min_broken_ratio"])
    min_confluence_score = int(thresholds["min_confluence_score"])
    rvol_filter_enabled = bool(thresholds["rvol_filter_enabled"])
    rvol_min_threshold = float(thresholds["rvol_min_threshold"])
    rvol_strong_threshold = float(thresholds["rvol_strong_threshold"])
    adaptive_window_enabled = bool(thresholds["adaptive_window_enabled"])
    gap_analysis_enabled = bool(thresholds["gap_analysis_enabled"])
    gap_min_pct = float(thresholds["gap_min_pct"])
    gap_momentum_threshold_pct = float(thresholds["gap_momentum_threshold_pct"])
    bounce_conflict_buffer = int(thresholds["bounce_conflict_buffer"])

    current_rvol = _to_optional_float(market_activity.get("rvol"))
    level_context = build_level_context(
        levels_raw=levels_raw,
        current_price=current_price,
        entry_tolerance_pct=entry_tolerance_pct,
        current_rvol=current_rvol,
        rvol_strong_threshold=rvol_strong_threshold,
    )
    total_levels = int(level_context["total_levels"])
    active_levels = list(level_context["active_levels"])
    tested_levels_count = int(level_context["tested_levels_count"])
    broken_levels_count = int(level_context["broken_levels_count"])
    near_levels = list(level_context["near_levels"])
    near_tested_levels = list(level_context["near_tested_levels"])
    near_confluence_score = int(level_context["near_confluence_score"])
    rvol_confluence_bonus = int(level_context["rvol_confluence_bonus"])
    near_memory_levels_count = int(level_context["near_memory_levels_count"])
    next_resistance_above = level_context["next_resistance_above"]
    next_support_below = level_context["next_support_below"]
    broken_ratio = float(level_context["broken_ratio"])

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

    event_context = extract_recent_events_context(
        recent_events=recent_events,
        current_bar_index=current_bar_index,
        break_cooldown_bars=break_cooldown_bars,
        recent_bounce_lookback_bars=recent_bounce_lookback_bars,
        bounce_conflict_buffer=bounce_conflict_buffer,
        direction=direction,
    )
    latest_event = event_context["latest_event"]
    bars_since_latest_event = event_context["bars_since_latest_event"]
    recent_break_event = event_context["recent_break_event"]
    recent_aligned_bounce = event_context["recent_aligned_bounce"]

    room_to_next_opposite_level_pct: Optional[float] = None
    if direction > 0 and next_resistance_above is not None and current_price > 0.0:
        room_to_next_opposite_level_pct = (
            (float(next_resistance_above["price"]) - current_price) / current_price
        ) * 100.0
    elif direction < 0 and next_support_below is not None and current_price > 0.0:
        room_to_next_opposite_level_pct = (
            (current_price - float(next_support_below["price"])) / current_price
        ) * 100.0

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
        target_price_override, untested_obstacles_to_poc = (
            apply_strategy_specific_entry_quality_checks(
                strategy_key=strategy_key,
                direction=direction,
                signal=signal,
                current_price=current_price,
                current_bar_index=current_bar_index,
                config=config,
                active_levels=active_levels,
                near_tested_levels=near_tested_levels,
                near_confluence_score=near_confluence_score,
                min_confluence_score=min_confluence_score,
                price_outside_value_area=price_outside_value_area,
                value_area_position=value_area_position,
                va_near_edge=va_near_edge,
                poc_on_trade_side=poc_on_trade_side,
                recent_break_event=recent_break_event,
                recent_aligned_bounce=recent_aligned_bounce,
                require_recent_bounce_for_mr=require_recent_bounce_for_mr,
                poc_migration_bias=poc_migration_bias,
                gap_analysis_enabled=gap_analysis_enabled,
                gap_abs_pct=gap_abs_pct,
                gap_min_pct=gap_min_pct,
                gap_momentum_threshold_pct=gap_momentum_threshold_pct,
                gap_favor_side=gap_favor_side,
                gap_momentum_bias_side=gap_momentum_bias_side,
                effective_poc_price=effective_poc_price,
                latest_event=latest_event,
                momentum_break_max_age_bars=momentum_break_max_age_bars,
                broken_ratio=broken_ratio,
                momentum_min_broken_ratio=momentum_min_broken_ratio,
                room_to_next_opposite_level_pct=room_to_next_opposite_level_pct,
                momentum_min_room_pct=momentum_min_room_pct,
                opening_breakout_direction=opening_breakout_direction,
                rotation_max_tests=rotation_max_tests,
                rotation_volume_max_ratio=rotation_volume_max_ratio,
                rvol_filter_enabled=rvol_filter_enabled,
                current_rvol=current_rvol,
                checks=checks,
                reasons=reasons,
                soft_reasons=soft_reasons,
            )
        )
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
