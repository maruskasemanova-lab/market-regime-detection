"""Strategy-specific rule branches for intraday level entry-quality gate."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..strategies.base_strategy import Signal
from ..trading_config import TradingConfig
from ..day_trading_runtime_sweep import (
    to_optional_float as _to_optional_float_impl,
)


def _to_optional_float(value: Any) -> Optional[float]:
    return _to_optional_float_impl(value)


def _event_direction(event: Optional[Dict[str, Any]]) -> int:
    if not isinstance(event, dict):
        return 0
    direction = str(event.get("direction", "")).strip().lower()
    if direction == "bullish":
        return 1
    if direction == "bearish":
        return -1
    return 0


def apply_strategy_specific_entry_quality_checks(
    *,
    strategy_key: str,
    direction: int,
    signal: Signal,
    current_price: float,
    current_bar_index: int,
    config: TradingConfig,
    active_levels: List[Dict[str, Any]],
    near_tested_levels: List[Dict[str, Any]],
    near_confluence_score: int,
    min_confluence_score: int,
    price_outside_value_area: bool,
    value_area_position: str,
    va_near_edge: bool,
    poc_on_trade_side: bool,
    recent_break_event: Optional[Dict[str, Any]],
    recent_aligned_bounce: Optional[Dict[str, Any]],
    require_recent_bounce_for_mr: bool,
    poc_migration_bias: str,
    gap_analysis_enabled: bool,
    gap_abs_pct: float,
    gap_min_pct: float,
    gap_momentum_threshold_pct: float,
    gap_favor_side: str,
    gap_momentum_bias_side: str,
    effective_poc_price: Optional[float],
    latest_event: Optional[Dict[str, Any]],
    momentum_break_max_age_bars: int,
    broken_ratio: float,
    momentum_min_broken_ratio: float,
    room_to_next_opposite_level_pct: Optional[float],
    momentum_min_room_pct: float,
    opening_breakout_direction: str,
    rotation_max_tests: int,
    rotation_volume_max_ratio: float,
    rvol_filter_enabled: bool,
    current_rvol: Optional[float],
    checks: Dict[str, bool],
    reasons: List[str],
    soft_reasons: List[str],
) -> Tuple[Optional[float], List[Dict[str, Any]]]:
    """Apply strategy-specific checks and mutate checks/reasons in place."""
    target_price_override: Optional[float] = None
    untested_obstacles_to_poc: List[Dict[str, Any]] = []

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
        if (
            gap_analysis_enabled
            and gap_abs_pct >= gap_min_pct
            and gap_abs_pct < gap_momentum_threshold_pct
        ):
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
                    if kind == "resistance" and current_price < level_price < effective_poc_price:
                        untested_obstacles_to_poc.append(level)
                else:
                    if kind == "support" and effective_poc_price < level_price < current_price:
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
        if (
            gap_analysis_enabled
            and gap_abs_pct >= gap_min_pct
            and gap_abs_pct < gap_momentum_threshold_pct
        ):
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
        checks["outside_value_area"] = price_outside_value_area or va_near_edge
        checks["poc_on_trade_side"] = poc_on_trade_side
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
        if (
            gap_analysis_enabled
            and gap_abs_pct >= gap_min_pct
            and gap_abs_pct < gap_momentum_threshold_pct
        ):
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
        checks["near_tested_level"] = len(near_tested_levels) >= 1
        checks["minimum_confluence_score"] = near_confluence_score >= 1
        checks["poc_on_trade_side"] = poc_on_trade_side or effective_poc_price is None

        pullback_rvol_min = max(
            0.0,
            float(getattr(config, "intraday_levels_pullback_rvol_min_threshold", 0.30)),
        )
        checks["rvol_minimum"] = bool(
            not rvol_filter_enabled
            or (current_rvol is not None and current_rvol >= pullback_rvol_min)
        )

        if not checks["near_tested_level"]:
            soft_reasons.append("pullback_prefers_tested_level_near_entry")
        if not checks["minimum_confluence_score"]:
            soft_reasons.append("pullback_prefers_confluence")

        if rvol_filter_enabled and checks.get("rvol_available", False) and not checks["rvol_minimum"]:
            if "rvol_below_min_threshold" not in reasons:
                reasons.append("rvol_below_pullback_min_threshold")
        elif "rvol_below_min_threshold" in reasons and checks["rvol_minimum"]:
            reasons.remove("rvol_below_min_threshold")

    else:
        checks["strategy_specific_gate_applied"] = False

    return target_price_override, untested_obstacles_to_poc


__all__ = ["apply_strategy_specific_entry_quality_checks"]
