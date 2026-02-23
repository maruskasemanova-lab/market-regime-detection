"""Liquidity sweep and shared numeric helpers for runtime processing."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from .day_trading_models import TradingSession
from .intraday_levels import ensure_intraday_levels_state
from .trading_config import TradingConfig


def to_optional_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_level_distance_pct(price: float, level_price: float) -> Optional[float]:
    if price <= 0.0 or level_price <= 0.0:
        return None
    return abs((level_price - price) / price) * 100.0


def order_flow_metadata_snapshot(flow_metrics: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(flow_metrics, dict):
        return {}
    payload: Dict[str, Any] = {}
    for key in (
        "has_l2_coverage",
        "flow_score",
        "signed_aggression",
        "l2_aggression_z",
        "directional_consistency",
        "imbalance_avg",
        "sweep_intensity",
        "book_pressure_avg",
        "l2_book_pressure_z",
        "book_pressure_trend",
        "absorption_rate",
        "delta_price_divergence",
        "delta_zscore",
        "price_change_pct",
        "delta_acceleration",
        "large_trader_activity",
        "vwap_execution_flow",
    ):
        if key not in flow_metrics:
            continue
        value = flow_metrics.get(key)
        if isinstance(value, bool):
            payload[key] = bool(value)
        elif isinstance(value, (int, float)):
            payload[key] = float(value)
    return payload


def feature_vector_value(fv: Any, field: str, default: float = 0.0) -> float:
    if fv is None:
        return float(default)
    raw = None
    if isinstance(fv, dict):
        raw = fv.get(field)
    else:
        raw = getattr(fv, field, None)
    if raw is None:
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return value


def liquidity_sweep_nearest_level(
    *,
    session: TradingSession,
    current_price: float,
) -> Dict[str, Any]:
    state = ensure_intraday_levels_state(getattr(session, "intraday_levels_state", {}))
    levels = state.get("levels")
    if not isinstance(levels, list):
        return {"has_level": False}

    config = (
        session.config
        if isinstance(getattr(session, "config", None), TradingConfig)
        else TradingConfig()
    )
    tolerance_pct = max(
        0.01,
        float(getattr(config, "intraday_levels_entry_tolerance_pct", 0.10)),
    )

    nearest: Optional[Dict[str, Any]] = None
    for row in levels:
        if not isinstance(row, dict):
            continue
        level_price = to_optional_float(row.get("price"))
        if level_price is None or level_price <= 0.0:
            continue
        if bool(row.get("broken", False)):
            continue
        kind = str(row.get("kind", "")).strip().lower()
        if kind not in {"support", "resistance"}:
            kind = "support" if level_price <= current_price else "resistance"
        distance_pct = safe_level_distance_pct(current_price, level_price)
        if distance_pct is None or distance_pct > tolerance_pct:
            continue
        candidate = {
            "has_level": True,
            "level_price": float(level_price),
            "level_kind": kind,
            "level_source": str(row.get("source", "")),
            "distance_pct": float(distance_pct),
        }
        if nearest is None or candidate["distance_pct"] < nearest["distance_pct"]:
            nearest = candidate

    return nearest or {"has_level": False}


def runtime_detect_liquidity_sweep(
    *,
    session: TradingSession,
    current_price: float,
    fv: Any,
    flow_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = (
        session.config
        if isinstance(getattr(session, "config", None), TradingConfig)
        else TradingConfig()
    )

    if not bool(getattr(config, "liquidity_sweep_detection_enabled", False)):
        return {
            "enabled": False,
            "sweep_detected": False,
            "reason": "disabled",
        }

    has_l2_coverage: Optional[bool] = None
    if isinstance(flow_metrics, dict) and "has_l2_coverage" in flow_metrics:
        has_l2_coverage = bool(flow_metrics.get("has_l2_coverage", False))
    if has_l2_coverage is False:
        return {
            "enabled": True,
            "sweep_detected": False,
            "reason": "insufficient_l2_coverage",
            "has_l2_coverage": False,
        }

    level_context = liquidity_sweep_nearest_level(
        session=session,
        current_price=current_price,
    )
    if not bool(level_context.get("has_level", False)):
        return {
            "enabled": True,
            "sweep_detected": False,
            "reason": "no_nearby_level",
        }

    aggression_z = feature_vector_value(fv, "l2_aggression_z", 0.0)
    book_pressure_z = feature_vector_value(fv, "l2_book_pressure_z", 0.0)
    trend_slope = feature_vector_value(fv, "tf5_trend_slope", 0.0)
    min_aggression_z = float(getattr(config, "sweep_min_aggression_z", -2.0))
    min_book_pressure_z = float(getattr(config, "sweep_min_book_pressure_z", 1.5))
    max_price_change_pct = max(
        0.0,
        float(getattr(config, "sweep_max_price_change_pct", 0.05)),
    )

    bullish_divergence = (
        aggression_z < min_aggression_z
        and book_pressure_z > min_book_pressure_z
    )
    bearish_divergence = (
        aggression_z > abs(min_aggression_z)
        and book_pressure_z < -abs(min_book_pressure_z)
    )
    slope_ok = abs(trend_slope) <= max_price_change_pct
    level_kind = str(level_context.get("level_kind", "")).strip().lower()

    signal_direction: Optional[str] = None
    if level_kind == "support" and bullish_divergence and slope_ok:
        signal_direction = "long"
    elif level_kind == "resistance" and bearish_divergence and slope_ok:
        signal_direction = "short"

    if signal_direction is None:
        return {
            "enabled": True,
            "sweep_detected": False,
            "reason": "divergence_not_met",
            "l2_aggression_z": aggression_z,
            "l2_book_pressure_z": book_pressure_z,
            "tf5_trend_slope": trend_slope,
            "level_context": level_context,
        }

    detected_bar_index = max(0, len(session.bars) - 1)
    sweep_context = {
        "direction": signal_direction,
        "level_price": float(level_context.get("level_price", current_price) or current_price),
        "level_kind": level_kind,
        "level_source": str(level_context.get("level_source", "")),
        "distance_pct": float(level_context.get("distance_pct", 0.0) or 0.0),
        "detected_bar_index": detected_bar_index,
        "l2_aggression_z": aggression_z,
        "l2_book_pressure_z": book_pressure_z,
        "tf5_trend_slope": trend_slope,
    }
    session.potential_sweep_active = True
    session.potential_sweep_context = dict(sweep_context)
    return {
        "enabled": True,
        "sweep_detected": True,
        "reason": "divergence_detected",
        **sweep_context,
    }


def resolve_liquidity_sweep_confirmation(
    *,
    session: TradingSession,
    current_bar_index: int,
    current_price: float,
    flow_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    if not bool(session.potential_sweep_active):
        return {"active": False, "confirmed": False, "reason": "inactive"}
    context = (
        dict(session.potential_sweep_context)
        if isinstance(session.potential_sweep_context, dict)
        else {}
    )
    direction = str(context.get("direction", "long")).strip().lower()
    level_price = to_optional_float(context.get("level_price"))
    if level_price is None or level_price <= 0.0:
        session.potential_sweep_active = False
        session.potential_sweep_context = {}
        return {"active": False, "confirmed": False, "reason": "invalid_level"}

    detected_bar_index = int(to_optional_float(context.get("detected_bar_index")) or current_bar_index)
    age_bars = max(0, current_bar_index - detected_bar_index)
    max_confirmation_bars = 6
    if age_bars > max_confirmation_bars:
        session.potential_sweep_active = False
        session.potential_sweep_context = {}
        return {
            "active": False,
            "confirmed": False,
            "reason": "expired",
            "age_bars": age_bars,
            "max_confirmation_bars": max_confirmation_bars,
        }

    signed_aggression = float(flow_metrics.get("signed_aggression", 0.0) or 0.0)
    if direction == "short":
        flow_flip = signed_aggression < 0.0
        reclaim_ok = current_price < level_price
    else:
        flow_flip = signed_aggression > 0.0
        reclaim_ok = current_price > level_price
        direction = "long"

    confirmed = bool(flow_flip and reclaim_ok)
    payload = {
        "active": True,
        "confirmed": confirmed,
        "direction": direction,
        "level_price": float(level_price),
        "age_bars": age_bars,
        "signed_aggression": signed_aggression,
        "flow_flip": bool(flow_flip),
        "price_reclaimed_level": bool(reclaim_ok),
    }
    if confirmed:
        session.potential_sweep_active = False
        session.potential_sweep_context = {}
        payload["reason"] = "confirmed"
    else:
        payload["reason"] = "awaiting_confirmation"
    return payload
