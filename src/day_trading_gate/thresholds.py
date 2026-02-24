"""Threshold helpers for gate evaluation."""

from __future__ import annotations

from datetime import time
from typing import Any, Dict, Optional, Tuple

from ..exit_policy_engine import ExitPolicyEngine


def time_of_day_threshold_boost(bar_time: time, strategy_key: str = "") -> float:
    return ExitPolicyEngine.time_of_day_threshold_boost(bar_time, strategy_key=strategy_key)


def cross_asset_headwind_threshold_boost(
    cross_asset_state: Any,
    decision_direction: Optional[str],
    *,
    activation_score: float = 0.5,
    min_boost: float = 5.0,
    max_boost: float = 10.0,
    strategy_key: str = "",
) -> Tuple[float, Dict[str, Any]]:
    """
    Convert strong cross-asset headwind into an entry threshold boost.

    Boost activates only when:
    - index context is available
    - index momentum opposes the candidate direction
    - effective headwind score is above activation threshold

    Contrarian/MR strategies intentionally trade against the market,
    so the boost is capped at +2 (from +5..+10) to avoid blocking
    exactly the setups they are designed to capture.

    Output boost is bounded to [min_boost, max_boost], default +5..+10 points.
    """
    metrics: Dict[str, Any] = {
        "index_available": False,
        "decision_direction": str(decision_direction or "").lower(),
        "activation_score": float(activation_score),
        "min_boost": float(min_boost),
        "max_boost": float(max_boost),
        "headwind_score": 0.0,
        "effective_headwind_score": 0.0,
        "sector_relative": 0.0,
        "index_momentum_5": 0.0,
        "index_opposes_direction": False,
        "boost_points": 0.0,
        "applied": False,
    }

    direction = metrics["decision_direction"]
    if direction not in {"bullish", "bearish"}:
        metrics["reason"] = "direction_unavailable"
        return 0.0, metrics

    if cross_asset_state is None:
        metrics["reason"] = "cross_asset_unavailable"
        return 0.0, metrics

    index_available = bool(getattr(cross_asset_state, "index_available", False))
    metrics["index_available"] = index_available
    if not index_available:
        metrics["reason"] = "index_not_available"
        return 0.0, metrics

    headwind_score = max(
        0.0,
        min(1.0, float(getattr(cross_asset_state, "headwind_score", 0.0) or 0.0)),
    )
    sector_relative = float(getattr(cross_asset_state, "sector_relative", 0.0) or 0.0)
    idx_mom_5 = float(getattr(cross_asset_state, "index_momentum_5", 0.0) or 0.0)

    metrics["headwind_score"] = headwind_score
    metrics["sector_relative"] = sector_relative
    metrics["index_momentum_5"] = idx_mom_5

    index_opposes_direction = (
        (direction == "bullish" and idx_mom_5 < 0.0)
        or (direction == "bearish" and idx_mom_5 > 0.0)
    )
    metrics["index_opposes_direction"] = index_opposes_direction
    if not index_opposes_direction:
        metrics["reason"] = "index_supports_direction"
        return 0.0, metrics

    directional_sector_relative = sector_relative if direction == "bullish" else -sector_relative
    sector_penalty = min(0.15, max(0.0, -directional_sector_relative))
    effective_headwind = min(1.0, headwind_score + sector_penalty)
    metrics["effective_headwind_score"] = round(effective_headwind, 4)

    activation = max(0.0, min(0.95, float(activation_score)))
    if effective_headwind <= activation:
        metrics["reason"] = "below_activation_threshold"
        return 0.0, metrics

    boost_floor = max(0.0, float(min_boost))
    boost_ceiling = max(boost_floor, float(max_boost))
    span = max(1e-9, 1.0 - activation)
    normalized = min(1.0, max(0.0, (effective_headwind - activation) / span))
    boost = boost_floor + normalized * (boost_ceiling - boost_floor)

    _sk = str(strategy_key or "").strip().lower()
    _contrarian = {"mean_reversion", "absorption_reversal", "rotation"}
    if _sk in _contrarian:
        boost = min(boost, 2.0)
        metrics["contrarian_headwind_cap"] = True

    boost = round(boost, 4)

    metrics["boost_points"] = boost
    metrics["applied"] = boost > 0.0
    return boost, metrics
