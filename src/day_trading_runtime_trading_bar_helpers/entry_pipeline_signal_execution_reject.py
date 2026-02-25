"""Threshold/headwind rejection helpers for the runtime entry pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from ..day_trading_models import TradingSession
from .entry_actions import _apply_threshold_rejection_payload


def _apply_threshold_rejection_for_decision(
    *,
    session: TradingSession,
    timestamp: datetime,
    regime: Any,
    decision: Any,
    passed_trade_threshold: bool,
    effective_trade_threshold: float,
    threshold_used_reason: str,
    tod_boost: float,
    headwind_boost: float,
    headwind_metrics: Any,
    result: Dict[str, Any],
) -> None:
    """Populate threshold/headwind rejection payload for a non-executed positive decision."""

    _apply_threshold_rejection_payload(
        decision=decision,
        passed_trade_threshold=passed_trade_threshold,
        effective_trade_threshold=effective_trade_threshold,
        threshold_used_reason=threshold_used_reason,
        regime=regime,
        micro_regime=session.micro_regime,
        tod_boost=tod_boost,
        headwind_boost=headwind_boost,
        headwind_metrics=headwind_metrics,
        timestamp=timestamp,
        result=result,
    )
