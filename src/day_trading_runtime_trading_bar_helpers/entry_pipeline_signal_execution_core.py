"""Core signal execution/rejection stage helpers for the runtime entry pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from ..day_trading_models import BarData, TradingSession
from .entry_pipeline_signal_execution_confirmed import (
    _execute_confirmed_decision_signal,
)
from .entry_pipeline_signal_execution_reject import _apply_threshold_rejection_for_decision


def _execute_or_reject_decision_signal(
    self,
    *,
    session: TradingSession,
    bar: BarData,
    timestamp: datetime,
    current_price: float,
    current_bar_index: int,
    indicators: Dict[str, Any],
    flow_metrics: Dict[str, Any],
    regime: Any,
    golden_setup_payload: Dict[str, Any],
    decision_state: Dict[str, Any],
    result: Dict[str, Any],
) -> bool:
    """Run signal gates/queueing for executable decisions or threshold rejection payloads."""

    decision = decision_state["decision"]
    effective_trade_threshold = float(decision_state["effective_trade_threshold"])
    passed_trade_threshold = bool(decision_state["passed_trade_threshold"])
    tod_boost = float(decision_state["tod_boost"])
    headwind_boost = float(decision_state["headwind_boost"])
    headwind_metrics = decision_state["headwind_metrics"]
    required_confirming_sources = int(decision_state["required_confirming_sources"])
    threshold_used_reason = str(decision_state["threshold_used_reason"])

    if decision.execute and decision.signal and passed_trade_threshold:
        return _execute_confirmed_decision_signal(
            self,
            session=session,
            bar=bar,
            timestamp=timestamp,
            current_price=current_price,
            current_bar_index=current_bar_index,
            indicators=indicators,
            flow_metrics=flow_metrics,
            regime=regime,
            golden_setup_payload=golden_setup_payload,
            decision=decision,
            effective_trade_threshold=effective_trade_threshold,
            tod_boost=tod_boost,
            required_confirming_sources=required_confirming_sources,
            result=result,
        )

    if decision.combined_score > 0 and (not decision.execute or not passed_trade_threshold):
        _apply_threshold_rejection_for_decision(
            session=session,
            timestamp=timestamp,
            regime=regime,
            decision=decision,
            passed_trade_threshold=passed_trade_threshold,
            effective_trade_threshold=effective_trade_threshold,
            threshold_used_reason=threshold_used_reason,
            tod_boost=tod_boost,
            headwind_boost=headwind_boost,
            headwind_metrics=headwind_metrics,
            result=result,
        )

    return False
