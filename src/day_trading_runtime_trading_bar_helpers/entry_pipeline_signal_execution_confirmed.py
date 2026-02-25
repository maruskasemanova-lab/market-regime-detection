"""Confirmed-signal execution path helpers for the runtime entry pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from ..day_trading_models import BarData, TradingSession
from ..strategies.base_strategy import SignalType
from .decision_gates import (
    _apply_confirming_sources_gate,
    _apply_golden_setup_signal_adjustments,
    _apply_l2_confirmation_gate,
    _apply_momentum_diversification_gate,
    _apply_momentum_flow_confirmation_gate,
    _apply_tcbbo_confirmation_gate,
)
from .entry_actions import (
    _apply_custom_entry_formula_gate,
    _apply_intraday_levels_entry_quality_gate,
    _enrich_signal_metadata_for_entry_pipeline,
    _publish_signal_candidate_payload,
    _queue_signal_for_next_bar_with_cost_gate,
)


def _execute_confirmed_decision_signal(
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
    decision: Any,
    effective_trade_threshold: float,
    tod_boost: float,
    required_confirming_sources: int,
    result: Dict[str, Any],
) -> bool:
    """Run confirmation gates and queueing for an executable threshold-passing decision."""

    signal = decision.signal
    l2_blocked, l2_metrics = _apply_l2_confirmation_gate(
        self,
        session=session,
        signal=signal,
        flow_metrics=flow_metrics,
        decision=decision,
        effective_trade_threshold=effective_trade_threshold,
        regime=regime,
        tod_boost=tod_boost,
        timestamp=timestamp,
        result=result,
    )
    if l2_blocked:
        return True

    if _apply_tcbbo_confirmation_gate(
        self,
        session=session,
        signal=signal,
        decision=decision,
        effective_trade_threshold=effective_trade_threshold,
        regime=regime,
        tod_boost=tod_boost,
        timestamp=timestamp,
        result=result,
    ):
        return True

    _apply_golden_setup_signal_adjustments(
        signal=signal,
        golden_setup_payload=golden_setup_payload,
        result=result,
    )

    if _apply_confirming_sources_gate(
        self,
        session=session,
        signal=signal,
        decision=decision,
        effective_trade_threshold=effective_trade_threshold,
        regime=regime,
        tod_boost=tod_boost,
        timestamp=timestamp,
        required_confirming_sources=required_confirming_sources,
        result=result,
    ):
        return True

    momentum_flow_blocked, momentum_flow_metrics = _apply_momentum_flow_confirmation_gate(
        self,
        session=session,
        signal=signal,
        decision=decision,
        effective_trade_threshold=effective_trade_threshold,
        regime=regime,
        tod_boost=tod_boost,
        timestamp=timestamp,
        result=result,
    )
    if momentum_flow_blocked:
        return True

    (
        momentum_diversification_blocked,
        momentum_diversification_metrics,
    ) = _apply_momentum_diversification_gate(
        self,
        session=session,
        signal=signal,
        flow_metrics=flow_metrics,
        decision=decision,
        effective_trade_threshold=effective_trade_threshold,
        regime=regime,
        tod_boost=tod_boost,
        timestamp=timestamp,
        result=result,
    )
    if momentum_diversification_blocked:
        return True

    _enrich_signal_metadata_for_entry_pipeline(
        session=session,
        signal=signal,
        flow_metrics=flow_metrics,
        l2_metrics=l2_metrics,
        momentum_flow_metrics=momentum_flow_metrics,
        momentum_diversification_metrics=momentum_diversification_metrics,
        regime=regime,
        result=result,
    )
    if _apply_intraday_levels_entry_quality_gate(
        self,
        session=session,
        signal=signal,
        current_price=current_price,
        current_bar_index=current_bar_index,
        decision=decision,
        effective_trade_threshold=effective_trade_threshold,
        regime=regime,
        tod_boost=tod_boost,
        timestamp=timestamp,
        result=result,
    ):
        return True
    if _apply_custom_entry_formula_gate(
        self,
        session=session,
        signal=signal,
        bar=bar,
        indicators=indicators,
        flow_metrics=flow_metrics,
        current_bar_index=current_bar_index,
        decision=decision,
        effective_trade_threshold=effective_trade_threshold,
        regime=regime,
        timestamp=timestamp,
        result=result,
    ):
        return True
    _publish_signal_candidate_payload(
        self,
        session=session,
        signal=signal,
        result=result,
    )

    if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
        if _queue_signal_for_next_bar_with_cost_gate(
            session=session,
            signal=signal,
            decision=decision,
            effective_trade_threshold=effective_trade_threshold,
            current_bar_index=current_bar_index,
            timestamp=timestamp,
            result=result,
        ):
            return True
    return False
