"""Confirmed-signal execution path helpers for the runtime entry pipeline."""

from __future__ import annotations

from datetime import datetime, time
from typing import Any, Dict

from ..day_trading_models import BarData, TradingSession
from ..strategies.base_strategy import Regime, SignalType
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


def _parse_hhmm_time(raw_value: Any, fallback: time) -> time:
    token = str(raw_value or "").strip()
    if not token:
        return fallback
    try:
        parts = token.split(":")
        if len(parts) != 2:
            return fallback
        hour = int(parts[0])
        minute = int(parts[1])
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return time(hour, minute)
    except (TypeError, ValueError):
        return fallback
    return fallback


def _apply_pullback_quality_gate(
    self,
    *,
    session: TradingSession,
    signal: Any,
    flow_metrics: Dict[str, Any],
    decision: Any,
    effective_trade_threshold: float,
    regime: Any,
    tod_boost: float,
    timestamp: datetime,
    result: Dict[str, Any],
) -> bool:
    strategy_key = self._canonical_strategy_key(signal.strategy_name or "")
    if strategy_key != "pullback":
        return False

    config = getattr(session, "config", None)
    market_time = self._to_market_time(timestamp).time()
    window_enabled = bool(getattr(config, "pullback_morning_window_enabled", True))
    window_start = _parse_hhmm_time(
        getattr(config, "pullback_entry_start_time", "10:00"),
        fallback=time(10, 0),
    )
    window_end = _parse_hhmm_time(
        getattr(config, "pullback_entry_end_time", "11:30"),
        fallback=time(11, 30),
    )

    def _reject(reason: str, **details: Any) -> bool:
        result["action"] = "pullback_quality_filtered"
        result["reason"] = reason
        result["signal_rejected"] = {
            "gate": "pullback_quality",
            "strategy": signal.strategy_name,
            "signal_type": signal.signal_type.value if signal.signal_type else None,
            "confidence": round(signal.confidence, 1),
            "combined_score": round(float(decision.combined_score), 1),
            "threshold_used": float(effective_trade_threshold),
            "regime": regime.value if hasattr(regime, "value") else str(regime or ""),
            "micro_regime": session.micro_regime,
            "tod_threshold_boost": float(tod_boost),
            "timestamp": timestamp.isoformat(),
            "details": details,
        }
        return True

    if window_enabled and not (window_start <= market_time <= window_end):
        return _reject(
            "pullback_outside_morning_window",
            market_time=market_time.isoformat(),
            window_start=window_start.isoformat(),
            window_end=window_end.isoformat(),
        )

    if bool(getattr(config, "pullback_block_choppy_macro", True)) and regime == Regime.CHOPPY:
        return _reject(
            "pullback_blocked_choppy_macro",
            macro_regime="CHOPPY",
        )

    blocked_micro_raw = getattr(
        config,
        "pullback_blocked_micro_regimes",
        ("CHOPPY", "TRANSITION"),
    )
    blocked_micro = {
        str(token).strip().upper()
        for token in (blocked_micro_raw if isinstance(blocked_micro_raw, (list, tuple, set)) else [])
        if str(token).strip()
    }
    micro_regime = str(session.micro_regime or "").strip().upper()
    if blocked_micro and micro_regime in blocked_micro:
        return _reject(
            "pullback_blocked_micro_regime",
            micro_regime=micro_regime,
            blocked_micro_regimes=sorted(blocked_micro),
        )

    try:
        trend_efficiency = float(flow_metrics.get("price_trend_efficiency", 0.0) or 0.0)
    except (TypeError, ValueError):
        trend_efficiency = 0.0
    try:
        min_trend_efficiency = float(
            getattr(config, "pullback_min_price_trend_efficiency", 0.15) or 0.15
        )
    except (TypeError, ValueError):
        min_trend_efficiency = 0.15
    min_trend_efficiency = max(0.0, min(1.0, min_trend_efficiency))
    if trend_efficiency < min_trend_efficiency:
        return _reject(
            "pullback_trend_efficiency_too_low",
            trend_efficiency=round(trend_efficiency, 6),
            min_required=round(min_trend_efficiency, 6),
        )

    require_poc_on_side = bool(getattr(config, "pullback_require_poc_on_trade_side", True))
    level_context = result.get("level_context") if isinstance(result.get("level_context"), dict) else {}
    volume_profile = (
        level_context.get("volume_profile")
        if isinstance(level_context.get("volume_profile"), dict)
        else {}
    )
    effective_poc_price = volume_profile.get("effective_poc_price")
    poc_on_trade_side = bool(volume_profile.get("poc_on_trade_side", False))
    if require_poc_on_side and effective_poc_price is not None and not poc_on_trade_side:
        return _reject(
            "pullback_requires_poc_on_trade_side",
            effective_poc_price=effective_poc_price,
            poc_on_trade_side=poc_on_trade_side,
        )

    return False


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
    if _apply_pullback_quality_gate(
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
