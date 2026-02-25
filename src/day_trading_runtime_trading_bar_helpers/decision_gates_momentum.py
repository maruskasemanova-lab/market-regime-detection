"""Decision-gate helpers for runtime trading-bar processing."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, Optional

from ..day_trading_models import TradingSession
from ..day_trading_runtime_portfolio import (
    cooldown_bars_remaining as _cooldown_bars_remaining_impl,
)
from ..strategies.base_strategy import Regime, Signal, SignalType


def _apply_momentum_flow_confirmation_gate(
    self,
    *,
    session: TradingSession,
    signal: Signal,
    decision: SimpleNamespace,
    effective_trade_threshold: float,
    regime: Optional[Regime],
    tod_boost: float,
    timestamp: datetime,
    result: Dict[str, Any],
) -> tuple[bool, Dict[str, Any]]:
    """Apply momentum-flow delta divergence confirmation gate."""

    momentum_flow_passed, momentum_flow_metrics = self._passes_momentum_flow_delta_confirmation(
        signal
    )
    result["momentum_flow_confirmation"] = momentum_flow_metrics

    if momentum_flow_passed:
        return False, momentum_flow_metrics

    result["action"] = "momentum_flow_filtered"
    result["reason"] = momentum_flow_metrics.get(
        "reason",
        "momentum_flow_delta_divergence_required",
    )
    result["signal_rejected"] = {
        "gate": "momentum_flow_delta_divergence",
        "strategy": signal.strategy_name,
        "signal_type": signal.signal_type.value if signal.signal_type else None,
        "confidence": round(signal.confidence, 1),
        "combined_score": round(decision.combined_score, 1),
        "threshold_used": effective_trade_threshold,
        "regime": regime.value if regime else None,
        "micro_regime": session.micro_regime,
        "momentum_flow_confirmation": momentum_flow_metrics,
        "tod_threshold_boost": tod_boost,
        "timestamp": timestamp.isoformat(),
    }
    return True, momentum_flow_metrics


def _apply_momentum_diversification_gate(
    self,
    *,
    session: TradingSession,
    signal: Signal,
    flow_metrics: Dict[str, Any],
    decision: SimpleNamespace,
    effective_trade_threshold: float,
    regime: Optional[Regime],
    tod_boost: float,
    timestamp: datetime,
    result: Dict[str, Any],
) -> tuple[bool, Dict[str, Any]]:
    """Apply momentum diversification gate."""

    momentum_diversification_passed, momentum_diversification_metrics = (
        self.gate_engine.passes_momentum_diversification_gate(
            session=session,
            signal=signal,
            flow_metrics=flow_metrics,
        )
    )
    result["momentum_diversification"] = momentum_diversification_metrics

    if momentum_diversification_passed:
        return False, momentum_diversification_metrics

    result["action"] = "momentum_diversification_filtered"
    result["reason"] = momentum_diversification_metrics.get(
        "reason",
        "momentum_diversification_gate_failed",
    )
    result["signal_rejected"] = {
        "gate": "momentum_diversification",
        "strategy": signal.strategy_name,
        "signal_type": signal.signal_type.value if signal.signal_type else None,
        "confidence": round(signal.confidence, 1),
        "combined_score": round(decision.combined_score, 1),
        "threshold_used": effective_trade_threshold,
        "regime": regime.value if regime else None,
        "micro_regime": session.micro_regime,
        "momentum_diversification": momentum_diversification_metrics,
        "tod_threshold_boost": tod_boost,
        "timestamp": timestamp.isoformat(),
    }
    return True, momentum_diversification_metrics
