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


def _apply_mu_choppy_regime_gate(
    self,
    *,
    session: TradingSession,
    regime: Optional[Regime],
    timestamp: datetime,
    golden_setup_payload: Dict[str, Any],
    result: Dict[str, Any],
) -> bool:
    """Apply MU choppy hard-block gate with TCBBO and golden-setup bypass overrides."""

    mu_choppy_filter_enabled = self._is_mu_choppy_filter_enabled(
        session.ticker,
        session=session,
    )
    tcbbo_regime_override = self.gate_engine.tcbbo_directional_override(session)
    if bool(tcbbo_regime_override.get("enabled", False)):
        result["tcbbo_regime_override"] = dict(tcbbo_regime_override)

    golden_bypass_choppy = bool(golden_setup_payload.get("bypass_choppy", False))
    if golden_bypass_choppy:
        result["golden_setup_choppy_bypass"] = True

    if not mu_choppy_filter_enabled or not self._is_mu_choppy_blocked(session, regime):
        return False
    if bool(tcbbo_regime_override.get("applied", False)) or golden_bypass_choppy:
        return False

    result["action"] = "regime_filter"
    result["reason"] = "MU choppy regime filter active"
    result["signal_rejected"] = {
        "gate": "mu_choppy_filter",
        "ticker": session.ticker,
        "regime": regime.value if regime else None,
        "micro_regime": session.micro_regime,
        "timestamp": timestamp.isoformat(),
        "mu_choppy_hard_block_enabled": bool(mu_choppy_filter_enabled),
        "tcbbo_regime_override": dict(tcbbo_regime_override),
        "golden_setup": dict(golden_setup_payload),
    }
    return True


def _apply_l2_confirmation_gate(
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
    """Apply L2 confirmation gate and weak-L2 fast break-even override."""

    l2_strategy_key = str(signal.strategy_name or "").strip().lower()
    l2_relaxed_strategies = {"mean_reversion", "rotation", "vwap_magnet", "volumeprofile"}
    l2_relaxed = l2_strategy_key in l2_relaxed_strategies

    l2_passed, l2_metrics = self.gate_engine.passes_l2_confirmation(
        session,
        signal,
        flow_metrics=flow_metrics,
    )
    result["l2_confirmation"] = l2_metrics

    if l2_relaxed and not l2_passed:
        l2_has_coverage = bool(l2_metrics.get("has_l2_coverage", False))
        if l2_has_coverage:
            l2_passed = True
            l2_metrics["relaxed_for_strategy"] = l2_strategy_key
            signal.metadata.setdefault("l2_relaxed_entry", True)

    if l2_passed:
        return False, l2_metrics

    weak_l2_enabled = bool(getattr(session.config, "weak_l2_fast_break_even_enabled", False))
    hard_l2_block = bool(l2_metrics.get("hard_block", False))
    l2_reason = str(l2_metrics.get("reason", "") or "")
    l2_aggression = abs(
        float(
            l2_metrics.get(
                "signed_aggression_avg",
                l2_metrics.get("signed_aggression", 0.0),
            )
            or 0.0
        )
    )
    weak_l2_threshold = float(getattr(session.config, "weak_l2_aggression_threshold", 0.05))
    if (
        (not hard_l2_block)
        and weak_l2_enabled
        and l2_reason == "l2_confirmation_failed"
        and l2_aggression <= weak_l2_threshold
    ):
        override_hold = int(getattr(session.config, "weak_l2_break_even_min_hold_bars", 2))
        signal.metadata["weak_l2_entry"] = True
        signal.metadata["weak_l2_break_even_override"] = {
            "break_even_min_hold_bars": override_hold,
            "original_aggression": round(l2_aggression, 4),
            "threshold": weak_l2_threshold,
        }
        result["weak_l2_entry"] = True
        result["weak_l2_override"] = signal.metadata["weak_l2_break_even_override"]
        return False, l2_metrics

    result["action"] = "l2_filtered"
    result["reason"] = l2_metrics.get("reason", "l2_confirmation_failed")
    result["signal_rejected"] = {
        "gate": "l2_confirmation",
        "strategy": signal.strategy_name,
        "signal_type": signal.signal_type.value if signal.signal_type else None,
        "confidence": round(signal.confidence, 1),
        "combined_score": round(decision.combined_score, 1),
        "threshold_used": effective_trade_threshold,
        "regime": regime.value if regime else None,
        "micro_regime": session.micro_regime,
        "l2_metrics": l2_metrics,
        "tod_threshold_boost": tod_boost,
        "timestamp": timestamp.isoformat(),
    }
    return True, l2_metrics


def _apply_tcbbo_confirmation_gate(
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
) -> bool:
    """Apply TCBBO confirmation gate and aligned confidence boost."""

    tcbbo_passed, tcbbo_metrics = self.gate_engine.passes_tcbbo_confirmation(session, signal)
    result["tcbbo_confirmation"] = tcbbo_metrics

    if not tcbbo_passed:
        result["action"] = "tcbbo_filtered"
        result["reason"] = tcbbo_metrics.get("reason", "tcbbo_confirmation_failed")
        result["signal_rejected"] = {
            "gate": "tcbbo_confirmation",
            "strategy": signal.strategy_name,
            "signal_type": signal.signal_type.value if signal.signal_type else None,
            "confidence": round(signal.confidence, 1),
            "combined_score": round(decision.combined_score, 1),
            "threshold_used": effective_trade_threshold,
            "regime": regime.value if regime else None,
            "micro_regime": session.micro_regime,
            "tcbbo_metrics": tcbbo_metrics,
            "tod_threshold_boost": tod_boost,
            "timestamp": timestamp.isoformat(),
        }
        return True

    tcbbo_boost = float(tcbbo_metrics.get("confidence_boost", 0.0))
    if tcbbo_boost > 0:
        signal.confidence = min(100.0, signal.confidence + tcbbo_boost)
        signal.metadata.setdefault("tcbbo_confirmation", tcbbo_metrics)
    return False

