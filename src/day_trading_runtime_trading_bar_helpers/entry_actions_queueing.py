"""Entry-stage action helpers for runtime trading-bar processing."""

from __future__ import annotations

from datetime import datetime
import logging
from types import SimpleNamespace
from typing import Any, Dict, Optional

from ..day_trading_models import BarData, TradingSession
from ..day_trading_runtime_entry_quality import (
    runtime_evaluate_intraday_levels_entry_quality as _runtime_evaluate_intraday_levels_entry_quality_impl,
)
from ..day_trading_runtime_sweep import (
    order_flow_metadata_snapshot as _order_flow_metadata_snapshot_impl,
    to_optional_float as _to_optional_float_impl,
)
from ..strategies.base_strategy import Regime, Signal

logger = logging.getLogger(__name__)

def _queue_signal_for_next_bar_with_cost_gate(
    *,
    session: TradingSession,
    signal: Signal,
    decision: SimpleNamespace,
    effective_trade_threshold: float,
    current_bar_index: int,
    timestamp: datetime,
    result: Dict[str, Any],
) -> bool:
    """Queue signal for next-bar execution, with cost-aware rejection gate."""

    logger.debug(
        "Queue candidate before cost filter: bar=%s ts=%s strategy=%s stop=%s price=%s",
        current_bar_index,
        timestamp.isoformat(),
        signal.strategy_name,
        getattr(signal, "stop_loss", 0),
        signal.price,
    )

    if signal.stop_loss and signal.stop_loss > 0:
        risk_pct = abs(signal.stop_loss - signal.price) / signal.price * 100
        cfg = getattr(session, "config", None)
        min_risk_vs_costs = float(getattr(cfg, "cost_gate_risk_multiplier", 5.0) or 5.0)
        cost_pct = float(getattr(cfg, "cost_gate_cost_pct", 0.02) or 0.02)
        cost_strat = str(signal.strategy_name or "").strip().lower()
        tight_sl_strats = {"rotation", "mean_reversion", "vwap_magnet", "volumeprofile"}
        if cost_strat in tight_sl_strats:
            min_risk_vs_costs = min(min_risk_vs_costs, 3.0)
        min_required = min_risk_vs_costs * cost_pct
        if risk_pct < min_required:
            logger.debug(
                "Signal rejected by cost-aware gate: risk_pct=%.4f min_required=%.4f ts=%s",
                risk_pct,
                min_required,
                timestamp.isoformat(),
            )
            result["signal_rejected"] = {
                "gate": "cost_aware",
                "risk_pct": round(risk_pct, 4),
                "min_required": round(min_required, 4),
                "reasoning": f"Risk {risk_pct:.3f}% < {min_risk_vs_costs}× costs",
            }
            return True

    signal.metadata["ev_margin"] = round(
        float(decision.combined_score) - float(effective_trade_threshold),
        2,
    )
    logger.debug(
        "Signal queued: bar=%s ts=%s strategy=%s",
        current_bar_index,
        timestamp.isoformat(),
        signal.strategy_name,
    )
    session.pending_signal = signal
    session.pending_signal_bar_index = current_bar_index
    result["action"] = "signal_queued"
    result["queued_for_next_bar"] = True

    cand_diag = signal.metadata.get("candidate_diagnostics", {})
    if cand_diag:
        result["entry_diagnostics"] = {
            "strategy_name": cand_diag.get("strategy_name"),
            "candidate_strategies_count": cand_diag.get("candidate_strategies_count", 0),
            "active_strategies_count": cand_diag.get("active_strategies_count", 0),
            "active_strategies": cand_diag.get("active_strategies", []),
            "top3": cand_diag.get("top3", []),
            "sources_confirm_breakdown": {
                "confirming_sources": result.get("layer_scores", {}).get("confirming_sources"),
                "aligned_source_keys": result.get("layer_scores", {}).get("aligned_source_keys", []),
                "combined_score": result.get("layer_scores", {}).get("combined_score"),
                "threshold_used": result.get("layer_scores", {}).get("threshold_used"),
                "l2_has_coverage": result.get("layer_scores", {}).get("l2_has_coverage"),
            },
        }
    return False


def _enrich_signal_metadata_for_entry_pipeline(
    *,
    session: TradingSession,
    signal: Signal,
    flow_metrics: Dict[str, Any],
    l2_metrics: Dict[str, Any],
    momentum_flow_metrics: Dict[str, Any],
    momentum_diversification_metrics: Dict[str, Any],
    regime: Optional[Regime],
    result: Dict[str, Any],
) -> None:
    """Attach runtime gate/flow/regime diagnostics to a signal before entry gates."""

    signal.metadata.setdefault("l2_confirmation", l2_metrics)
    signal.metadata.setdefault("momentum_flow_confirmation", momentum_flow_metrics)
    signal.metadata.setdefault(
        "momentum_diversification",
        momentum_diversification_metrics,
    )
    signal.metadata.setdefault(
        "order_flow",
        _order_flow_metadata_snapshot_impl(flow_metrics),
    )
    signal.metadata["regime"] = regime.value if regime else None
    signal.metadata["micro_regime"] = session.micro_regime
    signal.metadata.setdefault("layer_scores", {})
    signal.metadata["layer_scores"].update(result["layer_scores"])


def _publish_signal_candidate_payload(
    self,
    *,
    session: TradingSession,
    signal: Signal,
    result: Dict[str, Any],
) -> None:
    """Persist signal on session and expose frontend payloads before queueing."""

    signal.metadata["intraday_levels_payload"] = self._intraday_levels_indicator_payload(session)
    session.signals.append(signal)
    result["signal"] = signal.to_dict()
    result["signals"] = [signal.to_dict()]  # Array format for frontend
