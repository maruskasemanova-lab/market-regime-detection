"""Entry-stage action helpers for runtime trading-bar processing."""

from __future__ import annotations

from datetime import datetime
import logging
import re
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

_MARGIN_REASON_RE = re.compile(
    r"margin\s+(-?\d+(?:\.\d+)?)\s*<\s*required\s+(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _extract_threshold_rejection_context(
    *,
    decision: SimpleNamespace,
    effective_trade_threshold: float,
    passed_trade_threshold: bool,
) -> Dict[str, Any]:
    """Compute precise threshold rejection diagnostics for UI/debug payloads."""
    combined_score = float(getattr(decision, "combined_score", 0.0) or 0.0)
    model_threshold = float(getattr(decision, "threshold", 0.0) or 0.0)
    score_minus_model_threshold = combined_score - model_threshold
    score_minus_trade_threshold = combined_score - float(effective_trade_threshold)
    required_margin = None

    explicit_required = getattr(decision, "required_margin", None)
    try:
        if explicit_required is not None:
            required_margin = float(explicit_required)
    except (TypeError, ValueError):
        required_margin = None

    reasoning = str(getattr(decision, "reasoning", "") or "")
    match = _MARGIN_REASON_RE.search(reasoning)
    if match:
        try:
            score_minus_model_threshold = float(match.group(1))
            required_margin = float(match.group(2))
        except (TypeError, ValueError):
            pass

    passes_margin_gate = None
    if required_margin is not None:
        passes_margin_gate = score_minus_model_threshold >= required_margin

    if bool(getattr(decision, "execute", False)) and not passed_trade_threshold:
        rejection_subtype = "headwind_or_trade_threshold"
    elif score_minus_trade_threshold < 0:
        rejection_subtype = "score_below_trade_threshold"
    elif passes_margin_gate is False:
        rejection_subtype = "margin_insufficient"
    else:
        rejection_subtype = "ensemble_execute_false"

    return {
        "score_minus_trade_threshold": round(float(score_minus_trade_threshold), 4),
        "score_minus_model_threshold": round(float(score_minus_model_threshold), 4),
        "required_margin": (
            round(float(required_margin), 4) if required_margin is not None else None
        ),
        "passes_margin_gate": passes_margin_gate,
        "rejection_subtype": rejection_subtype,
    }


def _apply_intraday_levels_entry_quality_gate(
    self,
    *,
    session: TradingSession,
    signal: Signal,
    current_price: float,
    current_bar_index: int,
    decision: SimpleNamespace,
    effective_trade_threshold: float,
    regime: Optional[Regime],
    tod_boost: float,
    timestamp: datetime,
    result: Dict[str, Any],
) -> bool:
    """Apply intraday-levels entry quality gate and optional TP override."""

    level_context = _runtime_evaluate_intraday_levels_entry_quality_impl(
        self=self,
        session=session,
        signal=signal,
        current_price=current_price,
        current_bar_index=current_bar_index,
    )
    result["level_context"] = level_context
    signal.metadata["level_context"] = level_context

    if not level_context.get("passed", True):
        result["action"] = "intraday_levels_filtered"
        result["reason"] = level_context.get(
            "reason",
            "intraday_levels_entry_quality_failed",
        )
        result["signal_rejected"] = {
            "gate": "intraday_levels_entry_quality",
            "strategy": signal.strategy_name,
            "signal_type": signal.signal_type.value if signal.signal_type else None,
            "confidence": round(signal.confidence, 1),
            "combined_score": round(decision.combined_score, 1),
            "threshold_used": effective_trade_threshold,
            "regime": regime.value if regime else None,
            "micro_regime": session.micro_regime,
            "level_context": level_context,
            "tod_threshold_boost": tod_boost,
            "timestamp": timestamp.isoformat(),
        }
        return True

    target_override = _to_optional_float_impl(level_context.get("target_price_override"))
    if target_override is not None and target_override > 0.0:
        signal.take_profit = target_override
        signal.metadata["target_price_source"] = "intraday_levels_poc"
        signal.metadata["target_price_override"] = target_override
    return False


def _apply_custom_entry_formula_gate(
    self,
    *,
    session: TradingSession,
    signal: Signal,
    bar: BarData,
    indicators: Dict[str, Any],
    flow_metrics: Dict[str, Any],
    current_bar_index: int,
    decision: SimpleNamespace,
    effective_trade_threshold: float,
    regime: Optional[Regime],
    timestamp: datetime,
    result: Dict[str, Any],
) -> bool:
    """Apply strategy custom entry formula gate when configured."""

    strategy_key = self._canonical_strategy_key(signal.strategy_name or "")
    strategy_obj = self.strategies.get(strategy_key)
    if strategy_obj is None:
        return False

    entry_formula_ctx = self.strategy_evaluator.build_strategy_formula_context(
        session=session,
        bar=bar,
        indicators=indicators,
        flow=flow_metrics,
        current_bar_index=current_bar_index,
        signal=signal,
    )
    custom_entry_formula = self.strategy_evaluator.evaluate_strategy_custom_formula(
        strategy=strategy_obj,
        formula_type="entry",
        context=entry_formula_ctx,
    )
    if custom_entry_formula.get("enabled", False):
        result["custom_entry_formula"] = custom_entry_formula
    if custom_entry_formula.get("enabled", False) and not custom_entry_formula.get("passed", False):
        result["action"] = "custom_entry_formula_filtered"
        result["reason"] = custom_entry_formula.get("error") or "Custom entry formula returned false."
        result["signal_rejected"] = {
            "gate": "custom_entry_formula",
            "strategy": signal.strategy_name,
            "signal_type": signal.signal_type.value if signal.signal_type else None,
            "confidence": round(signal.confidence, 1),
            "combined_score": round(decision.combined_score, 1),
            "threshold_used": effective_trade_threshold,
            "regime": regime.value if regime else None,
            "micro_regime": session.micro_regime,
            "custom_entry_formula": custom_entry_formula,
            "timestamp": timestamp.isoformat(),
        }
        return True
    return False


def _apply_threshold_rejection_payload(
    *,
    decision: SimpleNamespace,
    passed_trade_threshold: bool,
    effective_trade_threshold: float,
    threshold_used_reason: str,
    regime: Optional[Regime],
    micro_regime: Any,
    tod_boost: float,
    headwind_boost: float,
    headwind_metrics: Any,
    timestamp: datetime,
    result: Dict[str, Any],
) -> None:
    """Populate rejection payload when a candidate fails threshold/headwind gating."""
    rejection_context = _extract_threshold_rejection_context(
        decision=decision,
        effective_trade_threshold=effective_trade_threshold,
        passed_trade_threshold=passed_trade_threshold,
    )

    result["signal_rejected"] = {
        "gate": "cross_asset_headwind"
        if (decision.execute and not passed_trade_threshold)
        else "threshold",
        "schema_version": 2,
        "combined_score": round(decision.combined_score, 1),
        "combined_raw": round(
            float(getattr(decision, "combined_raw", decision.combined_score) or 0.0),
            1,
        ),
        "combined_norm_0_100": round(
            float(getattr(decision, "combined_norm_0_100", 0.0) or 0.0),
            1,
        ),
        "threshold_used": effective_trade_threshold,
        "trade_gate_threshold": effective_trade_threshold,
        "threshold_used_reason": threshold_used_reason,
        "strategy_score": round(decision.strategy_score, 1),
        "regime": regime.value if regime else None,
        "micro_regime": micro_regime,
        "reasoning": decision.reasoning,
        "tod_threshold_boost": tod_boost,
        "headwind_threshold_boost": round(float(headwind_boost), 4),
        "cross_asset_headwind": headwind_metrics,
        "timestamp": timestamp.isoformat(),
        **rejection_context,
    }
