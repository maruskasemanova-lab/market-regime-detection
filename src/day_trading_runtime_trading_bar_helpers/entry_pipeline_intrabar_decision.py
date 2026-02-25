"""Intrabar checkpoint decision extraction helpers for the entry pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from ..day_trading_models import BarData, TradingSession
from .decision_gates import (
    _apply_decision_slice_result_payload,
    _build_decision_proxy_from_slice,
)
from .intrabar_trace import (
    _evaluate_intrabar_checkpoint_trace,
    _resolve_intrabar_checkpoint_meta,
)


def _build_intrabar_decision_state(
    self,
    *,
    session: TradingSession,
    bar: BarData,
    timestamp: datetime,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate intrabar checkpoints and build a decision-state payload."""

    step, checkpoints_meta = _resolve_intrabar_checkpoint_meta(
        session=session,
        bar=bar,
        timestamp=timestamp,
    )
    intrabar_eval_trace, last_slice_res, trigger_slice_res = _evaluate_intrabar_checkpoint_trace(
        self,
        session=session,
        timestamp=timestamp,
        checkpoints_meta=checkpoints_meta,
        capture_trigger_signal=True,
    )
    intrabar_eval_trace["step_seconds"] = step

    decision_slice_res = trigger_slice_res or last_slice_res or {}
    result["intrabar_eval_trace"] = intrabar_eval_trace
    _apply_decision_slice_result_payload(result, decision_slice_res)
    (
        decision,
        effective_trade_threshold,
        passed_trade_threshold,
        tod_boost,
        headwind_boost,
        headwind_metrics,
        required_confirming_sources,
        threshold_used_reason,
    ) = _build_decision_proxy_from_slice(decision_slice_res)

    return {
        "decision": decision,
        "effective_trade_threshold": effective_trade_threshold,
        "passed_trade_threshold": passed_trade_threshold,
        "tod_boost": tod_boost,
        "headwind_boost": headwind_boost,
        "headwind_metrics": headwind_metrics,
        "required_confirming_sources": required_confirming_sources,
        "threshold_used_reason": threshold_used_reason,
    }
