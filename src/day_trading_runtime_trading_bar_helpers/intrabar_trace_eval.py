"""Intrabar checkpoint trace helpers for runtime trading-bar processing."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from ..day_trading_models import BarData, TradingSession
from ..day_trading_runtime.intrabar_slice import (
    runtime_evaluate_intrabar_slice as _runtime_evaluate_intrabar_slice_impl,
)
from ..day_trading_runtime_intrabar import (
    calculate_intrabar_1s_snapshot as _calculate_intrabar_1s_snapshot_impl,
)
from ..day_trading_runtime_sweep import to_optional_float as _to_optional_float_impl


from .intrabar_trace_checkpoints import _resolve_intrabar_checkpoint_meta

def _evaluate_intrabar_checkpoint_trace(
    self,
    *,
    session: TradingSession,
    timestamp: datetime,
    checkpoints_meta: List[tuple[BarData, datetime, int]],
    capture_trigger_signal: bool,
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Evaluate intrabar checkpoints and build the frontend trace payload."""

    intrabar_eval_trace: Dict[str, Any] = {
        "schema_version": 1,
        "source": "intrabar_quote_checkpoints",
        "minute_timestamp": timestamp.replace(second=0, microsecond=0).isoformat(),
        "checkpoints": [],
    }

    last_slice_res: Optional[Dict[str, Any]] = None
    trigger_slice_res: Optional[Dict[str, Any]] = None
    slice_eval_impl = _runtime_evaluate_intrabar_slice_impl
    # Preserve existing test hooks that monkey-patch the runtime module alias.
    try:
        from .. import day_trading_runtime_trading_bar as _rtb_mod

        patched_impl = getattr(_rtb_mod, "_runtime_evaluate_intrabar_slice_impl", None)
        if callable(patched_impl):
            slice_eval_impl = patched_impl
    except Exception:
        pass

    for cp_bar, cp_ts, sec in checkpoints_meta:
        raw_slice_res = slice_eval_impl(self, session, cp_bar, cp_ts)
        slice_res = raw_slice_res if isinstance(raw_slice_res, dict) else {}
        cp_layer_scores = slice_res.get("layer_scores")

        cp_payload = {
            "timestamp": slice_res.get("timestamp", cp_ts.isoformat()),
            "offset_sec": sec,
            "layer_scores": cp_layer_scores,
            "intrabar_1s": _calculate_intrabar_1s_snapshot_impl(cp_bar),
            "provisional": True,
        }
        if "signal_rejected" in slice_res:
            cp_payload["signal_rejected"] = slice_res["signal_rejected"]
        if "candidate_diagnostics" in slice_res:
            cp_payload["candidate_diagnostics"] = slice_res["candidate_diagnostics"]

        intrabar_eval_trace["checkpoints"].append(cp_payload)
        last_slice_res = slice_res
        if (
            capture_trigger_signal
            and trigger_slice_res is None
            and slice_res.get("_raw_signal") is not None
            and bool((cp_layer_scores or {}).get("passed", False))
        ):
            trigger_slice_res = slice_res

    if intrabar_eval_trace["checkpoints"]:
        intrabar_eval_trace["checkpoints"][-1]["provisional"] = False
        intrabar_eval_trace["checkpoint_count"] = len(intrabar_eval_trace["checkpoints"])

    return intrabar_eval_trace, last_slice_res, trigger_slice_res


def _attach_active_position_intrabar_trace(
    self,
    *,
    session: TradingSession,
    bar: BarData,
    timestamp: datetime,
    result: Dict[str, Any],
) -> None:
    """Attach intrabar checkpoint trace to the result when a position is active."""

    step, checkpoints_meta = _resolve_intrabar_checkpoint_meta(
        session=session,
        bar=bar,
        timestamp=timestamp,
    )
    intrabar_eval_trace, _, _ = _evaluate_intrabar_checkpoint_trace(
        self,
        session=session,
        timestamp=timestamp,
        checkpoints_meta=checkpoints_meta,
        capture_trigger_signal=False,
    )
    intrabar_eval_trace["step_seconds"] = step
    result["intrabar_eval_trace"] = intrabar_eval_trace
