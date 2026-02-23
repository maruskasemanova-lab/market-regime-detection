"""Intrabar evaluation trace helpers for analyzer playback."""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..day_trading_models import BarData
from ..day_trading_runtime_intrabar import calculate_intrabar_1s_snapshot


def build_intrabar_eval_trace(
    *,
    timestamp: datetime,
    bar: BarData,
    result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Build additive analyzer-friendly checkpoint snapshots from intrabar quotes.

    This does not re-run the full decision engine at each checkpoint. It preserves the
    minute-level decision snapshot and adds truthful checkpointed intrabar metrics so
    UI scrubbing can move in 5s (or sampled) increments.
    """
    raw_quotes = getattr(bar, "intrabar_quotes_1s", None)
    if not isinstance(raw_quotes, list) or not raw_quotes:
        return None

    normalized_quotes: List[Dict[str, Any]] = []
    for item in raw_quotes:
        if not isinstance(item, dict):
            continue
        try:
            sec = int(float(item.get("s", 0) or 0))
        except (TypeError, ValueError):
            continue
        if sec < 0 or sec > 59:
            continue
        normalized_quotes.append(dict(item, s=sec))

    if not normalized_quotes:
        return None

    normalized_quotes.sort(key=lambda row: int(row.get("s", 0)))

    checkpoints: List[Dict[str, Any]] = []
    prefix_quotes: List[Dict[str, Any]] = []
    last_sec: Optional[int] = None
    for quote in normalized_quotes:
        sec = int(quote.get("s", 0))
        prefix_quotes.append(quote)
        if last_sec == sec:
            # Keep latest row for duplicated sec payloads.
            prefix_quotes[-2] = quote if len(prefix_quotes) >= 2 else quote
            if len(prefix_quotes) >= 2:
                prefix_quotes.pop()
            continue
        last_sec = sec

        checkpoint_bar = copy.copy(bar)
        checkpoint_bar.intrabar_quotes_1s = [dict(row) for row in prefix_quotes]
        checkpoint_ts = timestamp.replace(second=sec, microsecond=0)
        checkpoint_payload: Dict[str, Any] = {
            "timestamp": checkpoint_ts.isoformat(),
            "offset_sec": sec,
            "intrabar_1s": calculate_intrabar_1s_snapshot(checkpoint_bar),
            "provisional": True,
        }
        checkpoints.append(checkpoint_payload)

    if not checkpoints:
        return None

    checkpoints[-1]["provisional"] = False

    inferred_step = 0
    if len(checkpoints) >= 2:
        diffs = [
            int(checkpoints[idx]["offset_sec"]) - int(checkpoints[idx - 1]["offset_sec"])
            for idx in range(1, len(checkpoints))
            if int(checkpoints[idx]["offset_sec"]) > int(checkpoints[idx - 1]["offset_sec"])
        ]
        if diffs:
            inferred_step = max(1, min(diffs))

    return {
        "schema_version": 1,
        "source": "intrabar_quote_checkpoints",
        "minute_timestamp": timestamp.replace(second=0, microsecond=0).isoformat(),
        "checkpoint_count": len(checkpoints),
        "inferred_step_seconds": inferred_step or None,
        "analysis_snapshot_scope": "minute_bar_decision",
        "checkpoints": checkpoints,
    }


def attach_intrabar_eval_trace(
    *,
    timestamp: datetime,
    bar_data: Dict[str, Any],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return result
    if "intrabar_eval_trace" in result:
        return result

    raw_quotes = bar_data.get("intrabar_quotes_1s")
    if not isinstance(raw_quotes, list) or not raw_quotes:
        return result

    # Reconstruct only the fields needed by intrabar snapshot helper.
    bar = BarData(
        timestamp=timestamp,
        open=float(bar_data.get("open", 0) or 0),
        high=float(bar_data.get("high", 0) or 0),
        low=float(bar_data.get("low", 0) or 0),
        close=float(bar_data.get("close", 0) or 0),
        volume=float(bar_data.get("volume", 0) or 0),
        vwap=bar_data.get("vwap"),
        intrabar_quotes_1s=raw_quotes,
    )

    trace = build_intrabar_eval_trace(timestamp=timestamp, bar=bar, result=result)
    if trace is not None:
        result["intrabar_eval_trace"] = trace
    return result
