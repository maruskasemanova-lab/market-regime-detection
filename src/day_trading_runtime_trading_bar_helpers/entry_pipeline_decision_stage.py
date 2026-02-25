"""Decision/entry stage helpers for the runtime entry pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from ..day_trading_models import BarData, TradingSession
from .entry_pipeline_intrabar_decision import (
    _build_intrabar_decision_state,
)
from .entry_pipeline_signal_execution import _execute_or_reject_decision_signal


def _process_intrabar_decision_and_entry(
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
    result: Dict[str, Any],
) -> bool:
    """Run checkpoint decision evaluation and entry gating/queueing pipeline."""
    decision_state = _build_intrabar_decision_state(
        self,
        session=session,
        bar=bar,
        timestamp=timestamp,
        result=result,
    )
    return _execute_or_reject_decision_signal(
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
        decision_state=decision_state,
        result=result,
    )
