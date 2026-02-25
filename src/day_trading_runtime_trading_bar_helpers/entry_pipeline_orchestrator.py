"""Entry signal generation pipeline for runtime trading-bar processing."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from ..day_trading_models import BarData, TradingSession
from .decision_context import (
    _apply_liquidity_sweep_detection_and_confirmation,
    _prepare_decision_engine_context,
)
from .decision_gates import (
    _apply_mu_choppy_regime_gate,
    _apply_pre_entry_guards,
)
from .entry_pipeline_decision_stage import _process_intrabar_decision_and_entry


def _process_entry_signal_generation_pipeline(
    self,
    *,
    session: TradingSession,
    session_key: str,
    bar: BarData,
    timestamp: datetime,
    current_price: float,
    current_bar_index: int,
    formula_indicators: Any,
    result: Dict[str, Any],
) -> bool:
    """Run pre-entry guards and decision/gate pipeline when no position is active."""

    if session.active_position or not session.selected_strategy:
        return False

    if _apply_pre_entry_guards(
        self,
        session=session,
        timestamp=timestamp,
        current_bar_index=current_bar_index,
        session_key=session_key,
        result=result,
    ):
        return True

    decision_ctx = _prepare_decision_engine_context(
        self,
        session=session,
        bar=bar,
        current_price=current_price,
        current_bar_index=current_bar_index,
        formula_indicators=formula_indicators,
        result=result,
    )
    bars_data = decision_ctx["bars_data"]
    indicators = decision_ctx["indicators"]
    regime = decision_ctx["regime"]
    flow_metrics = decision_ctx["flow_metrics"]
    fv = decision_ctx["fv"]
    golden_setup_payload = decision_ctx["golden_setup_payload"]

    if _apply_liquidity_sweep_detection_and_confirmation(
        self,
        session=session,
        timestamp=timestamp,
        current_bar_index=current_bar_index,
        current_price=current_price,
        fv=fv,
        flow_metrics=flow_metrics,
        indicators=indicators,
        bars_data=bars_data,
        result=result,
    ):
        return True

    if _apply_mu_choppy_regime_gate(
        self,
        session=session,
        regime=regime,
        timestamp=timestamp,
        golden_setup_payload=golden_setup_payload,
        result=result,
    ):
        return True

    return _process_intrabar_decision_and_entry(
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
        result=result,
    )
