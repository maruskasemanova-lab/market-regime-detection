"""Decision-context and liquidity-sweep helpers for runtime trading-bar processing."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List
import logging

from ..day_trading_models import BarData, TradingSession
from ..strategies.base_strategy import Regime, Signal, SignalType
from ..trading_config import TradingConfig
from ..golden_setup_detector import build_golden_config_from_trading_config
from ..day_trading_runtime_sweep import (
    feature_vector_value as _feature_vector_value_impl,
    order_flow_metadata_snapshot as _order_flow_metadata_snapshot_impl,
    resolve_liquidity_sweep_confirmation as _resolve_liquidity_sweep_confirmation_impl,
    to_optional_float as _to_optional_float_impl,
)

logger = logging.getLogger(__name__)



def _prepare_decision_engine_context(
    self,
    *,
    session: TradingSession,
    bar: BarData,
    current_price: float,
    current_bar_index: int,
    formula_indicators: Any,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Prepare orchestrator/feature/flow/golden-setup context for decision evaluation."""

    orch = session.orchestrator
    if orch is None:
        raise RuntimeError("Orchestrator is not initialized")

    bars_data = session.bars[-100:] if len(session.bars) >= 100 else session.bars
    indicators = formula_indicators()
    regime = session.detected_regime or Regime.MIXED
    flow_metrics = dict(indicators.get("order_flow") or {})
    fv = orch.current_feature_vector
    flow_metrics["l2_aggression_z"] = _feature_vector_value_impl(fv, "l2_aggression_z", 0.0)
    flow_metrics["l2_book_pressure_z"] = _feature_vector_value_impl(fv, "l2_book_pressure_z", 0.0)

    golden_cfg = build_golden_config_from_trading_config(
        session.config
        if isinstance(getattr(session, "config", None), TradingConfig)
        else TradingConfig()
    )
    golden_setup_eval = self.golden_setup_detector.evaluate(
        bar=bar,
        bars=bars_data,
        flow_metrics=flow_metrics,
        intraday_levels_state=(
            dict(session.intraday_levels_state)
            if isinstance(getattr(session, "intraday_levels_state", None), dict)
            else {}
        ),
        current_price=float(current_price),
        vwap=_to_optional_float_impl(getattr(bar, "vwap", None)),
        regime=str(session.micro_regime or ""),
        golden_entries_today=int(getattr(session, "golden_setup_entries_today", 0) or 0),
        last_golden_bar_index=int(
            getattr(session, "golden_setup_last_entry_bar_index", -99) or -99
        ),
        current_bar_index=current_bar_index,
        config=golden_cfg,
    )
    golden_setup_payload = golden_setup_eval.to_dict()
    session.golden_setup_result = dict(golden_setup_payload)
    result["golden_setup"] = dict(golden_setup_payload)

    return {
        "orch": orch,
        "bars_data": bars_data,
        "indicators": indicators,
        "regime": regime,
        "flow_metrics": flow_metrics,
        "fv": fv,
        "golden_setup_payload": golden_setup_payload,
    }
