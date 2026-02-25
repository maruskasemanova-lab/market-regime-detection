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


from .decision_context_sweep_confirmed_signal import (
    _handle_confirmed_liquidity_sweep_signal,
)

def _apply_liquidity_sweep_detection_and_confirmation(
    self,
    *,
    session: TradingSession,
    timestamp: datetime,
    current_bar_index: int,
    current_price: float,
    fv: Any,
    flow_metrics: Dict[str, Any],
    indicators: Dict[str, Any],
    bars_data: List[BarData],
    result: Dict[str, Any],
) -> bool:
    """Populate sweep detection/confirmation payloads and queue confirmed sweep signals."""

    if not bool(session.potential_sweep_active):
        sweep_detection = self._detect_liquidity_sweep(
            session=session,
            current_price=current_price,
            fv=fv,
            flow_metrics=flow_metrics,
        )
    else:
        sweep_detection = {
            "enabled": bool(
                isinstance(session.config, TradingConfig)
                and getattr(session.config, "liquidity_sweep_detection_enabled", False)
            ),
            "sweep_detected": False,
            "reason": "awaiting_confirmation",
            **(
                dict(session.potential_sweep_context)
                if isinstance(session.potential_sweep_context, dict)
                else {}
            ),
        }

    reason = sweep_detection.get("reason")
    if reason and reason != "disabled" and reason != "awaiting_confirmation":
        logger.debug(
            "Liquidity sweep detection reason=%s payload=%s",
            reason,
            sweep_detection,
        )

    result["liquidity_sweep"] = dict(sweep_detection)

    sweep_confirmation = _resolve_liquidity_sweep_confirmation_impl(
        session=session,
        current_bar_index=current_bar_index,
        current_price=current_price,
        flow_metrics=flow_metrics,
    )
    if sweep_confirmation.get("active") or sweep_confirmation.get("reason") == "confirmed":
        result["liquidity_sweep_confirmation"] = dict(sweep_confirmation)

    return _handle_confirmed_liquidity_sweep_signal(
        self,
        session=session,
        timestamp=timestamp,
        current_bar_index=current_bar_index,
        current_price=current_price,
        indicators=indicators,
        bars_data=bars_data,
        flow_metrics=flow_metrics,
        sweep_confirmation=sweep_confirmation,
        result=result,
    )

