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



def _handle_confirmed_liquidity_sweep_signal(
    self,
    *,
    session: TradingSession,
    timestamp: datetime,
    current_bar_index: int,
    current_price: float,
    indicators: Dict[str, Any],
    bars_data: List[BarData],
    flow_metrics: Dict[str, Any],
    sweep_confirmation: Dict[str, Any],
    result: Dict[str, Any],
) -> bool:
    """Build and queue a liquidity-sweep-confirmed signal. Returns True if handled."""

    if not bool(sweep_confirmation.get("confirmed", False)):
        return False

    sweep_direction = str(sweep_confirmation.get("direction", "long")).strip().lower()
    atr_now = (
        _to_optional_float_impl(self._latest_indicator_value(indicators, "atr", bars_data))
        or 0.0
    )
    level_price = _to_optional_float_impl(sweep_confirmation.get("level_price")) or current_price
    risk_buffer_abs = max(current_price * 0.001, atr_now * 0.25)
    strategy_name = str(session.selected_strategy or "").strip()
    if not strategy_name:
        strategy_name = (
            session.active_strategies[0]
            if isinstance(session.active_strategies, list) and session.active_strategies
            else "absorption_reversal"
        )
    if sweep_direction == "short":
        signal_type = SignalType.SELL
        base_sl = max(current_price + risk_buffer_abs, level_price + risk_buffer_abs)
        risk_abs = max(base_sl - current_price, current_price * 0.002)
        base_tp = max(0.0, current_price - (risk_abs * 2.0))
        sweep_reason = (
            f"Liquidity sweep confirmed at resistance {level_price:.4f}; "
            "absorption exhausted and flow flipped lower."
        )
    else:
        sweep_direction = "long"
        signal_type = SignalType.BUY
        base_sl = max(0.0, min(current_price - risk_buffer_abs, level_price - risk_buffer_abs))
        risk_abs = max(current_price - base_sl, current_price * 0.002)
        base_tp = current_price + (risk_abs * 2.0)
        sweep_reason = (
            f"Liquidity sweep confirmed at support {level_price:.4f}; "
            "absorption held and flow flipped higher."
        )

    sweep_signal = Signal(
        strategy_name=strategy_name,
        signal_type=signal_type,
        price=float(current_price),
        timestamp=timestamp,
        confidence=92.0,
        stop_loss=float(base_sl),
        take_profit=float(base_tp),
        trailing_stop=True,
        trailing_stop_pct=0.8,
        reasoning=sweep_reason,
        metadata={
            "sweep_triggered": True,
            "sweep_detected": True,
            "liquidity_sweep": {
                **dict(sweep_confirmation),
                "detected": True,
                "atr": atr_now,
            },
        },
    )
    sweep_signal.metadata["order_flow"] = _order_flow_metadata_snapshot_impl(flow_metrics)
    sweep_signal.metadata["intraday_levels_payload"] = self._intraday_levels_indicator_payload(
        session
    )
    sweep_signal.metadata["level_context"] = {
        "gate": "liquidity_sweep_confirmation",
        "passed": True,
        "reason": "liquidity_sweep_confirmed",
        "direction": sweep_direction,
        "level_price": level_price,
        "vwap_execution_flow": float(sweep_confirmation.get("vwap_execution_flow", 0.0) or 0.0),
    }
    risk_pct = abs(float(sweep_signal.stop_loss) - float(sweep_signal.price)) / max(
        float(sweep_signal.price), 1e-9
    ) * 100.0
    if risk_pct < 0.10:
        result["action"] = "liquidity_sweep_filtered"
        result["reason"] = "cost_aware_sweep_risk_too_small"
        result["signal_rejected"] = {
            "gate": "cost_aware_sweep",
            "risk_pct": round(risk_pct, 4),
            "min_required": 0.10,
            "timestamp": timestamp.isoformat(),
        }
        return True

    session.signals.append(sweep_signal)
    result["signal"] = sweep_signal.to_dict()
    result["signals"] = [sweep_signal.to_dict()]
    session.pending_signal = sweep_signal
    session.pending_signal_bar_index = current_bar_index
    result["action"] = "signal_queued"
    result["queued_for_next_bar"] = True
    result["sweep_detected"] = True
    return True
