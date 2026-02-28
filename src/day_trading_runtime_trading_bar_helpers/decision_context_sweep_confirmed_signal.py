"""Decision-context and liquidity-sweep helpers for runtime trading-bar processing."""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, Dict, List, Optional

from ..day_trading_models import BarData, TradingSession
from ..strategies.base_strategy import BaseStrategy, Signal, SignalType
from ..day_trading_runtime_sweep import (
    order_flow_metadata_snapshot as _order_flow_metadata_snapshot_impl,
    to_optional_float as _to_optional_float_impl,
)

logger = logging.getLogger(__name__)


def _coerce_priority(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _resolve_liquidity_sweep_signal_owner(
    self,
    session: TradingSession,
) -> Dict[str, Any]:
    """Resolve which concrete strategy may own a confirmed sweep signal."""

    strategies = getattr(self, "strategies", {}) or {}
    canonicalizer = getattr(self, "_canonical_strategy_key", None)

    def normalize(name: Any) -> str:
        raw = str(name or "").strip()
        if not raw:
            return ""
        if callable(canonicalizer):
            try:
                normalized = str(canonicalizer(raw) or "").strip()
                if normalized:
                    return normalized
            except Exception:
                logger.debug("Failed to canonicalize strategy name %r", raw, exc_info=True)
        return raw.lower()

    def candidate_for(strategy_key: str) -> Optional[Dict[str, Any]]:
        if not strategy_key or strategy_key == "adaptive":
            return None
        strategy = strategies.get(strategy_key)
        if not isinstance(strategy, BaseStrategy):
            return None
        if not bool(getattr(strategy, "enabled", True)):
            return None
        if not bool(getattr(strategy, "liquidity_sweep_signal_enabled", True)):
            return None
        return {
            "key": strategy_key,
            "name": str(getattr(strategy, "name", "") or strategy_key),
            "priority": _coerce_priority(
                getattr(strategy, "liquidity_sweep_signal_priority", 0)
            ),
        }

    selected_key = normalize(getattr(session, "selected_strategy", None))
    selected_candidate = candidate_for(selected_key)
    if selected_candidate is not None:
        selected_candidate["reason"] = "selected_strategy"
        return selected_candidate
    if selected_key and selected_key != "adaptive":
        return {
            "key": None,
            "name": None,
            "priority": None,
            "reason": "selected_strategy_liquidity_sweep_disabled",
        }

    ranked_candidates: List[tuple[int, int, Dict[str, Any]]] = []
    for order, raw_name in enumerate(getattr(session, "active_strategies", []) or []):
        strategy_key = normalize(raw_name)
        candidate = candidate_for(strategy_key)
        if candidate is None:
            continue
        ranked_candidates.append((-int(candidate["priority"]), order, candidate))

    if ranked_candidates:
        ranked_candidates.sort(key=lambda item: (item[0], item[1]))
        best = ranked_candidates[0][2]
        best["reason"] = "active_strategy_priority"
        return best

    return {
        "key": None,
        "name": None,
        "priority": None,
        "reason": "no_active_strategy_liquidity_sweep_enabled",
    }


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

    owner = _resolve_liquidity_sweep_signal_owner(self, session)
    confirmation_payload = result.setdefault(
        "liquidity_sweep_confirmation",
        dict(sweep_confirmation),
    )
    confirmation_payload["strategy_owner"] = owner.get("key")
    confirmation_payload["strategy_owner_name"] = owner.get("name")
    confirmation_payload["strategy_owner_priority"] = owner.get("priority")
    confirmation_payload["strategy_owner_reason"] = owner.get("reason")
    if not owner.get("key") or not owner.get("name"):
        confirmation_payload["signal_queued"] = False
        return False

    strategy_name = str(owner["name"])
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
                "strategy_owner": owner.get("key"),
                "strategy_owner_name": owner.get("name"),
                "strategy_owner_reason": owner.get("reason"),
                "strategy_owner_priority": owner.get("priority"),
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
        confirmation_payload["signal_queued"] = False
        return True

    session.signals.append(sweep_signal)
    result["signal"] = sweep_signal.to_dict()
    result["signals"] = [sweep_signal.to_dict()]
    session.pending_signal = sweep_signal
    session.pending_signal_bar_index = current_bar_index
    result["action"] = "signal_queued"
    result["queued_for_next_bar"] = True
    result["sweep_detected"] = True
    confirmation_payload["signal_queued"] = True
    return True
