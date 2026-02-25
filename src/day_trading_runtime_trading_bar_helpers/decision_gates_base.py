"""Decision-gate helpers for runtime trading-bar processing."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, Optional

from ..day_trading_models import TradingSession
from ..day_trading_runtime_portfolio import (
    cooldown_bars_remaining as _cooldown_bars_remaining_impl,
)
from ..strategies.base_strategy import Regime, Signal, SignalType


def _normalize_direction_token(raw: Any) -> str:
    token = str(raw or "").strip().lower()
    if token in {"bullish", "long", "buy", "up", "positive"}:
        return "bullish"
    if token in {"bearish", "short", "sell", "down", "negative"}:
        return "bearish"
    return ""


def _signal_direction(signal: Optional[Signal]) -> str:
    if signal is None:
        return ""
    if signal.signal_type == SignalType.BUY:
        return "bullish"
    if signal.signal_type == SignalType.SELL:
        return "bearish"
    return ""


def _apply_pre_entry_guards(
    self,
    *,
    session: TradingSession,
    timestamp: datetime,
    current_bar_index: int,
    session_key: str,
    result: Dict[str, Any],
) -> bool:
    """Apply early-return guard rails before decision-engine signal generation."""

    cooldown_remaining = _cooldown_bars_remaining_impl(session, current_bar_index)
    if cooldown_remaining > 0:
        result["action"] = "consecutive_loss_cooldown"
        result["reason"] = f"Consecutive-loss cooldown: {cooldown_remaining} bars remaining"
        result["cooldown_bars_remaining"] = cooldown_remaining
        return True

    if str(session.micro_regime or "").upper() == "UNKNOWN":
        result["action"] = "regime_warmup"
        result["reason"] = "ADX warmup incomplete; signal generation paused."
        return True

    max_daily_trades = self._resolve_max_daily_trades(session.ticker, session=session)
    if max_daily_trades is not None and len(session.trades) >= max_daily_trades:
        result["action"] = "trade_limit_reached"
        result["reason"] = f"Max trades per day ({max_daily_trades}) reached"
        return True

    last_trade_bar = self.last_trade_bar_index.get(session_key, -self.trade_cooldown_bars)
    bars_since_last_trade = current_bar_index - last_trade_bar
    if bars_since_last_trade < self.trade_cooldown_bars:
        remaining = self.trade_cooldown_bars - bars_since_last_trade
        result["action"] = "cooldown_active"
        result["reason"] = f"Cooldown: {remaining} bars remaining"
        return True

    bar_hour = self._to_market_time(timestamp).time().hour
    ticker_aos_config = self.ticker_params.get(session.ticker.upper(), {})
    trading_hours = ticker_aos_config.get("trading_hours", None)
    if trading_hours and ticker_aos_config.get("time_filter_enabled", True):
        if bar_hour not in trading_hours:
            result["action"] = "time_filter"
            result["reason"] = f"Hour {bar_hour}:00 not in allowed hours {trading_hours}"
            return True

    return False


def _apply_decision_slice_result_payload(
    result: Dict[str, Any],
    decision_slice_res: Dict[str, Any],
) -> None:
    """Copy checkpoint decision diagnostics into the runtime result payload."""

    if "layer_scores" in decision_slice_res:
        result["layer_scores"] = decision_slice_res["layer_scores"]
    if "signal_rejected" in decision_slice_res:
        result["signal_rejected"] = decision_slice_res["signal_rejected"]
    if "candidate_diagnostics" in decision_slice_res:
        result["candidate_diagnostics"] = decision_slice_res["candidate_diagnostics"]


def _build_decision_proxy_from_slice(
    decision_slice_res: Dict[str, Any],
) -> tuple[SimpleNamespace, float, bool, float, float, Any, int, str]:
    """Build a decision-like proxy and threshold metadata from intrabar slice output."""

    layer_scores = decision_slice_res.get("layer_scores", {})
    combined_score = float(
        decision_slice_res.get("_combined_score_raw", layer_scores.get("combined_score", 0.0))
        or 0.0
    )
    effective_trade_threshold = float(layer_scores.get("threshold_used", 0.0))
    passed_trade_threshold = bool(
        decision_slice_res.get(
            "_passed_trade_threshold",
            combined_score >= effective_trade_threshold,
        )
    )
    tod_boost = float(layer_scores.get("tod_threshold_boost", 0.0))
    headwind_boost = float(layer_scores.get("headwind_threshold_boost", 0.0))
    headwind_metrics = layer_scores.get("cross_asset_headwind", {})
    required_confirming_sources = int(layer_scores.get("required_confirming_sources", 2))
    threshold_used_reason = str(layer_scores.get("threshold_used_reason", "base_threshold"))

    decision = SimpleNamespace(
        combined_score=combined_score,
        execute=bool(decision_slice_res.get("_decision_execute", layer_scores.get("passed", False))),
        signal=decision_slice_res.get("_raw_signal"),
        strategy_score=float(layer_scores.get("strategy_score", 0.0)),
        combined_raw=float(layer_scores.get("combined_raw", 0.0)),
        combined_norm_0_100=layer_scores.get("combined_norm_0_100", None),
        threshold=float(layer_scores.get("threshold", 0.0)),
        reasoning=str(
            decision_slice_res.get("_decision_reasoning", "Checkpoints evaluated")
            or "Checkpoints evaluated"
        ),
    )
    return (
        decision,
        effective_trade_threshold,
        passed_trade_threshold,
        tod_boost,
        headwind_boost,
        headwind_metrics,
        required_confirming_sources,
        threshold_used_reason,
    )



def _apply_golden_setup_signal_adjustments(
    *,
    signal: Signal,
    golden_setup_payload: Dict[str, Any],
    result: Dict[str, Any],
) -> None:
    """Apply golden-setup confidence/metadata enrichment when direction aligns."""

    golden_conf_boost = 0.0
    golden_active = bool(golden_setup_payload.get("active", False))
    golden_setup_direction = _normalize_direction_token(golden_setup_payload.get("best_direction"))
    signal_direction = _signal_direction(signal)
    golden_applied = (
        golden_active
        and bool(golden_setup_direction)
        and golden_setup_direction == signal_direction
    )

    if golden_applied:
        golden_conf_boost = max(
            0.0,
            float(golden_setup_payload.get("confidence_boost", 0.0) or 0.0),
        )
        if golden_conf_boost > 0.0:
            signal.confidence = min(100.0, signal.confidence + golden_conf_boost)
        signal.metadata["golden_setup"] = {
            **dict(golden_setup_payload),
            "applied": True,
            "signal_direction": signal_direction,
            "applied_confidence_boost": round(golden_conf_boost, 4),
            "applied_threshold_relief": round(
                float(result.get("layer_scores", {}).get("golden_setup_relief", 0.0) or 0.0),
                4,
            ),
        }
        result["golden_setup"] = dict(signal.metadata["golden_setup"])

    if isinstance(result.get("layer_scores"), dict):
        result["layer_scores"]["golden_setup_applied"] = bool(golden_applied)
        result["layer_scores"]["golden_setup_confidence_boost"] = round(
            float(golden_conf_boost), 4
        )


