"""Session-level guard and early-exit helpers for runtime trading-bar processing."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from ..day_trading_models import BarData, SessionPhase, TradingSession
from ..day_trading_runtime_portfolio import (
    realized_pnl_dollars as _realized_pnl_dollars_impl,
    unrealized_pnl_dollars as _unrealized_pnl_dollars_impl,
)
from ..trading_config import TradingConfig


def _apply_regime_update_payload(result: Dict[str, Any], regime_update: Dict[str, Any]) -> None:
    """Merge the standard regime refresh payload into the runtime result."""

    result["regime_update"] = regime_update
    result["regime"] = regime_update.get("regime")
    result["micro_regime"] = regime_update.get("micro_regime")
    result["strategies"] = regime_update.get("strategies", [])
    result["strategy"] = regime_update.get("strategy")
    result["indicators"] = regime_update.get("indicators", {})


def _handle_warmup_only_mode(
    self,
    *,
    session: TradingSession,
    current_bar_index: int,
    timestamp: datetime,
    result: Dict[str, Any],
) -> bool:
    """Process warmup-only bars while keeping regime state fresh and dropping pending entries."""

    if getattr(session, "pending_signal", None) is not None:
        session.pending_signal = None
        session.pending_signal_bar_index = -1
        result["dropped_pending_signal"] = True

    regime_update = self._maybe_refresh_regime(session, current_bar_index, timestamp)
    if regime_update:
        _apply_regime_update_payload(result, regime_update)

    result["action"] = "warmup_only"
    result["warmup_only"] = True
    result["reason"] = "Trading disabled for warmup bars"
    return True


def _handle_portfolio_drawdown_halt(
    self,
    *,
    session: TradingSession,
    bar: BarData,
    timestamp: datetime,
    current_bar_index: int,
    current_price: float,
    portfolio_drawdown: Dict[str, Any],
    result: Dict[str, Any],
) -> bool:
    """Handle run-level portfolio drawdown halts, closing positions and ending the session."""

    if not portfolio_drawdown.get("halted", False):
        return False

    if session.pending_signal is not None:
        session.pending_signal = None
        session.pending_signal_bar_index = -1
        result["dropped_pending_signal"] = True

    if session.active_position:
        trade = self._close_position(
            session,
            current_price,
            timestamp,
            "portfolio_drawdown_halt",
            bar_volume=bar.volume,
        )
        session.last_exit_bar_index = current_bar_index
        result["trade_closed"] = trade.to_dict()
        bars_held = len(
            [
                b
                for b in session.bars
                if b.timestamp >= trade.entry_time and b.timestamp <= trade.exit_time
            ]
        )
        result["position_closed"] = self.gate_engine.build_position_closed_payload(
            trade=trade,
            exit_reason="portfolio_drawdown_halt",
            bars_held=bars_held,
        )

    session.phase = SessionPhase.END_OF_DAY
    session.end_price = current_price
    self._persist_intraday_levels_memory(session)
    result["action"] = "portfolio_drawdown_halt"
    result["reason"] = (
        f"Run-level drawdown {portfolio_drawdown.get('drawdown_pct', 0.0):.2f}% "
        f"<= -{portfolio_drawdown.get('halt_threshold_pct', 0.0):.2f}%"
    )
    result["portfolio_halt_triggered"] = bool(portfolio_drawdown.get("halt_triggered", False))
    result["session_summary"] = self._get_session_summary(session)
    return True


def _apply_max_daily_loss_circuit_breaker(
    self,
    *,
    session: TradingSession,
    bar: BarData,
    timestamp: datetime,
    current_bar_index: int,
    current_price: float,
    result: Dict[str, Any],
) -> bool:
    """Apply max daily loss circuit breaker (realized + unrealized) and end session if tripped."""

    current_realized_pnl = _realized_pnl_dollars_impl(session)
    current_unrealized_pnl = _unrealized_pnl_dollars_impl(
        session=session,
        current_price=current_price,
        trading_costs=self.trading_costs,
        bar_volume=bar.volume,
    )
    current_total_pnl = current_realized_pnl + current_unrealized_pnl

    if current_total_pnl >= -self.max_daily_loss:
        return False

    if session.active_position:
        trade = self._close_position(
            session,
            current_price,
            timestamp,
            "max_daily_loss",
            bar_volume=bar.volume,
        )
        result["trade_closed"] = trade.to_dict()
        result["action"] = "max_loss_stop"
        bars_held = len(
            [b for b in session.bars if b.timestamp >= trade.entry_time and b.timestamp <= trade.exit_time]
        )
        result["position_closed"] = self.gate_engine.build_position_closed_payload(
            trade=trade,
            exit_reason="max_daily_loss",
            bars_held=bars_held,
        )

    result["max_daily_loss_trigger"] = {
        "realized_pnl_dollars": round(current_realized_pnl, 4),
        "unrealized_pnl_dollars": round(current_unrealized_pnl, 4),
        "total_pnl_dollars": round(current_total_pnl, 4),
        "max_daily_loss": float(self.max_daily_loss),
    }
    session.phase = SessionPhase.END_OF_DAY
    session.end_price = current_price
    self._persist_intraday_levels_memory(session)
    result["session_summary"] = self._get_session_summary(session)
    return True


def _build_default_liquidity_sweep_payload(session: TradingSession) -> Dict[str, Any]:
    """Build the default liquidity-sweep diagnostics payload for every runtime result."""

    sweep_enabled = bool(
        isinstance(getattr(session, "config", None), TradingConfig)
        and getattr(session.config, "liquidity_sweep_detection_enabled", False)
    )
    default_sweep_reason = "disabled"
    if sweep_enabled:
        default_sweep_reason = (
            "strategy_not_selected"
            if not bool(getattr(session, "selected_strategy", None))
            else "not_evaluated"
        )
    payload: Dict[str, Any] = {
        "enabled": sweep_enabled,
        "sweep_detected": False,
        "reason": default_sweep_reason,
    }
    if not bool(getattr(session, "selected_strategy", None)):
        warnings = [
            str(item)
            for item in getattr(session, "selection_warnings", [])
            if str(item).strip()
        ]
        if warnings:
            payload["selection_warnings"] = warnings
    return payload
