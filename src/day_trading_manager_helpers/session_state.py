"""Session summary and lifecycle state helpers for DayTradingManager."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..day_trading_models import SessionPhase, TradingSession


def build_session_summary(session: TradingSession) -> Dict[str, Any]:
    """Build a stable session summary payload."""
    trades = session.trades

    if not trades:
        return {
            "ticker": session.ticker,
            "date": session.date,
            "regime": session.detected_regime.value if session.detected_regime else None,
            "micro_regime": session.micro_regime,
            "strategy": session.selected_strategy,
            "selection_warnings": list(session.selection_warnings),
            "total_trades": 0,
            "total_pnl_pct": 0,
            "success": False,
        }

    winning = [trade for trade in trades if trade.pnl_pct > 0]
    losing = [trade for trade in trades if trade.pnl_pct <= 0]

    return {
        "ticker": session.ticker,
        "date": session.date,
        "run_id": session.run_id,
        "regime": session.detected_regime.value if session.detected_regime else None,
        "micro_regime": session.micro_regime,
        "strategy": session.selected_strategy,
        "selection_warnings": list(session.selection_warnings),
        "total_trades": len(trades),
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "win_rate": len(winning) / len(trades) * 100 if trades else 0,
        "trades": [trade.to_dict() for trade in trades],
        "total_pnl_pct": round(session.total_pnl, 2),
        "avg_pnl_pct": round(session.total_pnl / len(trades), 2) if trades else 0,
        "best_trade": round(max(trade.pnl_pct for trade in trades), 2) if trades else 0,
        "worst_trade": round(min(trade.pnl_pct for trade in trades), 2) if trades else 0,
        "bars_processed": len(session.bars),
        "pre_market_bars": len(session.pre_market_bars),
        "regime_history": list(session.regime_history),
        "success": session.total_pnl > 0,
    }


def clear_session_state(
    *,
    manager: Any,
    run_id: str,
    ticker: str,
    date: str,
) -> bool:
    """Remove one session and clean sticky run/ticker memory when applicable."""
    key = manager._get_session_key(run_id, ticker, date)
    if key not in manager.sessions:
        return False

    del manager.sessions[key]
    manager.last_trade_bar_index.pop(key, None)
    defaults_key = (run_id, ticker)
    memory_key = manager._intraday_memory_key(run_id, ticker)

    has_other_sessions = any(
        session.run_id == run_id and session.ticker == ticker
        for session in manager.sessions.values()
    )
    if not has_other_sessions:
        manager.run_defaults.pop(defaults_key, None)
        manager.intraday_memory.day_memory.pop(memory_key, None)
        manager.intraday_memory.export_marker = {
            marker_key: marker_val
            for marker_key, marker_val in manager.intraday_memory.export_marker.items()
            if not (
                marker_key[0] == str(run_id)
                and marker_key[1] == str(ticker).upper()
            )
        }

    has_other_run_sessions = any(
        session.run_id == run_id for session in manager.sessions.values()
    )
    if not has_other_run_sessions:
        manager.run_equity_state.pop(run_id, None)

    return True


def clear_sessions_for_run_state(
    *,
    manager: Any,
    run_id: str,
    ticker: Optional[str] = None,
) -> int:
    """Clear all sessions/run-scoped state for a run id, optionally by ticker."""
    normalized_ticker = str(ticker).upper() if ticker else None

    keys_to_clear = [
        key
        for key, session in manager.sessions.items()
        if session.run_id == run_id
        and (normalized_ticker is None or session.ticker == normalized_ticker)
    ]

    for key in keys_to_clear:
        manager.sessions.pop(key, None)
        manager.last_trade_bar_index.pop(key, None)

    defaults_to_clear = [
        key
        for key in manager.run_defaults.keys()
        if key[0] == run_id
        and (normalized_ticker is None or key[1] == normalized_ticker)
    ]
    for key in defaults_to_clear:
        manager.run_defaults.pop(key, None)

    memory_to_clear = [
        key
        for key in manager.intraday_memory.day_memory.keys()
        if key[0] == str(run_id)
        and (normalized_ticker is None or key[1] == normalized_ticker)
    ]
    for key in memory_to_clear:
        manager.intraday_memory.day_memory.pop(key, None)

    manager.intraday_memory.export_marker = {
        key: value
        for key, value in manager.intraday_memory.export_marker.items()
        if not (
            key[0] == str(run_id)
            and (normalized_ticker is None or key[1] == normalized_ticker)
        )
    }

    has_other_run_sessions = any(
        session.run_id == run_id for session in manager.sessions.values()
    )
    if not has_other_run_sessions:
        manager.run_equity_state.pop(run_id, None)

    return len(keys_to_clear)


def reset_backtest_state(
    *,
    manager: Any,
    scope: str = "all",
    clear_sessions: bool = True,
) -> Dict[str, Any]:
    """Reset manager/orchestrator state for deterministic backtest reruns."""
    normalized_scope = str(scope or "all").strip().lower()
    if normalized_scope not in {"session", "learning", "all"}:
        raise ValueError(f"Unsupported reset scope: {scope}")

    sessions_before = len(manager.sessions)
    defaults_before = len(manager.run_defaults)

    if clear_sessions:
        manager.sessions.clear()
        manager.last_trade_bar_index.clear()
        manager.run_defaults.clear()
        manager.run_equity_state.clear()
        manager.intraday_memory.day_memory.clear()
        manager.intraday_memory.export_marker.clear()

    orchestrator = manager.orchestrator
    if orchestrator:
        if normalized_scope == "all":
            orchestrator.full_reset()
        elif normalized_scope == "session":
            orchestrator.new_session()
        elif normalized_scope == "learning":
            orchestrator.reset_learning_state()

    return {
        "scope": normalized_scope,
        "sessions_before": sessions_before,
        "sessions_after": len(manager.sessions),
        "run_defaults_before": defaults_before,
        "run_defaults_after": len(manager.run_defaults),
        "clear_sessions": bool(clear_sessions),
        "orchestrator_present": orchestrator is not None,
    }


def close_session_and_collect_summary(
    *,
    manager: Any,
    session: TradingSession,
) -> Dict[str, Any]:
    """Close active position if needed, persist memory, and return summary."""
    if session.active_position and session.bars:
        last_bar = session.bars[-1]
        manager._close_position(
            session,
            last_bar.close,
            last_bar.timestamp,
            "manual_close",
            bar_volume=last_bar.volume,
        )

    session.phase = SessionPhase.CLOSED
    if session.bars:
        session.end_price = session.bars[-1].close
    manager._persist_intraday_levels_memory(session)

    return build_session_summary(session)


__all__ = [
    "build_session_summary",
    "clear_session_state",
    "clear_sessions_for_run_state",
    "close_session_and_collect_summary",
    "reset_backtest_state",
]
