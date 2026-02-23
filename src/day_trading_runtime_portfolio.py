"""Portfolio and PnL helper functions for runtime processing."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .day_trading_models import TradingSession


def unrealized_pnl_dollars(
    session: TradingSession,
    current_price: float,
    *,
    trading_costs: Any,
    bar_volume: Optional[float] = None,
) -> float:
    """Estimate unrealized mark-to-market PnL including conservative costs."""
    pos = session.active_position
    if pos is None or current_price <= 0:
        return 0.0

    shares = max(0.0, float(pos.size))
    if shares <= 0:
        return 0.0

    if pos.side == "long":
        gross_pnl = (current_price - pos.entry_price) * shares
    else:
        gross_pnl = (pos.entry_price - current_price) * shares

    costs_total = 0.0
    try:
        costs = trading_costs.calculate_costs(
            entry_price=pos.entry_price,
            exit_price=current_price,
            shares=shares,
            side=pos.side,
            avg_bar_volume=bar_volume,
        )
        costs_total = float(costs.get("total", 0.0) or 0.0)
    except Exception:
        costs_total = 0.0
    return gross_pnl - costs_total


def cooldown_bars_remaining(session: TradingSession, current_bar_index: int) -> int:
    if session.loss_cooldown_until_bar_index < current_bar_index:
        return 0
    return int(session.loss_cooldown_until_bar_index - current_bar_index + 1)


def realized_pnl_dollars(session: TradingSession) -> float:
    return float(sum(float(t.pnl_dollars or 0.0) for t in session.trades))


def portfolio_drawdown_snapshot(
    manager: Any,
    *,
    run_id: str,
    mark_session: Optional[TradingSession] = None,
    mark_price: Optional[float] = None,
    mark_bar_volume: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute run-level equity and drawdown snapshot.

    Equity model:
      baseline_equity + cumulative_realized_pnl + cumulative_unrealized_pnl

    baseline_equity is anchored from the highest configured account size seen
    for the run to avoid shrinking denominator artifacts across session changes.
    """
    threshold_pct = max(
        0.0,
        float(getattr(manager, "portfolio_drawdown_halt_pct", 0.0) or 0.0),
    )
    state_map = getattr(manager, "run_equity_state", None)
    if not isinstance(state_map, dict):
        state_map = {}
        setattr(manager, "run_equity_state", state_map)

    run_sessions = [
        s for s in manager.sessions.values()
        if getattr(s, "run_id", "") == run_id
    ]
    if not run_sessions:
        return {
            "enabled": threshold_pct > 0.0,
            "run_id": run_id,
            "session_count": 0,
            "drawdown_pct": 0.0,
            "halt_threshold_pct": threshold_pct,
            "halted": False,
            "halt_triggered": False,
            "current_equity": 0.0,
            "peak_equity": 0.0,
            "base_equity": 0.0,
            "realized_pnl_dollars": 0.0,
            "unrealized_pnl_dollars": 0.0,
        }

    state = state_map.get(run_id, {})
    if not isinstance(state, dict):
        state = {}

    account_candidates = [
        max(0.0, float(getattr(s, "account_size_usd", 0.0) or 0.0))
        for s in run_sessions
    ]
    observed_baseline = max(account_candidates) if account_candidates else 0.0
    base_equity = max(
        float(state.get("base_equity", 0.0) or 0.0),
        observed_baseline,
    )
    if base_equity <= 0.0:
        base_equity = 10_000.0

    realized_pnl = 0.0
    unrealized_pnl = 0.0
    for s in run_sessions:
        realized_pnl += realized_pnl_dollars(s)
        if not s.active_position:
            continue
        if s is mark_session and mark_price is not None:
            px = float(mark_price)
            vol = mark_bar_volume
        elif s.bars:
            px = float(s.bars[-1].close)
            vol = s.bars[-1].volume
        else:
            px = float(getattr(s.active_position, "entry_price", 0.0) or 0.0)
            vol = None
        unrealized_pnl += unrealized_pnl_dollars(
            s,
            px,
            trading_costs=manager.trading_costs,
            bar_volume=vol,
        )

    current_equity = base_equity + realized_pnl + unrealized_pnl
    peak_equity = max(
        base_equity,
        float(state.get("peak_equity", base_equity) or base_equity),
        current_equity,
    )
    drawdown_pct = (
        ((current_equity - peak_equity) / peak_equity) * 100.0
        if peak_equity > 0.0 else 0.0
    )

    was_halted = bool(state.get("halted", False))
    halted = was_halted
    halt_triggered = False
    if threshold_pct > 0.0 and drawdown_pct <= -threshold_pct:
        halted = True
        halt_triggered = not was_halted

    state.update(
        {
            "base_equity": float(base_equity),
            "peak_equity": float(peak_equity),
            "current_equity": float(current_equity),
            "realized_pnl_dollars": float(realized_pnl),
            "unrealized_pnl_dollars": float(unrealized_pnl),
            "drawdown_pct": float(drawdown_pct),
            "halt_threshold_pct": float(threshold_pct),
            "halted": bool(halted),
        }
    )
    if halt_triggered:
        state["halted_at_equity"] = float(current_equity)
        state["halted_at_drawdown_pct"] = float(drawdown_pct)
    state_map[run_id] = state

    return {
        "enabled": threshold_pct > 0.0,
        "run_id": run_id,
        "session_count": len(run_sessions),
        "base_equity": round(float(base_equity), 4),
        "peak_equity": round(float(peak_equity), 4),
        "current_equity": round(float(current_equity), 4),
        "realized_pnl_dollars": round(float(realized_pnl), 4),
        "unrealized_pnl_dollars": round(float(unrealized_pnl), 4),
        "drawdown_pct": round(float(drawdown_pct), 4),
        "halt_threshold_pct": round(float(threshold_pct), 4),
        "halted": bool(halted),
        "halt_triggered": bool(halt_triggered),
    }
