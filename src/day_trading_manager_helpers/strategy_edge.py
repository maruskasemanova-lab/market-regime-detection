"""Strategy edge scoring helper for DayTradingManager."""

from __future__ import annotations

from typing import Any

from ..day_trading_models import TradingSession


def compute_strategy_edge_adjustment(
    *,
    manager: Any,
    session: TradingSession,
    strategy_key: str,
) -> float:
    """Estimate strategy edge from realized session trades with warmup ramp."""
    relevant = [
        trade
        for trade in session.trades
        if manager._canonical_strategy_key(getattr(trade, "strategy", "")) == strategy_key
    ]
    if not relevant:
        return 0.0

    trade_count = len(relevant)
    edge_warmup_trades = 3
    if trade_count < edge_warmup_trades:
        return 0.0

    ramp = min(1.0, (trade_count - edge_warmup_trades + 1) / 3.0)

    wins = [trade for trade in relevant if trade.pnl_pct > 0]
    losses = [trade for trade in relevant if trade.pnl_pct <= 0]
    win_rate = (len(wins) + 1.0) / (trade_count + 2.0)
    avg_win = (sum(trade.pnl_pct for trade in wins) / len(wins)) if wins else 0.0
    avg_loss = (abs(sum(trade.pnl_pct for trade in losses)) / len(losses)) if losses else 0.0
    expectancy = (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss)
    profit_factor = (
        (sum(trade.pnl_pct for trade in wins) / abs(sum(trade.pnl_pct for trade in losses)))
        if losses
        else (2.0 if wins else 0.0)
    )

    edge = ((win_rate - 0.5) * 36.0) + ((profit_factor - 1.0) * 6.0) + (expectancy * 4.0)
    edge = max(-12.0, min(12.0, edge))
    return edge * ramp


__all__ = [
    "compute_strategy_edge_adjustment",
]
