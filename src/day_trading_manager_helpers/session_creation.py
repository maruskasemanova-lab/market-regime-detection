"""Session creation helpers for DayTradingManager."""

from __future__ import annotations

from typing import Any

from ..day_trading_models import SessionPhase, TradingSession
from .session_config import build_ticker_session_config_payload
from .session_lifecycle import attach_orchestrator_to_new_session


def create_session_with_defaults(
    *,
    manager: Any,
    key: str,
    run_id: str,
    ticker: str,
    date: str,
    regime_detection_minutes: int,
    cold_start_each_day: bool,
) -> TradingSession:
    """Create, configure, and attach runtime dependencies for a new session."""

    # Refresh AOS config for newly created sessions so dashboard updates are picked up.
    manager._load_aos_config()
    # Defensive reset: if a previous run used the same session key, stale cooldown
    # state must not leak into the new replay.
    manager.last_trade_bar_index.pop(key, None)

    session = TradingSession(
        run_id=run_id,
        ticker=ticker,
        date=date,
        regime_detection_minutes=regime_detection_minutes,
    )
    manager.sessions[key] = session

    ticker_cfg = manager.ticker_params.get(str(ticker).upper(), {})
    if not isinstance(ticker_cfg, dict):
        ticker_cfg = {}

    config_payload = build_ticker_session_config_payload(
        manager,
        ticker_cfg=ticker_cfg,
        regime_detection_minutes=regime_detection_minutes,
    )
    base_config = manager._canonical_trading_config(config_payload)
    manager._apply_trading_config_to_session(session, base_config, normalize_momentum=False)
    manager._inject_intraday_levels_memory_into_session(session)

    # Attach orchestrator and reset session state for new day.
    attach_orchestrator_to_new_session(
        session,
        manager.orchestrator,
        cold_start_each_day=cold_start_each_day,
    )

    # Optional config-driven day filter.
    if not manager._is_day_allowed(date, ticker):
        session.phase = SessionPhase.CLOSED

    return session


__all__ = ["create_session_with_defaults"]

