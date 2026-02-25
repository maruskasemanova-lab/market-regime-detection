"""Session/orchestrator lifecycle helpers for DayTradingManager."""

from __future__ import annotations

from typing import Any


def _reset_orchestrator_session(orchestrator: Any, *, cold_start_each_day: bool) -> None:
    if cold_start_each_day:
        orchestrator.full_reset()
    else:
        orchestrator.new_session()


def recover_existing_session_orchestrator(
    session: Any,
    orchestrator: Any,
    *,
    cold_start_each_day: bool,
) -> None:
    """Recover a partially created session with a missing orchestrator reference."""

    if session.orchestrator is not None or orchestrator is None:
        return

    session.orchestrator = orchestrator
    if not session.bars and not session.pre_market_bars:
        _reset_orchestrator_session(orchestrator, cold_start_each_day=cold_start_each_day)


def attach_orchestrator_to_new_session(
    session: Any,
    orchestrator: Any,
    *,
    cold_start_each_day: bool,
) -> None:
    """Attach orchestrator to a newly created session and reset day state."""

    if orchestrator is None:
        return

    session.orchestrator = orchestrator
    _reset_orchestrator_session(orchestrator, cold_start_each_day=cold_start_each_day)
