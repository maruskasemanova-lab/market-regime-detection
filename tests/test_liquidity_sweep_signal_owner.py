from datetime import datetime, timezone

from src.day_trading_manager import DayTradingManager
from src.day_trading_runtime_trading_bar_helpers.decision_context_sweep_confirmed_signal import (
    _handle_confirmed_liquidity_sweep_signal,
)


def _confirmed_sweep(direction: str = "long") -> dict:
    return {
        "confirmed": True,
        "direction": direction,
        "level_price": 99.5 if direction == "long" else 100.5,
        "vwap_execution_flow": 0.25,
    }


def test_confirmed_sweep_prefers_highest_priority_enabled_active_strategy() -> None:
    manager = DayTradingManager()
    session = manager.get_or_create_session("run", "MU", "2026-02-24")
    session.selected_strategy = "adaptive"
    session.active_strategies = ["evidence_scalp", "pullback", "momentum_flow"]

    manager.strategies["evidence_scalp"].liquidity_sweep_signal_enabled = False
    manager.strategies["pullback"].liquidity_sweep_signal_enabled = True
    manager.strategies["pullback"].liquidity_sweep_signal_priority = 100
    manager.strategies["momentum_flow"].liquidity_sweep_signal_enabled = True
    manager.strategies["momentum_flow"].liquidity_sweep_signal_priority = 10

    result: dict = {}
    handled = _handle_confirmed_liquidity_sweep_signal(
        manager,
        session=session,
        timestamp=datetime(2026, 2, 24, 15, 0, tzinfo=timezone.utc),
        current_bar_index=12,
        current_price=100.0,
        indicators={"atr": [1.0]},
        bars_data=[],
        flow_metrics={"has_l2_coverage": True},
        sweep_confirmation=_confirmed_sweep(),
        result=result,
    )

    assert handled is True
    assert session.pending_signal is not None
    assert session.pending_signal.strategy_name == "Pullback"
    assert result["signal"]["strategy"] == "Pullback"
    assert result["liquidity_sweep_confirmation"]["strategy_owner"] == "pullback"
    assert result["liquidity_sweep_confirmation"]["strategy_owner_reason"] == (
        "active_strategy_priority"
    )
    assert result["liquidity_sweep_confirmation"]["signal_queued"] is True


def test_confirmed_sweep_skips_disabled_selected_strategy() -> None:
    manager = DayTradingManager()
    session = manager.get_or_create_session("run", "MU", "2026-02-24")
    session.selected_strategy = "evidence_scalp"
    session.active_strategies = ["evidence_scalp"]

    manager.strategies["evidence_scalp"].liquidity_sweep_signal_enabled = False

    result: dict = {}
    handled = _handle_confirmed_liquidity_sweep_signal(
        manager,
        session=session,
        timestamp=datetime(2026, 2, 24, 15, 0, tzinfo=timezone.utc),
        current_bar_index=12,
        current_price=100.0,
        indicators={"atr": [1.0]},
        bars_data=[],
        flow_metrics={"has_l2_coverage": True},
        sweep_confirmation=_confirmed_sweep(),
        result=result,
    )

    assert handled is False
    assert session.pending_signal is None
    assert not session.signals
    assert result["liquidity_sweep_confirmation"]["strategy_owner"] is None
    assert result["liquidity_sweep_confirmation"]["strategy_owner_reason"] == (
        "selected_strategy_liquidity_sweep_disabled"
    )
    assert result["liquidity_sweep_confirmation"]["signal_queued"] is False


def test_confirmed_sweep_skips_when_no_active_strategy_allows_it() -> None:
    manager = DayTradingManager()
    session = manager.get_or_create_session("run", "MU", "2026-02-24")
    session.selected_strategy = "adaptive"
    session.active_strategies = ["evidence_scalp", "momentum_flow"]

    manager.strategies["evidence_scalp"].liquidity_sweep_signal_enabled = False
    manager.strategies["momentum_flow"].liquidity_sweep_signal_enabled = False

    result: dict = {}
    handled = _handle_confirmed_liquidity_sweep_signal(
        manager,
        session=session,
        timestamp=datetime(2026, 2, 24, 15, 0, tzinfo=timezone.utc),
        current_bar_index=12,
        current_price=100.0,
        indicators={"atr": [1.0]},
        bars_data=[],
        flow_metrics={"has_l2_coverage": True},
        sweep_confirmation=_confirmed_sweep(),
        result=result,
    )

    assert handled is False
    assert session.pending_signal is None
    assert not session.signals
    assert result["liquidity_sweep_confirmation"]["strategy_owner"] is None
    assert result["liquidity_sweep_confirmation"]["strategy_owner_reason"] == (
        "no_active_strategy_liquidity_sweep_enabled"
    )
    assert result["liquidity_sweep_confirmation"]["signal_queued"] is False
