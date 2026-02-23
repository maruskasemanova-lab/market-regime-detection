from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.day_trading_manager import DayTradingManager
from src.day_trading_models import BarData, TradingSession
from src.exit_policy_engine import ExitPolicyEngine
from src.strategies.base_strategy import Position


def _bar(
    *,
    ts: datetime,
    close: float,
    high: float,
    low: float,
    l2_delta: float = 0.0,
    l2_imbalance: float = 0.0,
    l2_book_pressure: float = 0.0,
) -> BarData:
    return BarData(
        timestamp=ts,
        open=close,
        high=high,
        low=low,
        close=close,
        volume=10_000.0,
        l2_delta=l2_delta,
        l2_imbalance=l2_imbalance,
        l2_book_pressure=l2_book_pressure,
    )


def _session_for_break_even(ts: datetime) -> TradingSession:
    session = TradingSession(run_id="run-be", ticker="MU", date="2026-02-19")
    session.break_even_costs_pct = 0.0
    session.break_even_buffer_pct = 0.0
    session.break_even_min_buffer_pct = 0.0
    session.break_even_atr_buffer_k = 0.0
    session.break_even_5m_atr_buffer_k = 0.0
    session.break_even_tick_size = 0.01
    session.break_even_min_tick_buffer = 0
    session.break_even_min_hold_bars = 1
    session.bars = [_bar(ts=ts, close=100.2, high=100.4, low=99.8)]
    return session


def _position(ts: datetime) -> Position:
    return Position(
        strategy_name="VWAPMagnet",
        entry_price=100.0,
        entry_time=ts,
        side="long",
        size=10.0,
        stop_loss=99.0,
        take_profit=104.0,
        initial_size=10.0,
        initial_stop_loss=99.0,
        entry_bar_index=0,
    )


def test_force_move_break_even_includes_reason_costs_buffer_and_partial_fields() -> None:
    ts = datetime(2026, 2, 19, 14, 30, tzinfo=timezone.utc)
    session = _session_for_break_even(ts)
    pos = _position(ts)
    pos.partial_tp_filled = True
    pos.partial_tp_size = 5.0
    pos.partial_realized_r = 0.5

    snapshot = ExitPolicyEngine.force_move_to_break_even(
        session=session,
        pos=pos,
        bar=session.bars[-1],
        reason="partial_take_profit_protect",
    )

    assert snapshot["break_even_move_reason"] == "partial_protect"
    assert isinstance(snapshot["break_even_costs_pct"], float)
    assert isinstance(snapshot["break_even_buffer_pct"], float)
    assert snapshot["partial_tp_filled"] is True
    assert snapshot["partial_tp_size"] == pytest.approx(5.0)
    assert snapshot["partial_realized_r"] == pytest.approx(0.5)
    assert isinstance(snapshot.get("computed_break_even"), dict)


def test_update_trailing_keeps_computed_break_even_when_proof_fails_after_partial_move() -> None:
    ts = datetime(2026, 2, 19, 14, 30, tzinfo=timezone.utc)
    session = _session_for_break_even(ts)
    pos = _position(ts)
    pos.partial_tp_filled = True
    pos.partial_tp_size = 5.0
    pos.partial_realized_r = 0.5

    ExitPolicyEngine.force_move_to_break_even(
        session=session,
        pos=pos,
        bar=session.bars[-1],
        reason="partial_take_profit_protect",
    )

    next_bar = _bar(
        ts=ts.replace(minute=31),
        close=100.15,
        high=100.2,
        low=100.1,
        l2_delta=0.0,
        l2_imbalance=0.0,
        l2_book_pressure=0.0,
    )
    session.bars.append(next_bar)
    diagnostics = ExitPolicyEngine.update_trailing_from_close(session, pos, next_bar)

    assert diagnostics["proof_passed"] is False
    assert diagnostics["break_even_move_reason"] == "partial_protect"
    assert isinstance(diagnostics.get("computed_break_even"), dict)
    assert diagnostics["break_even_costs_pct"] == pytest.approx(
        diagnostics["computed_break_even"]["total_costs_pct"]
    )
    assert diagnostics["break_even_buffer_pct"] == pytest.approx(
        diagnostics["computed_break_even"]["buffer"]["selected_buffer_pct"]
    )


def test_partial_take_profit_sets_partial_audit_fields_for_break_even(monkeypatch) -> None:
    manager = DayTradingManager(regime_detection_minutes=180)
    ts = datetime(2026, 2, 19, 14, 30, tzinfo=timezone.utc)
    session = TradingSession(run_id="run-partial", ticker="MU", date="2026-02-19")
    bar = _bar(ts=ts, close=101.0, high=101.2, low=99.8)
    session.bars = [bar]
    pos = _position(ts)
    pos.signal_metadata = {"break_even": {}}
    session.active_position = pos

    monkeypatch.setattr(
        manager.exit_engine,
        "should_take_partial_profit",
        lambda **_: {
            "reason": "partial_take_profit",
            "partial_price": 101.0,
            "exit_price": 101.0,
            "close_fraction": 0.5,
        },
    )

    trade = manager._maybe_take_partial_profit(
        session=session,
        pos=pos,
        bar=bar,
        timestamp=ts,
    )

    assert trade is not None
    assert pos.partial_tp_filled is True
    assert pos.partial_tp_size == pytest.approx(5.0)
    assert pos.partial_realized_r == pytest.approx(0.5)
    assert pos.break_even_last_update["break_even_move_reason"] == "partial_protect"


def test_break_even_activation_formula_can_block_move_even_when_base_conditions_pass() -> None:
    ts = datetime(2026, 2, 19, 14, 30, tzinfo=timezone.utc)
    session = _session_for_break_even(ts)
    session.break_even_activation_use_levels = False
    session.break_even_activation_use_l2 = False
    session.break_even_activation_min_mfe_pct = 0.0
    session.break_even_activation_min_r = 0.0
    session.break_even_activation_min_r_trending_5m = 0.0
    session.break_even_activation_min_r_choppy_5m = 0.0
    session.break_even_activation_formula_enabled = True
    session.break_even_activation_formula = "False"
    pos = _position(ts)

    next_bar = _bar(
        ts=ts.replace(minute=31),
        close=100.4,
        high=100.6,
        low=100.1,
    )
    session.bars.append(next_bar)
    diagnostics = ExitPolicyEngine.update_trailing_from_close(session, pos, next_bar)

    assert diagnostics["movement_passed"] is True
    assert diagnostics["proof_passed"] is True
    assert diagnostics["activation_eligible"] is False
    assert diagnostics["active"] is False
    assert pos.break_even_stop_active is False
    runtime_formulas = diagnostics.get("runtime_formulas", {})
    activation_formula = runtime_formulas.get("break_even_activation", {})
    assert activation_formula.get("enabled") is True
    assert activation_formula.get("passed") is False


def test_trailing_handoff_formula_can_block_trailing_update_after_break_even() -> None:
    ts = datetime(2026, 2, 19, 14, 30, tzinfo=timezone.utc)
    session = _session_for_break_even(ts)
    session.break_even_activation_use_levels = False
    session.break_even_activation_use_l2 = False
    session.break_even_activation_min_mfe_pct = 0.0
    session.break_even_activation_min_r = 0.0
    session.break_even_activation_min_r_trending_5m = 0.0
    session.break_even_activation_min_r_choppy_5m = 0.0
    session.trailing_stop_pct = 0.5
    session.break_even_trailing_handoff_formula_enabled = True
    session.break_even_trailing_handoff_formula = "False"
    pos = _position(ts)
    pos.trailing_stop_active = True

    be_bar = _bar(
        ts=ts.replace(minute=31),
        close=100.5,
        high=100.7,
        low=100.2,
    )
    session.bars.append(be_bar)
    first_diag = ExitPolicyEngine.update_trailing_from_close(session, pos, be_bar)
    assert pos.break_even_stop_active is True
    assert pos.trailing_activation_pnl_met is True
    assert first_diag["runtime_formulas"]["break_even_trailing_handoff"]["enabled"] is True

    before_trailing = float(pos.trailing_stop_price or 0.0)
    trail_bar = _bar(
        ts=ts.replace(minute=32),
        close=100.8,
        high=101.0,
        low=100.6,
    )
    session.bars.append(trail_bar)
    diagnostics = ExitPolicyEngine.update_trailing_from_close(session, pos, trail_bar)

    assert diagnostics["trailing_can_update"] is False
    assert diagnostics["trailing_updated"] is False
    assert float(pos.trailing_stop_price or 0.0) == pytest.approx(before_trailing)
    trailing_formula = diagnostics["runtime_formulas"]["break_even_trailing_handoff"]
    assert trailing_formula.get("enabled") is True
    assert trailing_formula.get("passed") is False
