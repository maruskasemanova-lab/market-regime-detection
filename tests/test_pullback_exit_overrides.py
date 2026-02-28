from datetime import datetime, timezone

from src.day_trading_models import BarData, TradingSession
from src.exit_policy.break_even_activation import evaluate_break_even_activation_context
from src.exit_policy_engine import ExitPolicyEngine
from src.strategies.base_strategy import Position, Regime
from src.trading_config import TradingConfig


def _bar(ts: datetime, close: float, high: float, low: float) -> BarData:
    return BarData(
        timestamp=ts,
        open=close,
        high=high,
        low=low,
        close=close,
        volume=10_000.0,
        l2_delta=0.0,
        l2_imbalance=0.0,
        l2_book_pressure=0.0,
    )


def test_pullback_time_exit_uses_strategy_specific_limit() -> None:
    ts = datetime(2026, 2, 20, 15, 30, tzinfo=timezone.utc)
    session = TradingSession(run_id="r", ticker="MU", date="2026-02-20")
    session.apply_trading_config(
        TradingConfig(
            time_exit_bars=40,
            pullback_time_exit_bars=7,
        )
    )
    session.detected_regime = Regime.TRENDING
    session.bars = [_bar(ts, close=100.0, high=100.2, low=99.8)]

    pullback_pos = Position(
        strategy_name="pullback",
        entry_price=100.0,
        entry_time=ts,
        side="long",
        stop_loss=99.0,
        take_profit=102.0,
        entry_bar_index=0,
    )
    momentum_pos = Position(
        strategy_name="momentum",
        entry_price=100.0,
        entry_time=ts,
        side="long",
        stop_loss=99.0,
        take_profit=102.0,
        entry_bar_index=0,
    )

    pullback_exit = ExitPolicyEngine.should_time_exit(
        session=session,
        pos=pullback_pos,
        current_bar_index=7,
        flow_metrics={},
    )
    momentum_exit = ExitPolicyEngine.should_time_exit(
        session=session,
        pos=momentum_pos,
        current_bar_index=7,
        flow_metrics={},
    )

    assert pullback_exit is True
    assert momentum_exit is False


def test_pullback_break_even_uses_lower_r_and_can_skip_proof() -> None:
    ts = datetime(2026, 2, 20, 15, 30, tzinfo=timezone.utc)
    session = TradingSession(run_id="r", ticker="MU", date="2026-02-20")
    session.apply_trading_config(
        TradingConfig(
            break_even_min_hold_bars=1,
            break_even_activation_min_mfe_pct=0.0,
            break_even_activation_min_r=0.60,
            break_even_activation_use_levels=True,
            break_even_activation_use_l2=True,
            pullback_break_even_proof_required=False,
            pullback_break_even_activation_min_r=0.40,
        )
    )
    bar = _bar(ts, close=100.5, high=100.5, low=100.0)
    session.bars = [bar]
    pos = Position(
        strategy_name="pullback",
        entry_price=100.0,
        entry_time=ts,
        side="long",
        stop_loss=99.0,
        take_profit=103.0,
        entry_bar_index=0,
        highest_price=100.0,
    )
    session.active_position = pos

    ctx = evaluate_break_even_activation_context(
        session=session,
        pos=pos,
        bar=bar,
        current_bar_index=1,
    )

    assert ctx["strategy"] == "pullback"
    assert ctx["proof_required"] is False
    assert ctx["required_r"] == 0.4
    assert ctx["movement_passed"] is True
    assert ctx["activation_eligible"] is True
