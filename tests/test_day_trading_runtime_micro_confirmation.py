from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.day_trading_models import BarData, TradingSession
from src.day_trading_runtime_impl import (
    _intrabar_confirmation_snapshot,
    _micro_confirmation_snapshot,
)
from src.strategies.base_strategy import Signal, SignalType


def _bar(ts: datetime, close: float, intrabar_quotes_1s=None) -> BarData:
    return BarData(
        timestamp=ts,
        open=close - 0.05,
        high=close + 0.1,
        low=close - 0.1,
        close=close,
        volume=10_000.0,
        intrabar_quotes_1s=intrabar_quotes_1s,
    )


def _signal(ts: datetime, side: SignalType) -> Signal:
    return Signal(
        strategy_name="test_strategy",
        signal_type=side,
        price=100.0,
        timestamp=ts,
        confidence=80.0,
        stop_loss=99.0,
        take_profit=101.0,
        metadata={},
    )


def test_micro_confirmation_allows_zero_required_bars() -> None:
    ts = datetime(2026, 2, 12, 15, 30, tzinfo=timezone.utc)
    session = TradingSession(run_id="r1", ticker="MU", date="2026-02-12")
    session.bars = [_bar(ts, 100.0)]
    signal = _signal(ts, SignalType.BUY)

    snapshot = _micro_confirmation_snapshot(
        session=session,
        signal=signal,
        current_bar_index=1,
        signal_bar_index=0,
        required_bars=0,
    )

    assert snapshot["required_bars"] == 0
    assert snapshot["ready"] is True
    assert snapshot["passed"] is True
    assert snapshot["reason"] == "micro_confirmation_passed"


def test_intrabar_confirmation_passes_with_strong_long_push() -> None:
    ts = datetime(2026, 2, 12, 15, 31, tzinfo=timezone.utc)
    session = TradingSession(run_id="r2", ticker="MU", date="2026-02-12")
    session.bars = [
        _bar(
            ts,
            100.0,
            intrabar_quotes_1s=[
                {"s": 1, "bid": 100.00, "ask": 100.02},
                {"s": 2, "bid": 100.01, "ask": 100.03},
                {"s": 3, "bid": 100.02, "ask": 100.04},
                {"s": 4, "bid": 100.03, "ask": 100.05},
            ],
        )
    ]
    signal = _signal(ts, SignalType.BUY)

    snapshot = _intrabar_confirmation_snapshot(
        session=session,
        signal=signal,
        current_bar_index=1,
        signal_bar_index=0,
        window_seconds=5,
        min_coverage_points=2,
        min_move_pct=0.005,
        min_push_ratio=0.1,
        max_spread_bps=20.0,
    )

    assert snapshot["ready"] is True
    assert snapshot["passed"] is True
    assert snapshot["reason"] == "intrabar_confirmation_passed"


def test_intrabar_confirmation_rejects_low_push_ratio() -> None:
    ts = datetime(2026, 2, 12, 15, 32, tzinfo=timezone.utc)
    session = TradingSession(run_id="r3", ticker="MU", date="2026-02-12")
    session.bars = [
        _bar(
            ts,
            100.0,
            intrabar_quotes_1s=[
                {"s": 1, "bid": 99.99, "ask": 100.01},  # 100.00
                {"s": 2, "bid": 100.00, "ask": 100.02}, # 100.01
                {"s": 3, "bid": 100.01, "ask": 100.03}, # 100.02
                {"s": 4, "bid": 100.02, "ask": 100.04}, # 100.03
                {"s": 5, "bid": 100.01, "ask": 100.03}, # 100.02
                {"s": 6, "bid": 100.00, "ask": 100.02}, # 100.01
                {"s": 7, "bid": 100.01, "ask": 100.03}, # 100.02
                {"s": 8, "bid": 100.02, "ask": 100.04}, # 100.03
                {"s": 9, "bid": 100.03, "ask": 100.05}, # 100.04
            ],
        )
    ]
    signal = _signal(ts, SignalType.BUY)

    snapshot = _intrabar_confirmation_snapshot(
        session=session,
        signal=signal,
        current_bar_index=1,
        signal_bar_index=0,
        window_seconds=10,
        min_coverage_points=2,
        min_move_pct=0.035,
        min_push_ratio=0.8,
        max_spread_bps=20.0,
    )

    assert snapshot["ready"] is True
    assert snapshot["passed"] is False
    assert snapshot["reason"] == "intrabar_push_below_threshold"
