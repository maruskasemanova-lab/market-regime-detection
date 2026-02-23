"""Tests for pending-signal entry handling."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from src.day_trading_runtime_pending import process_pending_signal_entry
from src.strategies.base_strategy import SignalType
from src.trading_config import TradingConfig


class _DummySignal:
    def __init__(self) -> None:
        self.strategy_name = "momentum"
        self.signal_type = SignalType.BUY
        self.stop_loss = 99.0
        self.take_profit = 101.0
        self.reasoning = "test"
        self.confidence = 70.0
        self.metadata = {}


class _DummyPosition:
    def __init__(self, *, entry_price: float, strategy_name: str, signal_metadata: dict) -> None:
        self.size = 1.0
        self.entry_price = entry_price
        self.side = "long"
        self.strategy_name = strategy_name
        self.fill_ratio = 1.0
        self.stop_loss = 99.0
        self.take_profit = 101.0
        self.partial_take_profit_price = 100.5
        self.signal_metadata = dict(signal_metadata)

    def to_dict(self) -> dict:
        return {"size": self.size, "entry_price": self.entry_price, "side": self.side}


class _DummyManager:
    def __init__(self) -> None:
        self.pending_signal_ttl_bars = 3
        self.last_trade_bar_index = {}
        self.open_calls = 0

    @staticmethod
    def _get_session_key(run_id: str, ticker: str, date: str) -> str:
        return f"{run_id}:{ticker}:{date}"

    @staticmethod
    def _intraday_levels_indicator_payload(_session) -> dict:
        return {}

    def _open_position(
        self,
        _session,
        signal,
        *,
        entry_price: float,
        entry_time,
        signal_bar_index: int,
        entry_bar_index: int,
        entry_bar_volume: float,
    ):
        del entry_time, signal_bar_index, entry_bar_index, entry_bar_volume
        self.open_calls += 1
        return _DummyPosition(
            entry_price=entry_price,
            strategy_name=signal.strategy_name,
            signal_metadata=signal.metadata,
        )


def test_unknown_micro_regime_does_not_force_pending_drop() -> None:
    manager = _DummyManager()
    signal = _DummySignal()
    session = SimpleNamespace(
        pending_signal=signal,
        active_position=None,
        pending_signal_bar_index=0,
        micro_regime="UNKNOWN",
        loss_cooldown_until_bar_index=-1,
        config=TradingConfig(),
        run_id="run",
        ticker="MU",
        date="2026-02-04",
    )
    bar = SimpleNamespace(open=100.0, volume=25_000.0)
    result: dict = {}

    should_return_early = process_pending_signal_entry(
        manager=manager,
        session=session,
        bar=bar,
        timestamp=datetime(2026, 2, 4, 14, 31),
        current_bar_index=1,
        result=result,
        formula_indicators=lambda: {},
    )

    assert should_return_early is False
    assert manager.open_calls == 1
    assert result.get("regime_warmup") is True
    assert result.get("action") == "position_opened"
    assert result.get("dropped_pending_signal") is not True
