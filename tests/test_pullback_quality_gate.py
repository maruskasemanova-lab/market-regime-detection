from datetime import datetime, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from src.day_trading_models import TradingSession
from src.day_trading_runtime_trading_bar_helpers.entry_pipeline_signal_execution_confirmed import (
    _apply_pullback_quality_gate,
)
from src.strategies.base_strategy import Regime, Signal, SignalType
from src.trading_config import TradingConfig


class _ManagerStub:
    def _canonical_strategy_key(self, strategy_name: str) -> str:
        return str(strategy_name or "").strip().lower()

    def _to_market_time(self, ts: datetime) -> datetime:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(ZoneInfo("America/New_York"))


def _pullback_signal(ts: datetime) -> Signal:
    return Signal(
        strategy_name="pullback",
        signal_type=SignalType.BUY,
        price=100.0,
        timestamp=ts,
        confidence=72.0,
    )


def _base_result(poc_on_trade_side: bool = True) -> dict:
    return {
        "level_context": {
            "volume_profile": {
                "effective_poc_price": 100.8,
                "poc_on_trade_side": poc_on_trade_side,
            }
        }
    }


def test_pullback_quality_gate_blocks_outside_morning_window() -> None:
    manager = _ManagerStub()
    ts = datetime(2026, 2, 20, 17, 0, tzinfo=timezone.utc)  # 12:00 ET
    session = TradingSession(run_id="r", ticker="MU", date="2026-02-20")
    session.config = TradingConfig()
    session.micro_regime = "TRENDING_UP"
    decision = SimpleNamespace(combined_score=77.0)
    result = _base_result()

    blocked = _apply_pullback_quality_gate(
        manager,
        session=session,
        signal=_pullback_signal(ts),
        flow_metrics={"price_trend_efficiency": 0.25},
        decision=decision,
        effective_trade_threshold=60.0,
        regime=Regime.TRENDING,
        tod_boost=0.0,
        timestamp=ts,
        result=result,
    )

    assert blocked is True
    assert result["action"] == "pullback_quality_filtered"
    assert result["reason"] == "pullback_outside_morning_window"


def test_pullback_quality_gate_blocks_low_trend_efficiency() -> None:
    manager = _ManagerStub()
    ts = datetime(2026, 2, 20, 15, 30, tzinfo=timezone.utc)  # 10:30 ET
    session = TradingSession(run_id="r", ticker="MU", date="2026-02-20")
    session.config = TradingConfig()
    session.micro_regime = "TRENDING_UP"
    decision = SimpleNamespace(combined_score=77.0)
    result = _base_result()

    blocked = _apply_pullback_quality_gate(
        manager,
        session=session,
        signal=_pullback_signal(ts),
        flow_metrics={"price_trend_efficiency": 0.08},
        decision=decision,
        effective_trade_threshold=60.0,
        regime=Regime.TRENDING,
        tod_boost=0.0,
        timestamp=ts,
        result=result,
    )

    assert blocked is True
    assert result["reason"] == "pullback_trend_efficiency_too_low"


def test_pullback_quality_gate_blocks_poc_mismatch() -> None:
    manager = _ManagerStub()
    ts = datetime(2026, 2, 20, 15, 30, tzinfo=timezone.utc)  # 10:30 ET
    session = TradingSession(run_id="r", ticker="MU", date="2026-02-20")
    session.config = TradingConfig()
    session.micro_regime = "TRENDING_UP"
    decision = SimpleNamespace(combined_score=77.0)
    result = _base_result(poc_on_trade_side=False)

    blocked = _apply_pullback_quality_gate(
        manager,
        session=session,
        signal=_pullback_signal(ts),
        flow_metrics={"price_trend_efficiency": 0.25},
        decision=decision,
        effective_trade_threshold=60.0,
        regime=Regime.TRENDING,
        tod_boost=0.0,
        timestamp=ts,
        result=result,
    )

    assert blocked is True
    assert result["reason"] == "pullback_requires_poc_on_trade_side"


def test_pullback_quality_gate_accepts_high_quality_setup() -> None:
    manager = _ManagerStub()
    ts = datetime(2026, 2, 20, 15, 30, tzinfo=timezone.utc)  # 10:30 ET
    session = TradingSession(run_id="r", ticker="MU", date="2026-02-20")
    session.config = TradingConfig()
    session.micro_regime = "TRENDING_UP"
    decision = SimpleNamespace(combined_score=77.0)
    result = _base_result(poc_on_trade_side=True)

    blocked = _apply_pullback_quality_gate(
        manager,
        session=session,
        signal=_pullback_signal(ts),
        flow_metrics={"price_trend_efficiency": 0.25},
        decision=decision,
        effective_trade_threshold=60.0,
        regime=Regime.TRENDING,
        tod_boost=0.0,
        timestamp=ts,
        result=result,
    )

    assert blocked is False
    assert "action" not in result
