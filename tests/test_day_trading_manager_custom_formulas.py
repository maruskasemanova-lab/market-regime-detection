from __future__ import annotations
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from datetime import datetime, timezone

from src.day_trading_manager import BarData, DayTradingManager, SessionPhase
from src.multi_layer_decision import DecisionResult
from src.strategies.base_strategy import Position, Regime, Signal, SignalType


def _signal(
    strategy_name: str = "mean_reversion",
    signal_type: SignalType = SignalType.BUY,
    metadata: Optional[dict] = None,
) -> Signal:
    return Signal(
        strategy_name=strategy_name,
        signal_type=signal_type,
        price=100.0,
        timestamp=datetime(2026, 2, 3, 14, 30, tzinfo=timezone.utc),
        confidence=80.0,
        stop_loss=99.0,
        take_profit=103.0,
        trailing_stop=False,
        reasoning="formula-test",
        metadata=metadata or {},
    )


class _StubEvidenceEngine:
    def __init__(self, fixed_decision: DecisionResult):
        self._decision = fixed_decision

    def evaluate(self, **_kwargs) -> DecisionResult:
        return self._decision


class _StubConfig:
    use_evidence_engine = True


class _StubOrchestrator:
    def __init__(self, decision: DecisionResult):
        self.config = _StubConfig()
        self.evidence_engine = _StubEvidenceEngine(decision)
        self.current_feature_vector = None
        self.current_regime_state = None
        self.current_cross_asset_state = None


def test_custom_entry_formula_can_reject_signal() -> None:
    manager = DayTradingManager(
        regime_detection_minutes=0,
        trade_cooldown_bars=0,
        max_trades_per_day=10,
    )
    session = manager.get_or_create_session("run", "MU", "2026-02-03")
    session.phase = SessionPhase.TRADING
    session.selected_strategy = "mean_reversion"
    session.detected_regime = Regime.TRENDING
    session.micro_regime = "TRENDING_UP"
    session.active_strategies = ["mean_reversion"]
    # Isolate custom formula behavior from intraday-level entry gate.
    session.config = replace(session.config, intraday_levels_entry_quality_enabled=False)

    strategy = manager.strategies["mean_reversion"]
    strategy.custom_entry_formula_enabled = True
    strategy.custom_entry_formula = "flow_score > 999"

    ts = datetime(2026, 2, 3, 15, 0, tzinfo=timezone.utc)
    bar = BarData(
        timestamp=ts,
        open=100.0,
        high=100.2,
        low=99.8,
        close=100.0,
        volume=120_000.0,
        vwap=100.0,
    )
    session.bars.append(bar)

    decision = DecisionResult(
        execute=True,
        direction="bullish",
        signal=_signal(
            "mean_reversion",
            metadata={
                "layer_scores": {"confirming_sources": 5},
                "aligned_evidence_sources": 5,
            },
        ),
        combined_score=80.0,
        combined_raw=80.0,
        combined_norm_0_100=80.0,
        strategy_score=80.0,
        threshold=55.0,
        trade_gate_threshold=55.0,
    )
    session.orchestrator = _StubOrchestrator(decision)

    result = manager._process_trading_bar(session, bar, ts)

    assert result.get("action") == "custom_entry_formula_filtered"
    assert result.get("signal_rejected", {}).get("gate") == "custom_entry_formula"
    assert session.pending_signal is None


def test_custom_exit_formula_can_force_close_position() -> None:
    manager = DayTradingManager(
        regime_detection_minutes=0,
        trade_cooldown_bars=0,
        max_trades_per_day=10,
    )
    session = manager.get_or_create_session("run", "MU", "2026-02-03")
    session.phase = SessionPhase.TRADING
    session.detected_regime = Regime.MIXED
    session.micro_regime = "MIXED"
    session.selected_strategy = None
    session.active_strategies = []

    strategy = manager.strategies["mean_reversion"]
    strategy.custom_exit_formula_enabled = True
    strategy.custom_exit_formula = "position_side == 'long' and price >= entry_price"

    entry_ts = datetime(2026, 2, 3, 14, 45, tzinfo=timezone.utc)
    position = Position(
        strategy_name="mean_reversion",
        entry_price=100.0,
        entry_time=entry_ts,
        side="long",
        size=100,
        stop_loss=95.0,
        take_profit=110.0,
    )
    position.entry_bar_index = 0
    session.active_position = position

    ts = datetime(2026, 2, 3, 15, 0, tzinfo=timezone.utc)
    bar = BarData(
        timestamp=ts,
        open=100.1,
        high=100.5,
        low=99.9,
        close=100.2,
        volume=90_000.0,
        vwap=100.15,
    )
    session.bars.append(bar)

    result = manager._process_trading_bar(session, bar, ts)

    assert result.get("action") == "position_closed_custom_formula_exit"
    assert result.get("trade_closed") is not None
    assert result.get("custom_exit_formula", {}).get("enabled") is True
