"""Tests for context-aware exit policies (Phase 3).

Validates that ContextAwareExitPolicy correctly adjusts exit behavior
in response to ContextChangeEvents from the PositionContextMonitor.
"""
import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.exit_policy_engine import (
    ContextAwareExitPolicy,
    ExitDecision,
    DEFAULT_CONTEXT_EXIT_CONFIG,
)
from src.position_context import (
    ContextChangeEvent,
    EntrySnapshot,
    PositionContextMonitor,
)


# ── Minimal stub classes for testing ─────────────────────────────────

@dataclass
class StubBar:
    close: float = 100.0
    high: float = 101.0
    low: float = 99.0
    open: float = 100.0
    volume: float = 1000.0
    timestamp: Any = None


@dataclass
class StubPosition:
    side: str = "long"
    entry_price: float = 100.0
    stop_loss: float = 98.0
    take_profit: float = 104.0
    trailing_stop_active: bool = False
    trailing_stop_price: float = 0.0
    break_even_stop_active: bool = False
    highest_price: float = 100.0
    lowest_price: float = 100.0
    size: float = 100.0
    strategy_name: str = "momentum_flow"
    entry_bar_index: int = 0
    signal_metadata: Dict = field(default_factory=dict)
    partial_exit_done: bool = False
    partial_take_profit_price: float = 0.0
    initial_size: float = 100.0
    fill_ratio: float = 1.0
    signal_bar_index: int = 0
    signal_timestamp: str = ""
    signal_price: float = 100.0
    trailing_activation_pnl_met: bool = False


@dataclass
class StubSession:
    bars: List[StubBar] = field(default_factory=list)
    active_position: Optional[StubPosition] = None
    time_exit_bars: int = 40
    choppy_time_exit_bars: int = 12
    break_even_buffer_pct: float = 0.03
    ticker: str = "MU"

    def __post_init__(self):
        if not self.bars:
            self.bars = [StubBar()]


# ── Helper to build events ───────────────────────────────────────────

def _regime_flip_event(severity="critical", bar_index=5):
    return ContextChangeEvent(
        event_type="regime_flip",
        severity=severity,
        bar_index=bar_index,
        details={"from_regime": "TRENDING", "to_regime": "CHOPPY"},
    )


def _flow_reversal_event(severity="critical", bar_index=5):
    return ContextChangeEvent(
        event_type="flow_reversal",
        severity=severity,
        bar_index=bar_index,
        details={"entry_signed_aggression": 0.15, "current_signed_aggression": -0.12},
    )


def _momentum_stall_event(bar_index=5):
    return ContextChangeEvent(
        event_type="momentum_stall",
        severity="warning",
        bar_index=bar_index,
        details={"entry_flow_score": 75.0, "current_flow_score": 50.0},
    )


def _volatility_spike_event(bar_index=5):
    return ContextChangeEvent(
        event_type="volatility_spike",
        severity="warning",
        bar_index=bar_index,
        details={"entry_atr": 0.50, "current_atr": 1.10, "atr_ratio": 2.2},
    )


# ── Regime Flip Tests ────────────────────────────────────────────────

class TestRegimeFlipResponse:
    def test_regime_flip_tightens_stop_loss(self):
        """Regime flip should tighten stop by configured percentage."""
        policy = ContextAwareExitPolicy({"regime_flip_tighten_stop_pct": 0.30})
        pos = StubPosition(side="long", entry_price=100.0, stop_loss=97.0, highest_price=102.0)
        session = StubSession(bars=[StubBar(close=101.0)])

        # Distance from highest_price to stop: 102 - 97 = 5.0
        # Tighten by 30%: 97 + (5.0 * 0.30) = 98.5
        result = policy.evaluate(
            context_events=[_regime_flip_event(severity="warning")],
            session=session,
            pos=pos,
            current_bar_index=10,
        )
        assert result is None  # no immediate exit for warning
        assert pos.stop_loss == pytest.approx(98.5, abs=0.01)

    def test_regime_flip_critical_exits_when_losing(self):
        """Critical regime flip with losing position exits immediately."""
        policy = ContextAwareExitPolicy({
            "regime_flip_exit_when_losing": True,
            "regime_flip_exit_loss_threshold_pct": 0.3,
        })
        # Position is losing: entry 100, current 99.5 → -0.5%
        pos = StubPosition(side="long", entry_price=100.0, stop_loss=97.0)
        session = StubSession(bars=[StubBar(close=99.5)])

        result = policy.evaluate(
            context_events=[_regime_flip_event(severity="critical")],
            session=session,
            pos=pos,
            current_bar_index=10,
        )
        assert result is not None
        assert result.should_exit is True
        assert result.reason == "context_regime_flip_exit"
        assert result.exit_price == 99.5

    def test_regime_flip_critical_no_exit_when_profitable(self):
        """Critical regime flip does not exit when position is profitable."""
        policy = ContextAwareExitPolicy({
            "regime_flip_exit_when_losing": True,
            "regime_flip_exit_loss_threshold_pct": 0.3,
        })
        # Position is profitable: entry 100, current 101
        pos = StubPosition(side="long", entry_price=100.0, stop_loss=97.0, highest_price=101.0)
        session = StubSession(bars=[StubBar(close=101.0)])

        result = policy.evaluate(
            context_events=[_regime_flip_event(severity="critical")],
            session=session,
            pos=pos,
            current_bar_index=10,
        )
        assert result is None  # should not exit
        # But stop should still be tightened
        assert pos.stop_loss > 97.0

    def test_regime_flip_shortens_time_exit(self):
        """Regime flip should shorten time exit by configured %."""
        policy = ContextAwareExitPolicy({"regime_flip_shorten_time_pct": 0.50})
        pos = StubPosition(side="long", entry_price=100.0, stop_loss=97.0, highest_price=100.5)
        session = StubSession(bars=[StubBar(close=100.5)])
        session.time_exit_bars = 40
        session.choppy_time_exit_bars = 12

        policy.evaluate(
            context_events=[_regime_flip_event(severity="warning")],
            session=session,
            pos=pos,
            current_bar_index=10,
        )
        assert session.time_exit_bars == 20  # 40 * 0.50 = 20
        assert session.choppy_time_exit_bars == 6  # 12 * 0.50 = 6

    def test_regime_flip_short_side(self):
        """Regime flip tightens stop for short positions correctly."""
        policy = ContextAwareExitPolicy({"regime_flip_tighten_stop_pct": 0.30})
        pos = StubPosition(
            side="short", entry_price=100.0, stop_loss=103.0,
            lowest_price=98.0, highest_price=100.0,
        )
        session = StubSession(bars=[StubBar(close=99.0)])

        # Distance from stop to lowest: 103 - 98 = 5.0
        # Tighten by 30%: 103 - (5.0 * 0.30) = 101.5
        policy.evaluate(
            context_events=[_regime_flip_event(severity="warning")],
            session=session,
            pos=pos,
            current_bar_index=10,
        )
        assert pos.stop_loss == pytest.approx(101.5, abs=0.01)


# ── Flow Reversal Tests ──────────────────────────────────────────────

class TestFlowReversalResponse:
    def test_flow_reversal_moves_to_breakeven_when_profitable(self):
        """Profitable position + flow reversal → stop moves to breakeven."""
        policy = ContextAwareExitPolicy({"flow_reversal_move_to_breakeven": True})
        pos = StubPosition(side="long", entry_price=100.0, stop_loss=97.0)
        session = StubSession(bars=[StubBar(close=102.0)])  # profitable
        session.break_even_buffer_pct = 0.03

        result = policy.evaluate(
            context_events=[_flow_reversal_event()],
            session=session,
            pos=pos,
            current_bar_index=10,
        )
        assert result is None  # no exit, just stop adjustment
        expected_be = 100.0 * (1 + 0.03 / 100)
        assert pos.stop_loss >= expected_be
        assert pos.break_even_stop_active is True

    def test_flow_reversal_exits_when_losing(self):
        """Losing position + flow reversal → immediate exit."""
        policy = ContextAwareExitPolicy({"flow_reversal_exit_when_losing": True})
        pos = StubPosition(side="long", entry_price=100.0, stop_loss=97.0)
        session = StubSession(bars=[StubBar(close=99.5)])  # losing

        result = policy.evaluate(
            context_events=[_flow_reversal_event()],
            session=session,
            pos=pos,
            current_bar_index=10,
        )
        assert result is not None
        assert result.should_exit is True
        assert result.reason == "context_flow_reversal_exit"
        assert result.exit_price == 99.5

    def test_flow_reversal_no_exit_when_disabled(self):
        """Flow reversal exit disabled → no exit even when losing."""
        policy = ContextAwareExitPolicy({
            "flow_reversal_exit_when_losing": False,
            "flow_reversal_move_to_breakeven": False,
        })
        pos = StubPosition(side="long", entry_price=100.0, stop_loss=97.0)
        session = StubSession(bars=[StubBar(close=99.5)])

        result = policy.evaluate(
            context_events=[_flow_reversal_event()],
            session=session,
            pos=pos,
            current_bar_index=10,
        )
        assert result is None

    def test_flow_reversal_short_breakeven(self):
        """Short position: flow reversal moves stop down to breakeven."""
        policy = ContextAwareExitPolicy({"flow_reversal_move_to_breakeven": True})
        pos = StubPosition(
            side="short", entry_price=100.0, stop_loss=103.0,
        )
        session = StubSession(bars=[StubBar(close=98.0)])  # profitable short
        session.break_even_buffer_pct = 0.03

        result = policy.evaluate(
            context_events=[_flow_reversal_event()],
            session=session,
            pos=pos,
            current_bar_index=10,
        )
        assert result is None
        expected_be = 100.0 * (1 - 0.03 / 100)
        assert pos.stop_loss <= expected_be
        assert pos.break_even_stop_active is True


# ── Momentum Stall Tests ─────────────────────────────────────────────

class TestMomentumStallResponse:
    def test_momentum_stall_shortens_time_exit(self):
        """Momentum stall should reduce time exit by configured multiplier."""
        policy = ContextAwareExitPolicy({"momentum_stall_time_multiplier": 0.7})
        pos = StubPosition()
        session = StubSession()
        session.time_exit_bars = 40
        session.choppy_time_exit_bars = 12

        result = policy.evaluate(
            context_events=[_momentum_stall_event()],
            session=session,
            pos=pos,
            current_bar_index=10,
        )
        assert result is None
        # Reduce by 30%: 40 * 0.7 = 28
        assert session.time_exit_bars == 28
        assert session.choppy_time_exit_bars == 8  # 12 * 0.7 = 8.4 → 8


# ── Volatility Spike Tests ───────────────────────────────────────────

class TestVolatilitySpikeResponse:
    def test_volatility_spike_tightens_stop(self):
        """Volatility spike should tighten stop by configured %."""
        policy = ContextAwareExitPolicy({"volatility_spike_tighten_pct": 0.20})
        pos = StubPosition(side="long", entry_price=100.0, stop_loss=96.0, highest_price=102.0)
        session = StubSession(bars=[StubBar(close=101.0)])

        # Distance: 102 - 96 = 6.0; tighten 20%: 96 + (6.0 * 0.20) = 97.2
        policy.evaluate(
            context_events=[_volatility_spike_event()],
            session=session,
            pos=pos,
            current_bar_index=10,
        )
        assert pos.stop_loss == pytest.approx(97.2, abs=0.01)


# ── Multiple Events Tests ────────────────────────────────────────────

class TestMultipleEvents:
    def test_multiple_events_processed_in_order(self):
        """Multiple events in one bar are processed sequentially."""
        policy = ContextAwareExitPolicy({
            "regime_flip_tighten_stop_pct": 0.20,
            "volatility_spike_tighten_pct": 0.20,
        })
        pos = StubPosition(side="long", entry_price=100.0, stop_loss=96.0, highest_price=102.0)
        session = StubSession(bars=[StubBar(close=101.0)])
        session.time_exit_bars = 40

        events = [
            _regime_flip_event(severity="warning"),
            _volatility_spike_event(),
        ]
        result = policy.evaluate(
            context_events=events,
            session=session,
            pos=pos,
            current_bar_index=10,
        )
        assert result is None
        # Both tightenings applied:
        # regime_flip: 96 + (6.0 * 0.20) = 97.2
        # volatility: from 97.2, distance=102-97.2=4.8, + (4.8 * 0.20) = 97.2 + 0.96 = 98.16
        assert pos.stop_loss > 97.0
        assert policy._applied_regime_flip is True
        assert policy._applied_volatility_spike is True

    def test_event_fires_only_once(self):
        """Same event type on next bar doesn't re-apply."""
        policy = ContextAwareExitPolicy({"regime_flip_tighten_stop_pct": 0.30})
        pos = StubPosition(side="long", entry_price=100.0, stop_loss=97.0, highest_price=102.0)
        session = StubSession(bars=[StubBar(close=101.0)])

        # First call
        policy.evaluate(
            context_events=[_regime_flip_event(severity="warning")],
            session=session,
            pos=pos,
            current_bar_index=10,
        )
        stop_after_first = pos.stop_loss

        # Second call with same event
        policy.evaluate(
            context_events=[_regime_flip_event(severity="warning")],
            session=session,
            pos=pos,
            current_bar_index=11,
        )
        assert pos.stop_loss == stop_after_first  # unchanged


# ── Default Config (no behavior change) Tests ────────────────────────

class TestDefaultBehavior:
    def test_no_events_no_change(self):
        """Empty event list produces no changes."""
        policy = ContextAwareExitPolicy()
        pos = StubPosition(side="long", entry_price=100.0, stop_loss=97.0)
        session = StubSession()
        original_stop = pos.stop_loss

        result = policy.evaluate([], session, pos, 10)
        assert result is None
        assert pos.stop_loss == original_stop

    def test_reset_allows_reuse(self):
        """After reset, events can fire again."""
        policy = ContextAwareExitPolicy({"regime_flip_tighten_stop_pct": 0.30})
        pos = StubPosition(side="long", entry_price=100.0, stop_loss=97.0, highest_price=102.0)
        session = StubSession(bars=[StubBar(close=101.0)])

        policy.evaluate(
            context_events=[_regime_flip_event(severity="warning")],
            session=session,
            pos=pos,
            current_bar_index=10,
        )
        assert policy._applied_regime_flip is True

        policy.reset()
        assert policy._applied_regime_flip is False

    def test_summary_reports_applied_responses(self):
        """get_applied_summary returns correct flags."""
        policy = ContextAwareExitPolicy()
        pos = StubPosition()
        session = StubSession()

        policy.evaluate([_regime_flip_event(severity="warning")], session, pos, 10)
        summary = policy.get_applied_summary()

        assert summary["regime_flip_response"] is True
        assert summary["flow_reversal_response"] is False
        assert summary["momentum_stall_response"] is False
        assert summary["volatility_spike_response"] is False


# ── Integration with PositionContextMonitor ──────────────────────────

class TestIntegrationWithMonitor:
    def test_monitor_events_feed_into_policy(self):
        """PositionContextMonitor events work with ContextAwareExitPolicy."""
        snapshot = EntrySnapshot(
            macro_regime="TRENDING",
            micro_regime="TRENDING_UP",
            strategy_name="momentum_flow",
            flow_score=80.0,
            signed_aggression=0.15,
            book_pressure=0.10,
            directional_consistency=0.6,
            delta_acceleration=50.0,
            entry_bar_index=0,
            entry_price=100.0,
            side="long",
            atr_at_entry=0.50,
            volatility_at_entry=0.02,
        )
        monitor = PositionContextMonitor(snapshot, {"context_grace_bars": 1})
        policy = ContextAwareExitPolicy({"regime_flip_tighten_stop_pct": 0.25})

        pos = StubPosition(side="long", entry_price=100.0, stop_loss=97.0, highest_price=101.0)
        session = StubSession(bars=[StubBar(close=100.5)])

        # Bar 1: grace period
        events1 = monitor.update("TRENDING", "TRENDING_UP", {"flow_score": 78}, 1, 0.50)
        assert len(events1) == 0

        # Bar 2: regime flips to CHOPPY
        events2 = monitor.update("CHOPPY", "CHOPPY", {"flow_score": 70}, 2, 0.55)
        assert len(events2) >= 1
        assert events2[0].event_type == "regime_flip"

        # Feed to policy
        result = policy.evaluate(monitor.events, session, pos, 2)
        # Momentum strategy in TRENDING→CHOPPY = critical, losing check
        assert pos.stop_loss > 97.0  # stop was tightened
