"""Tests for first-bar stop-loss protection.

After the look-ahead bias fix, positions could get stopped out immediately
on the bar they opened because intrabar quotes from before the fill second
breached the stop-loss.  The fix: resolve_exit_for_bar(is_entry_bar=True)
skips intrabar SL and uses bar.close for SL comparison instead of bar.low.
"""
import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Minimal stubs ────────────────────────────────────────────────────

@dataclass
class StubBar:
    open: float = 100.0
    high: float = 101.0
    low: float = 99.0
    close: float = 100.5
    volume: float = 1000.0
    intrabar_quotes_1s: Optional[List[Dict[str, Any]]] = None


@dataclass
class StubPosition:
    side: str = "long"
    entry_price: float = 100.0
    stop_loss: float = 99.50
    take_profit: float = 103.0
    trailing_stop_active: bool = False
    trailing_stop_price: float = 0.0
    break_even_stop_active: bool = False
    entry_bar_index: int = 10


# ── Tests ────────────────────────────────────────────────────────────

from src.exit_policy_engine import ExitPolicyEngine


class TestEntryBarSkipsIntrabarSL:
    """Entry-bar should skip intrabar-quote SL checks."""

    def test_entry_bar_skips_intrabar_sl(self):
        """Intrabar quotes with bid below SL should NOT trigger exit on entry bar."""
        pos = StubPosition(side="long", stop_loss=99.70)
        bar = StubBar(
            open=100.0, high=100.5, low=99.60, close=100.2,
            intrabar_quotes_1s=[
                {"s": 0, "bid": 99.65, "ask": 99.75},  # bid < SL
                {"s": 5, "bid": 99.68, "ask": 99.78},  # bid < SL
                {"s": 10, "bid": 100.0, "ask": 100.10},
            ],
        )
        result = ExitPolicyEngine.resolve_exit_for_bar(pos, bar, is_entry_bar=True)
        assert result is None, (
            f"Expected no exit on entry bar, but got {result}"
        )

    def test_non_entry_bar_uses_intrabar_sl(self):
        """Same scenario but NOT entry bar → should trigger SL via intrabar."""
        pos = StubPosition(side="long", stop_loss=99.70)
        bar = StubBar(
            open=100.0, high=100.5, low=99.60, close=100.2,
            intrabar_quotes_1s=[
                {"s": 0, "bid": 99.65, "ask": 99.75},  # bid < SL
                {"s": 5, "bid": 100.0, "ask": 100.10},
            ],
        )
        result = ExitPolicyEngine.resolve_exit_for_bar(pos, bar, is_entry_bar=False)
        assert result is not None
        assert result[0] == "stop_loss"


class TestEntryBarUsesCloseForSL:
    """Entry-bar SL should compare against bar.close, not bar.low."""

    def test_entry_bar_no_exit_when_low_below_sl_but_close_above(self):
        """bar.low < SL but bar.close > SL → no exit on entry bar."""
        pos = StubPosition(side="long", stop_loss=99.50)
        bar = StubBar(open=100.0, high=100.5, low=99.30, close=100.0)
        result = ExitPolicyEngine.resolve_exit_for_bar(pos, bar, is_entry_bar=True)
        assert result is None

    def test_entry_bar_exits_when_close_below_sl(self):
        """bar.close < SL → exit even on entry bar."""
        pos = StubPosition(side="long", stop_loss=99.50)
        bar = StubBar(open=100.0, high=100.0, low=99.0, close=99.30)
        result = ExitPolicyEngine.resolve_exit_for_bar(pos, bar, is_entry_bar=True)
        assert result is not None
        assert result[0] == "stop_loss"

    def test_non_entry_bar_uses_low_for_sl(self):
        """bar.low < SL on non-entry bar → exit."""
        pos = StubPosition(side="long", stop_loss=99.50)
        bar = StubBar(open=100.0, high=100.5, low=99.30, close=100.0)
        result = ExitPolicyEngine.resolve_exit_for_bar(pos, bar, is_entry_bar=False)
        assert result is not None
        assert result[0] == "stop_loss"


class TestEntryBarShortSide:
    """Same protection for short positions."""

    def test_entry_bar_no_exit_short_when_high_above_sl_but_close_below(self):
        """Short: bar.high > SL but bar.close < SL → no exit on entry bar."""
        pos = StubPosition(side="short", stop_loss=100.50, take_profit=0)
        bar = StubBar(open=100.0, high=100.80, low=99.50, close=100.20)
        result = ExitPolicyEngine.resolve_exit_for_bar(pos, bar, is_entry_bar=True)
        assert result is None

    def test_entry_bar_exits_short_when_close_above_sl(self):
        """Short: bar.close > SL → exit on entry bar."""
        pos = StubPosition(side="short", stop_loss=100.50, take_profit=0)
        bar = StubBar(open=100.0, high=101.0, low=99.50, close=100.80)
        result = ExitPolicyEngine.resolve_exit_for_bar(pos, bar, is_entry_bar=True)
        assert result is not None
        assert result[0] == "stop_loss"

    def test_entry_bar_skips_intrabar_sl_short(self):
        """Short: intrabar ask > SL should NOT trigger on entry bar."""
        pos = StubPosition(side="short", stop_loss=100.50, take_profit=0)
        bar = StubBar(
            open=100.0, high=100.80, low=99.50, close=100.20,
            intrabar_quotes_1s=[
                {"s": 0, "bid": 100.55, "ask": 100.65},  # ask > SL
                {"s": 5, "bid": 100.10, "ask": 100.20},
            ],
        )
        result = ExitPolicyEngine.resolve_exit_for_bar(pos, bar, is_entry_bar=True)
        assert result is None


class TestEntryBarTPStillWorks:
    """Take profit should still work on entry bar."""

    def test_entry_bar_tp_triggers(self):
        """TP should still trigger on entry bar (not affected by entry-bar protection)."""
        pos = StubPosition(side="long", stop_loss=99.0, take_profit=101.0)
        bar = StubBar(open=100.0, high=101.5, low=99.80, close=101.2)
        result = ExitPolicyEngine.resolve_exit_for_bar(pos, bar, is_entry_bar=True)
        assert result is not None
        assert result[0] == "take_profit"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
