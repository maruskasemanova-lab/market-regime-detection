"""Tests proving no same-minute look-ahead bias in 5s intrabar checkpoints.

Each test targets a specific bias vector that was previously identified:
  P0-A/B: Orchestrator and intraday levels deferred until after trading eval
  P0-C:   Checkpoint bars have L2 fields zeroed
  P0-E:   Checkpoint FV uses pre-bar (previous) L2, not current-minute L2
"""

import copy
import unittest
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

from src.day_trading_manager import DayTradingManager
from src.day_trading_models import BarData, SessionPhase, TradingSession
from src.feature_store import FeatureStore
from src.feature_store_helpers.types import FeatureVector


def _make_bar(
    ts: datetime,
    close: float = 100.0,
    volume: float = 1000.0,
    **overrides: Any,
) -> Dict[str, Any]:
    return {
        "open": close - 0.05,
        "high": close + 0.10,
        "low": close - 0.10,
        "close": close,
        "volume": volume,
        "vwap": close,
        **overrides,
    }


def _make_bar_with_l2(ts: datetime, close: float = 100.0) -> Dict[str, Any]:
    return _make_bar(
        ts,
        close,
        l2_delta=500.0,
        l2_buy_volume=3000.0,
        l2_sell_volume=2500.0,
        l2_volume=5500.0,
        l2_imbalance=0.09,
        l2_bid_depth_total=10000.0,
        l2_ask_depth_total=9500.0,
        l2_book_pressure=1.05,
        l2_book_pressure_change=0.02,
        l2_iceberg_buy_count=3,
        l2_iceberg_sell_count=1,
        l2_iceberg_bias=0.5,
        l2_quality_flags="full",
        l2_quality=1.0,
    )


def _make_bar_with_intrabar_quotes(
    ts: datetime,
    close: float = 101.5,
) -> Dict[str, Any]:
    """Create a bar with 1s intrabar quotes that diverge from minute OHLCV."""
    bar = _make_bar_with_l2(ts, close=close)
    # Quotes that show a different intrabar trajectory than the bar close
    bar["open"] = 100.0
    bar["high"] = 102.0
    bar["low"] = 99.5
    bar["close"] = close  # 101.5 — final minute close
    bar["intrabar_quotes_1s"] = [
        {"s": 1, "bid": 100.05, "ask": 100.15},
        {"s": 5, "bid": 100.10, "ask": 100.20},  # mid @ 5s = 100.15
        {"s": 10, "bid": 100.30, "ask": 100.40},
        {"s": 15, "bid": 100.50, "ask": 100.60},
        {"s": 20, "bid": 100.80, "ask": 100.90},
        {"s": 25, "bid": 101.00, "ask": 101.10},
        {"s": 30, "bid": 101.20, "ask": 101.30},
        {"s": 35, "bid": 101.30, "ask": 101.40},
        {"s": 40, "bid": 101.10, "ask": 101.20},
        {"s": 45, "bid": 101.20, "ask": 101.30},
        {"s": 50, "bid": 101.35, "ask": 101.45},
        {"s": 55, "bid": 101.40, "ask": 101.50},
        {"s": 59, "bid": 101.45, "ask": 101.55},
    ]
    return bar


class TestCheckpointBarNoL2Leak(unittest.TestCase):
    """P0-C: Checkpoint bars must have L2 fields zeroed."""

    def test_checkpoint_bar_l2_fields_are_none(self) -> None:
        """When building checkpoint bars from a bar with L2 data,
        the checkpoint bars must NOT inherit any L2 fields."""
        manager = DayTradingManager(
            regime_detection_minutes=0,
            max_trades_per_day=5,
            trade_cooldown_bars=0,
        )
        manager._load_aos_config = lambda: None  # type: ignore
        manager._select_strategies = lambda session: ["mean_reversion"]  # type: ignore
        manager._required_confirming_sources = lambda session, bar_time: 1  # type: ignore

        start = datetime(2026, 2, 3, 14, 30, tzinfo=timezone.utc)
        # Feed warmup bar
        manager.process_bar(
            run_id="l2-leak-test",
            ticker="TEST",
            timestamp=start,
            bar_data=_make_bar(start, close=100.0),
        )
        session = manager.get_session("l2-leak-test", "TEST", "2026-02-03")
        self.assertIsNotNone(session)

        # Transition to trading
        ts1 = datetime(2026, 2, 3, 14, 31, tzinfo=timezone.utc)
        manager.process_bar(
            run_id="l2-leak-test",
            ticker="TEST",
            timestamp=ts1,
            bar_data=_make_bar(ts1, close=100.2),
        )
        self.assertEqual(session.phase, SessionPhase.TRADING)

        # Capture checkpoint bars via the intrabar_slice eval
        captured_cp_bars: list = []

        # Monkey-patch intrabar slice to capture checkpoint bars
        from src.day_trading_runtime.intrabar_slice import runtime_evaluate_intrabar_slice
        original_fn = runtime_evaluate_intrabar_slice

        def patched_slice(self_mgr, session, cp_bar, cp_ts):
            captured_cp_bars.append(copy.copy(cp_bar))
            return original_fn(self_mgr, session, cp_bar, cp_ts)

        import src.day_trading_runtime_trading_bar as rtb_mod
        original_impl = rtb_mod._runtime_evaluate_intrabar_slice_impl
        rtb_mod._runtime_evaluate_intrabar_slice_impl = patched_slice

        try:
            # Process bar with L2 + intrabar quotes
            ts2 = datetime(2026, 2, 3, 14, 32, tzinfo=timezone.utc)
            bar_data = _make_bar_with_intrabar_quotes(ts2, close=101.5)
            manager.process_bar(
                run_id="l2-leak-test",
                ticker="TEST",
                timestamp=ts2,
                bar_data=bar_data,
            )
        finally:
            rtb_mod._runtime_evaluate_intrabar_slice_impl = original_impl

        # We should have captured some checkpoint bars
        self.assertGreater(len(captured_cp_bars), 0, "No checkpoint bars captured")

        for i, cp_bar in enumerate(captured_cp_bars):
            # All L2 fields must be None on checkpoint bars
            self.assertIsNone(cp_bar.l2_delta, f"cp[{i}]: l2_delta should be None")
            self.assertIsNone(cp_bar.l2_volume, f"cp[{i}]: l2_volume should be None")
            self.assertIsNone(cp_bar.l2_imbalance, f"cp[{i}]: l2_imbalance should be None")
            self.assertIsNone(cp_bar.l2_book_pressure, f"cp[{i}]: l2_book_pressure should be None")
            self.assertIsNone(cp_bar.l2_quality, f"cp[{i}]: l2_quality should be None")


class TestCheckpointFVUsesPreBarL2(unittest.TestCase):
    """P0-E: compute_checkpoint_fv must use pre_bar_fv for L2, not parent_fv."""

    def test_l2_comes_from_pre_bar_fv(self) -> None:
        store = FeatureStore()
        # Feed 5 bars to warm up
        for i in range(5):
            store.update({"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000})

        # Create two FVs with different L2 values
        pre_bar_fv = store.update({
            "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1200,
            "l2_delta": 100.0, "l2_volume": 5000.0, "l2_imbalance": 0.05,
            "l2_book_pressure": 1.1,
        })

        current_fv = store.update({
            "open": 101, "high": 102, "low": 100, "close": 101.5, "volume": 2000,
            "l2_delta": 900.0, "l2_volume": 12000.0, "l2_imbalance": 0.30,
            "l2_book_pressure": 1.8,
        })

        # Compute checkpoint with pre_bar_fv
        cp_bar = {"open": 101, "high": 101.2, "low": 100.8, "close": 101.1, "volume": 500}
        cp_fv = store.compute_checkpoint_fv(cp_bar, current_fv, pre_bar_fv=pre_bar_fv)

        # L2 must come from pre_bar_fv, NOT current_fv
        self.assertAlmostEqual(cp_fv.l2_delta, pre_bar_fv.l2_delta, places=4)
        self.assertAlmostEqual(cp_fv.l2_book_pressure, pre_bar_fv.l2_book_pressure, places=4)

        # But price-sensitive fields should be recomputed from checkpoint bar
        self.assertAlmostEqual(cp_fv.close, 101.1, places=4)
        self.assertNotAlmostEqual(cp_fv.close, current_fv.close, places=4)

    def test_falls_back_to_parent_when_no_pre_bar(self) -> None:
        store = FeatureStore()
        for i in range(5):
            store.update({"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000})
        parent_fv = store.update({
            "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000,
            "l2_delta": 200.0, "l2_book_pressure": 1.3,
        })

        cp_bar = {"open": 100, "high": 100.5, "low": 99.5, "close": 100.2, "volume": 300}
        cp_fv = store.compute_checkpoint_fv(cp_bar, parent_fv, pre_bar_fv=None)

        # Falls back to parent_fv for L2
        self.assertAlmostEqual(cp_fv.l2_delta, parent_fv.l2_delta, places=4)
        self.assertAlmostEqual(cp_fv.l2_book_pressure, parent_fv.l2_book_pressure, places=4)


class TestIntradayLevelsDeferredDuringTrading(unittest.TestCase):
    """P0-B / P1-A: Intraday levels must not be updated before _process_trading_bar."""

    def test_levels_not_updated_before_trading_eval(self) -> None:
        manager = DayTradingManager(
            regime_detection_minutes=0,
            max_trades_per_day=5,
            trade_cooldown_bars=0,
        )
        manager._load_aos_config = lambda: None  # type: ignore
        manager._select_strategies = lambda session: ["mean_reversion"]  # type: ignore
        manager._required_confirming_sources = lambda session, bar_time: 1  # type: ignore

        start = datetime(2026, 2, 3, 14, 30, tzinfo=timezone.utc)

        # First bar to initialize session
        manager.process_bar(
            run_id="levels-defer",
            ticker="TEST",
            timestamp=start,
            bar_data=_make_bar(start, close=100.0),
        )
        session = manager.get_session("levels-defer", "TEST", "2026-02-03")
        self.assertIsNotNone(session)

        # Transition to trading
        ts1 = datetime(2026, 2, 3, 14, 31, tzinfo=timezone.utc)
        manager.process_bar(
            run_id="levels-defer",
            ticker="TEST",
            timestamp=ts1,
            bar_data=_make_bar(ts1, close=100.2),
        )
        self.assertEqual(session.phase, SessionPhase.TRADING)

        # Capture levels state BEFORE next bar
        levels_state_before = copy.deepcopy(
            getattr(session, "intraday_levels_state", {})
        )
        levels_snapshot_before = levels_state_before.get("snapshot", {})
        last_bar_index_before = levels_state_before.get("last_bar_index", -1)

        # Patch _process_trading_bar to capture levels state during evaluation
        original_ptb = manager._process_trading_bar
        levels_during_eval = {}

        def capture_levels_during_eval(session, bar, timestamp, warmup_only=False):
            levels_during_eval["state"] = copy.deepcopy(
                getattr(session, "intraday_levels_state", {})
            )
            levels_during_eval["last_bar_index"] = levels_during_eval["state"].get(
                "last_bar_index", -1
            )
            return original_ptb(session, bar, timestamp, warmup_only=warmup_only)

        manager._process_trading_bar = capture_levels_during_eval

        # Process next trading bar
        ts2 = datetime(2026, 2, 3, 14, 32, tzinfo=timezone.utc)
        manager.process_bar(
            run_id="levels-defer",
            ticker="TEST",
            timestamp=ts2,
            bar_data=_make_bar(ts2, close=100.5),
        )

        # During trading evaluation, levels should NOT have been updated yet
        self.assertEqual(
            levels_during_eval["last_bar_index"],
            last_bar_index_before,
            "Intraday levels were updated BEFORE trading evaluation — look-ahead bias!",
        )

        # But AFTER process_bar returns, levels should be updated
        levels_after = getattr(session, "intraday_levels_state", {})
        self.assertGreater(
            levels_after.get("last_bar_index", -1),
            last_bar_index_before,
            "Intraday levels should be updated AFTER trading evaluation",
        )


class TestOrchestratorDeferredDuringTrading(unittest.TestCase):
    """P0-A: Orchestrator update_bar must not run before _process_trading_bar."""

    def test_orchestrator_not_updated_before_trading_eval(self) -> None:
        manager = DayTradingManager(
            regime_detection_minutes=0,
            max_trades_per_day=5,
            trade_cooldown_bars=0,
        )
        manager._load_aos_config = lambda: None  # type: ignore
        manager._select_strategies = lambda session: ["mean_reversion"]  # type: ignore
        manager._required_confirming_sources = lambda session, bar_time: 1  # type: ignore

        start = datetime(2026, 2, 3, 14, 30, tzinfo=timezone.utc)
        manager.process_bar(
            run_id="orch-defer",
            ticker="TEST",
            timestamp=start,
            bar_data=_make_bar(start, close=100.0),
        )
        session = manager.get_session("orch-defer", "TEST", "2026-02-03")
        self.assertIsNotNone(session)

        ts1 = datetime(2026, 2, 3, 14, 31, tzinfo=timezone.utc)
        manager.process_bar(
            run_id="orch-defer",
            ticker="TEST",
            timestamp=ts1,
            bar_data=_make_bar(ts1, close=100.2),
        )
        self.assertEqual(session.phase, SessionPhase.TRADING)

        # Record bar count and FV before next bar
        orch = session.orchestrator
        bar_count_before = orch._bar_count
        fv_before = orch.current_feature_vector

        # Capture during trading eval
        original_ptb = manager._process_trading_bar
        captured = {}

        def capture_orch_state(session, bar, timestamp, warmup_only=False):
            captured["bar_count"] = session.orchestrator._bar_count
            captured["fv"] = session.orchestrator.current_feature_vector
            captured["pre_bar_fv"] = getattr(session, "_pre_bar_fv", "MISSING")
            return original_ptb(session, bar, timestamp, warmup_only=warmup_only)

        manager._process_trading_bar = capture_orch_state

        ts2 = datetime(2026, 2, 3, 14, 32, tzinfo=timezone.utc)
        manager.process_bar(
            run_id="orch-defer",
            ticker="TEST",
            timestamp=ts2,
            bar_data=_make_bar(ts2, close=100.5),
        )

        # During trading eval: orchestrator should NOT have been updated
        self.assertEqual(
            captured["bar_count"],
            bar_count_before,
            "Orchestrator was updated BEFORE trading evaluation — look-ahead bias!",
        )

        # pre_bar_fv should be set
        self.assertNotEqual(captured["pre_bar_fv"], "MISSING")

        # After process_bar: orchestrator should be updated
        self.assertEqual(
            orch._bar_count,
            bar_count_before + 1,
            "Orchestrator should be updated AFTER trading evaluation",
        )


if __name__ == "__main__":
    unittest.main()
