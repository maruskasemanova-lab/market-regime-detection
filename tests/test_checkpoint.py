"""
Tests for checkpoint persistence (save/load roundtrip).

Verifies that:
  - ConfidenceCalibrator state survives save/load cycle
  - EdgeMonitor state survives save/load cycle
  - AdaptiveWeightCombiner state survives save/load cycle
  - Empty state saves/loads without error
  - Corrupt/invalid checkpoints raise appropriate errors
  - Version mismatch is detected
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

from src.confidence_calibrator import ConfidenceCalibrator
from src.edge_monitor import EdgeMonitor
from src.ensemble_combiner import AdaptiveWeightCombiner
from src.trading_orchestrator import TradingOrchestrator
from src.checkpoint import (
    CHECKPOINT_VERSION,
    save_checkpoint,
    load_checkpoint,
    apply_checkpoint,
    restore_calibrator,
    restore_edge_monitor,
    restore_combiner,
)


class TestCheckpointRoundtrip(unittest.TestCase):
    """Full roundtrip: populate → save → load → verify identical."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.cp_path = os.path.join(self.tmp_dir, "test_checkpoint.json")

    def _populate_calibrator(self, cal: ConfidenceCalibrator):
        """Feed realistic trade outcomes into calibrator."""
        outcomes = [
            ("momentum_flow", "TRENDING", 72.0, True),
            ("momentum_flow", "TRENDING", 65.0, False),
            ("momentum_flow", "TRENDING", 80.0, True),
            ("mean_reversion", "CHOPPY", 55.0, True),
            ("mean_reversion", "CHOPPY", 60.0, False),
        ]
        for strategy, regime, conf, won in outcomes:
            cal.update(strategy, conf, regime, won)

    def _populate_edge_monitor(self, mon: EdgeMonitor):
        """Feed realistic trades into edge monitor."""
        trades = [
            ("momentum_flow", "TRENDING", 1.5, True),
            ("momentum_flow", "TRENDING", -0.8, False),
            ("momentum_flow", "TRENDING", 2.0, True),
            ("mean_reversion", "CHOPPY", 0.5, True),
            ("mean_reversion", "CHOPPY", -1.0, False),
        ]
        for strategy, regime, pnl, won in trades:
            mon.update_trade(strategy, regime, pnl, won, bar_index=0)

    def _populate_combiner(self, comb: AdaptiveWeightCombiner):
        """Feed realistic outcomes into combiner."""
        outcomes = [
            ("strategy", "momentum_flow", "TRENDING", True, 1.5),
            ("strategy", "momentum_flow", "TRENDING", False, -0.8),
            ("pattern", "hammer", "TRENDING", True, 1.0),
            ("strategy", "mean_reversion", "CHOPPY", True, 0.5),
        ]
        for stype, sname, regime, won, pnl in outcomes:
            comb.update_outcome(stype, sname, regime, won, pnl)

    def test_full_roundtrip(self):
        """Populate all three components, save, load into fresh instances, verify."""
        # 1. Create and populate
        cal = ConfidenceCalibrator(lookback_trades=50)
        self._populate_calibrator(cal)

        mon = EdgeMonitor()
        self._populate_edge_monitor(mon)

        comb = AdaptiveWeightCombiner()
        self._populate_combiner(comb)

        # 2. Save
        save_checkpoint(cal, mon, comb, self.cp_path, {"ticker": "MU"})

        # 3. Load into fresh instances
        cal2 = ConfidenceCalibrator(lookback_trades=50)
        mon2 = EdgeMonitor()
        comb2 = AdaptiveWeightCombiner()

        data = load_checkpoint(self.cp_path)
        restore_calibrator(cal2, data["calibrator"])
        restore_edge_monitor(mon2, data["edge_monitor"])
        restore_combiner(comb2, data["combiner"])

        # 4. Verify calibrator
        self.assertEqual(len(cal2._calibrators), len(cal._calibrators))
        for key in cal._calibrators:
            self.assertIn(key, cal2._calibrators)
            self.assertEqual(
                list(cal._calibrators[key]._outcomes),
                list(cal2._calibrators[key]._outcomes),
            )
        self.assertEqual(
            list(cal._global_calibrator._outcomes),
            list(cal2._global_calibrator._outcomes),
        )

        # 5. Verify edge monitor
        self.assertEqual(len(mon2._trackers), len(mon._trackers))
        for key in mon._trackers:
            self.assertIn(key, mon2._trackers)
            self.assertEqual(
                list(mon._trackers[key]._trades),
                list(mon2._trackers[key]._trades),
            )
            self.assertEqual(
                mon._trackers[key]._total_trades,
                mon2._trackers[key]._total_trades,
            )
            self.assertEqual(
                mon._trackers[key]._total_wins,
                mon2._trackers[key]._total_wins,
            )
        self.assertEqual(
            list(mon._global_tracker._trades),
            list(mon2._global_tracker._trades),
        )

        # 6. Verify combiner
        self.assertEqual(len(comb2._performance), len(comb._performance))
        for key in comb._performance:
            self.assertIn(key, comb2._performance)
            self.assertEqual(comb._performance[key].trades, comb2._performance[key].trades)
            self.assertEqual(comb._performance[key].wins, comb2._performance[key].wins)
            self.assertAlmostEqual(
                comb._performance[key].total_pnl_r,
                comb2._performance[key].total_pnl_r,
                places=5,
            )
            self.assertEqual(
                list(comb._performance[key].recent_pnls),
                list(comb2._performance[key].recent_pnls),
            )

    def test_empty_state_roundtrip(self):
        """Save and load completely empty state without errors."""
        cal = ConfidenceCalibrator()
        mon = EdgeMonitor()
        comb = AdaptiveWeightCombiner()

        save_checkpoint(cal, mon, comb, self.cp_path)
        data = load_checkpoint(self.cp_path)

        cal2 = ConfidenceCalibrator()
        mon2 = EdgeMonitor()
        comb2 = AdaptiveWeightCombiner()

        restore_calibrator(cal2, data["calibrator"])
        restore_edge_monitor(mon2, data["edge_monitor"])
        restore_combiner(comb2, data["combiner"])

        self.assertEqual(len(cal2._calibrators), 0)
        self.assertEqual(len(mon2._trackers), 0)
        self.assertEqual(len(comb2._performance), 0)

    def test_version_mismatch_raises(self):
        """Invalid checkpoint version should raise ValueError."""
        with open(self.cp_path, "w") as f:
            json.dump({"version": 999, "calibrator": {}, "edge_monitor": {}, "combiner": {}}, f)

        with self.assertRaises(ValueError):
            load_checkpoint(self.cp_path)

    def test_file_not_found_raises(self):
        """Missing checkpoint path should raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_checkpoint("/nonexistent/path.json")

    def test_metadata_preserved(self):
        """Checkpoint metadata (source, created_at) is preserved."""
        cal = ConfidenceCalibrator()
        mon = EdgeMonitor()
        comb = AdaptiveWeightCombiner()

        metadata = {"run_id": "test-run", "ticker": "MU", "date_from": "2026-01-20"}
        save_checkpoint(cal, mon, comb, self.cp_path, metadata)

        data = load_checkpoint(self.cp_path)
        self.assertEqual(data["source"]["run_id"], "test-run")
        self.assertEqual(data["source"]["ticker"], "MU")
        self.assertIn("created_at", data)
        self.assertEqual(data["version"], CHECKPOINT_VERSION)

    def test_calibrator_lookback_preserved(self):
        """IsotonicCalibrator lookback parameter roundtrips correctly."""
        cal = ConfidenceCalibrator(lookback_trades=30)
        cal.update("strat_a", 70.0, "TRENDING", True)

        mon = EdgeMonitor()
        comb = AdaptiveWeightCombiner()

        save_checkpoint(cal, mon, comb, self.cp_path)
        data = load_checkpoint(self.cp_path)

        cal2 = ConfidenceCalibrator()
        restore_calibrator(cal2, data["calibrator"])

        key = ("strat_a", "TRENDING")
        self.assertEqual(cal2._calibrators[key].lookback, 30)

    def test_edge_monitor_total_counters(self):
        """Edge monitor total_trades and total_wins survive roundtrip."""
        mon = EdgeMonitor()
        for i in range(5):
            mon.update_trade("strat_a", "TRENDING", 1.0, True, bar_index=i)
        for i in range(3):
            mon.update_trade("strat_a", "TRENDING", -0.5, False, bar_index=i + 5)

        cal = ConfidenceCalibrator()
        comb = AdaptiveWeightCombiner()

        save_checkpoint(cal, mon, comb, self.cp_path)
        data = load_checkpoint(self.cp_path)

        mon2 = EdgeMonitor()
        restore_edge_monitor(mon2, data["edge_monitor"])

        tracker = mon2._trackers[("strat_a", "TRENDING")]
        self.assertEqual(tracker._total_trades, 8)
        self.assertEqual(tracker._total_wins, 5)


class TestOrchestratorCheckpoint(unittest.TestCase):
    """Integration test: save/load via TradingOrchestrator methods."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.cp_path = os.path.join(self.tmp_dir, "orch_checkpoint.json")

    def test_orchestrator_save_load(self):
        """TradingOrchestrator.save_checkpoint/load_checkpoint roundtrip."""
        orch = TradingOrchestrator()

        # Populate via record_trade_outcome
        orch.record_trade_outcome(
            strategy="momentum_flow", regime="TRENDING",
            raw_confidence=75.0, was_profitable=True, pnl_r=1.5,
        )
        orch.record_trade_outcome(
            strategy="momentum_flow", regime="TRENDING",
            raw_confidence=60.0, was_profitable=False, pnl_r=-0.8,
        )

        path = orch.save_checkpoint(path=self.cp_path, metadata={"ticker": "MU"})
        self.assertEqual(path, self.cp_path)
        self.assertTrue(os.path.exists(path))

        # Load into fresh orchestrator
        orch2 = TradingOrchestrator()
        info = orch2.load_checkpoint(path)

        self.assertEqual(info["source"]["ticker"], "MU")
        self.assertEqual(info["source"]["total_trades"], 2)

        # Verify calibrator was restored
        key = ("momentum_flow", "TRENDING")
        self.assertIn(key, orch2.calibrator._calibrators)
        self.assertEqual(orch2.calibrator._calibrators[key].n_trades, 2)

        # Verify edge monitor was restored
        self.assertIn(key, orch2.edge_monitor._trackers)
        self.assertEqual(orch2.edge_monitor._trackers[key]._total_trades, 2)

    def test_warmup_feature_store(self):
        """warmup_feature_store processes bars and increments bar_count."""
        orch = TradingOrchestrator()
        orch.new_session()

        bars = [
            {"open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1000}
            for _ in range(50)
        ]
        count = orch.warmup_feature_store(bars)

        self.assertEqual(count, 50)
        self.assertEqual(orch._bar_count, 50)
        self.assertIsNotNone(orch.current_feature_vector)


if __name__ == "__main__":
    unittest.main()
