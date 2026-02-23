"""Tests for ConfidenceCalibrator: isotonic regression calibration."""
import pytest
from src.confidence_calibrator import (
    ConfidenceCalibrator, IsotonicCalibrator, MIN_CALIBRATION_TRADES,
)


class TestIsotonicCalibrator:
    def test_insufficient_data_returns_identity(self):
        cal = IsotonicCalibrator()
        # Less than MIN_CALIBRATION_TRADES -> conservative warmup mapping.
        for i in range(5):
            cal.update(70.0, True)
        result = cal.calibrate(70.0)
        assert result == pytest.approx(0.56, abs=0.01)

    def test_calibration_with_enough_data(self):
        cal = IsotonicCalibrator(n_bins=5, lookback=50)
        # High confidence → always wins, low confidence → always loses
        for _ in range(15):
            cal.update(80.0, True)   # High conf → win
            cal.update(30.0, False)  # Low conf → loss
        # Should be calibrated now
        assert cal.n_trades >= MIN_CALIBRATION_TRADES
        high_cal = cal.calibrate(80.0)
        low_cal = cal.calibrate(30.0)
        # High confidence should calibrate higher than low
        assert high_cal > low_cal

    def test_pava_monotonicity(self):
        result = IsotonicCalibrator._pava([0.8, 0.3, 0.5, 0.9])
        # Should be non-decreasing
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1] + 1e-10


class TestConfidenceCalibrator:
    def test_fallback_to_identity(self):
        cal = ConfidenceCalibrator()
        # No data -> conservative warmup mapping
        result = cal.calibrate('momentum', 70.0, 'TRENDING')
        assert result == pytest.approx(0.56, abs=0.01)

    def test_per_strategy_calibration(self):
        cal = ConfidenceCalibrator(lookback_trades=30)
        # Momentum with high confidence always wins
        for _ in range(25):
            cal.update('momentum', 80.0, 'TRENDING', True)
            cal.update('momentum', 30.0, 'TRENDING', False)
        # Should have separate calibration
        high = cal.calibrate('momentum', 80.0, 'TRENDING')
        low = cal.calibrate('momentum', 30.0, 'TRENDING')
        assert high > low

    def test_cross_regime_fallback(self):
        cal = ConfidenceCalibrator(lookback_trades=30)
        # Only train on TRENDING
        for _ in range(25):
            cal.update('momentum', 80.0, 'TRENDING', True)
        # Query for CHOPPY → should fall back to (strategy, ALL)
        result = cal.calibrate('momentum', 80.0, 'CHOPPY')
        assert result > 0.5

    def test_global_fallback(self):
        cal = ConfidenceCalibrator(lookback_trades=10)
        # Train different strategies
        for _ in range(15):
            cal.update('strategy_a', 70.0, 'TRENDING', True)
        # Query unknown strategy → global fallback
        result = cal.calibrate('unknown_strategy', 70.0, 'TRENDING')
        assert 0.0 <= result <= 1.0

    def test_reset(self):
        cal = ConfidenceCalibrator()
        cal.update('x', 50.0, 'MIXED', True)
        cal.reset()
        stats = cal.get_stats()
        assert stats['global']['n_trades'] == 0
