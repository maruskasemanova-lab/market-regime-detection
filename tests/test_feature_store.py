"""Tests for FeatureStore: streaming indicators with rolling normalization."""
import pytest
from src.feature_store import FeatureStore, FeatureVector, RollingStats


def _make_bar(close, open_=None, high=None, low=None, volume=1000, vwap=None, **l2):
    o = open_ or close * 0.999
    h = high or close * 1.001
    lo = low or close * 0.998
    bar = {
        'open': o, 'high': h, 'low': lo, 'close': close,
        'volume': volume, 'vwap': vwap,
    }
    bar.update(l2)
    return bar


class TestRollingStats:
    def test_empty(self):
        rs = RollingStats(10)
        assert rs.mean == 0.0
        assert rs.std == 0.0
        assert rs.z_score(5.0) == 0.0
        assert rs.percentile_rank(5.0) == 0.5

    def test_single_value(self):
        rs = RollingStats(10)
        rs.update(5.0)
        assert rs.mean == 5.0
        assert rs.std == 0.0

    def test_z_score(self):
        rs = RollingStats(100)
        for v in range(100):
            rs.update(float(v))
        z = rs.z_score(50.0)
        assert abs(z - 0.017) < 0.1  # ~0 (near mean)

    def test_percentile_rank(self):
        rs = RollingStats(100)
        for v in range(100):
            rs.update(float(v))
        assert rs.percentile_rank(50.0) == pytest.approx(0.51, abs=0.02)
        assert rs.percentile_rank(0.0) == pytest.approx(0.01, abs=0.02)
        assert rs.percentile_rank(99.0) == pytest.approx(1.0, abs=0.02)


class TestFeatureStore:
    def test_basic_update(self):
        store = FeatureStore()
        fv = store.update(_make_bar(100.0))
        assert isinstance(fv, FeatureVector)
        assert fv.bar_index == 1
        assert fv.close == 100.0

    def test_indicators_converge(self):
        store = FeatureStore()
        for i in range(50):
            price = 100.0 + i * 0.1
            fv = store.update(_make_bar(price))
        # EMA should be close to recent prices
        assert abs(fv.ema_10 - price) < 2.0
        assert abs(fv.ema_20 - price) < 3.0
        # RSI should be high (consistently rising prices)
        assert fv.rsi_14 > 60
        # ATR should be positive
        assert fv.atr_14 > 0

    def test_z_scores_populate(self):
        store = FeatureStore(zscore_window=20)
        for i in range(30):
            fv = store.update(_make_bar(100.0 + i * 0.5, volume=1000 + i * 10))
        # Z-scores should be non-zero after window fills
        assert fv.rsi_z != 0.0
        assert fv.volume_z != 0.0

    def test_roc_computation(self):
        store = FeatureStore()
        for i in range(25):
            store.update(_make_bar(100.0))
        # After flat prices, ROC should be ~0
        fv = store.update(_make_bar(100.0))
        assert abs(fv.roc_5) < 0.01
        assert abs(fv.roc_10) < 0.01

    def test_l2_features_without_data(self):
        store = FeatureStore()
        for i in range(5):
            fv = store.update(_make_bar(100.0))
        assert fv.l2_has_coverage is False
        assert fv.l2_delta == 0.0

    def test_l2_features_with_data(self):
        store = FeatureStore()
        for i in range(25):
            fv = store.update(_make_bar(
                100.0 + i * 0.1, volume=5000,
                l2_delta=100.0, l2_volume=3000.0,
                l2_imbalance=0.1, l2_book_pressure=0.05,
            ))
        assert fv.l2_has_coverage is True
        assert fv.l2_delta != 0.0
        assert fv.l2_signed_aggression != 0.0

    def test_multi_timeframe(self):
        store = FeatureStore()
        for i in range(20):
            fv = store.update(_make_bar(100.0 + i * 0.2, volume=1000))
        # After 20 bars, we should have some 5-min aggregated bars (20/5 = 4)
        # tf5_trend_slope should be non-zero with trending prices
        # (need 6+ 5min bars = 30+ 1min bars for slope)
        for i in range(20, 40):
            fv = store.update(_make_bar(100.0 + i * 0.2, volume=1000))
        assert fv.tf5_trend_slope != 0.0

    def test_to_legacy_indicators(self):
        store = FeatureStore()
        for i in range(20):
            fv = store.update(_make_bar(100.0 + i * 0.1))
        order_flow = {'has_l2_coverage': False}
        legacy = store.to_legacy_indicators(fv, order_flow)
        assert 'ema' in legacy
        assert 'ema_fast' in legacy
        assert 'rsi' in legacy
        assert 'atr' in legacy
        assert 'order_flow' in legacy
        assert '_feature_vector' in legacy

    def test_reset(self):
        store = FeatureStore()
        for i in range(10):
            store.update(_make_bar(100.0))
        store.reset()
        fv = store.update(_make_bar(100.0))
        assert fv.bar_index == 1

    def test_bollinger_width(self):
        store = FeatureStore()
        # Feed identical prices → width should be 0
        for i in range(25):
            fv = store.update(_make_bar(100.0, open_=100.0, high=100.0, low=100.0))
        assert fv.bollinger_width == pytest.approx(0.0, abs=0.001)

    def test_obv_direction(self):
        store = FeatureStore()
        # Rising prices → OBV should increase
        for i in range(20):
            fv = store.update(_make_bar(100.0 + i, volume=1000))
        assert fv.obv > 0
        assert fv.obv_slope > 0
