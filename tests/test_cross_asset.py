"""Tests for CrossAssetContext: index and sector relative strength."""
import pytest
from src.cross_asset import CrossAssetContext, CrossAssetState


def _bar(close, volume=1000):
    return {'open': close * 0.999, 'high': close * 1.001,
            'low': close * 0.998, 'close': close, 'volume': volume}


class TestCrossAssetContext:
    def test_no_reference_data(self):
        ctx = CrossAssetContext(reference_tickers=['QQQ'])
        state = ctx.update_target(_bar(100.0))
        assert state.index_available is False
        assert state.headwind_score == 0.0

    def test_with_reference_data(self):
        ctx = CrossAssetContext(reference_tickers=['QQQ'])
        # Feed 10 bars of QQQ and target
        for i in range(10):
            ctx.update_reference('QQQ', _bar(500.0 + i * 0.5))
            state = ctx.update_target(_bar(100.0 + i * 0.1))
        assert state.index_available is True
        assert state.correlation_20 != 0.0

    def test_headwind_when_index_drops(self):
        ctx = CrossAssetContext(reference_tickers=['QQQ'])
        # Index drops strongly, target rises → headwind
        for i in range(25):
            ctx.update_reference('QQQ', _bar(500.0 - i * 2.0))  # Dropping
            state = ctx.update_target(_bar(100.0 + i * 0.5))    # Rising
        # Should show some headwind (target diverging from index)
        assert state.index_available is True

    def test_tailwind_when_aligned(self):
        ctx = CrossAssetContext(reference_tickers=['QQQ'])
        # Both rising together → should have positive correlation, low headwind
        for i in range(25):
            ctx.update_reference('QQQ', _bar(500.0 + i * 1.0))
            state = ctx.update_target(_bar(100.0 + i * 0.5))
        assert state.index_available is True
        assert state.correlation_20 > 0

    def test_flat_index_no_headwind(self):
        ctx = CrossAssetContext(reference_tickers=['QQQ'])
        for i in range(25):
            ctx.update_reference('QQQ', _bar(500.0))  # Flat
            state = ctx.update_target(_bar(100.0 + i * 0.1))
        assert state.headwind_score == pytest.approx(0.0, abs=0.05)

    def test_reset(self):
        ctx = CrossAssetContext(reference_tickers=['QQQ'])
        for i in range(5):
            ctx.update_reference('QQQ', _bar(500.0))
            ctx.update_target(_bar(100.0))
        ctx.reset()
        state = ctx.update_target(_bar(100.0))
        assert state.index_available is False

    def test_sector_relative_strength(self):
        ctx = CrossAssetContext(reference_tickers=['QQQ'])
        # Target outperforms index → positive relative strength
        for i in range(25):
            ctx.update_reference('QQQ', _bar(500.0 + i * 0.2))
            state = ctx.update_target(_bar(100.0 + i * 0.8))  # 4x faster
        assert state.sector_relative > 0

    def test_to_dict(self):
        state = CrossAssetState(
            index_trend=1.0, index_momentum_5=0.5,
            headwind_score=0.3, index_available=True,
        )
        d = state.to_dict()
        assert d['index_available'] is True
        assert d['headwind_score'] == 0.3
