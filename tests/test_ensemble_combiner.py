"""Tests for AdaptiveWeightCombiner: adaptive weights with Bayesian shrinkage."""
import pytest
from src.ensemble_combiner import (
    AdaptiveWeightCombiner, CalibratedSignal, EnsembleScore,
    SourcePerformance, WEIGHT_FLOOR,
)


def _sig(source_type, source_name, direction='bullish', raw=70, cal=0.6):
    return CalibratedSignal(
        source_type=source_type, source_name=source_name,
        direction=direction, raw_confidence=raw,
        calibrated_confidence=cal,
    )


class TestAdaptiveWeightCombiner:
    def test_no_signals_returns_empty(self):
        combiner = AdaptiveWeightCombiner()
        result = combiner.combine([], 'TRENDING')
        assert result.execute is False
        assert 'No signals' in result.reasoning

    def test_single_strong_signal_below_min_sources(self):
        combiner = AdaptiveWeightCombiner(min_confirming_sources=2)
        signals = [_sig('strategy', 'momentum', cal=0.8)]
        result = combiner.combine(signals, 'TRENDING')
        # Should fail: only 1 confirming source, need 2
        assert result.execute is False

    def test_two_confirming_signals_execute(self):
        combiner = AdaptiveWeightCombiner(
            min_confirming_sources=2, base_threshold=50.0,
        )
        signals = [
            _sig('strategy', 'momentum', cal=0.7),
            _sig('pattern', 'hammer', cal=0.65),
        ]
        result = combiner.combine(signals, 'TRENDING', regime_confidence=0.8)
        # Two confirming sources, high scores → should execute
        assert result.execute is True
        assert result.direction == 'bullish'
        assert result.confirming_sources == 2

    def test_conflicting_directions_picks_stronger(self):
        combiner = AdaptiveWeightCombiner(
            min_confirming_sources=1, base_threshold=50.0,
        )
        signals = [
            _sig('strategy', 'momentum', direction='bullish', cal=0.7),
            _sig('strategy', 'mean_rev', direction='bearish', cal=0.3),
        ]
        result = combiner.combine(signals, 'TRENDING', regime_confidence=0.8)
        assert result.direction == 'bullish'

    def test_equal_weights_with_no_history(self):
        combiner = AdaptiveWeightCombiner(
            min_confirming_sources=1, base_threshold=40.0,
        )
        signals = [
            _sig('strategy', 'a', cal=0.6),
            _sig('strategy', 'b', cal=0.6),
        ]
        result = combiner.combine(signals, 'TRENDING', regime_confidence=0.8)
        # With no trade history → equal weights
        weights = result.source_weights
        if weights:
            vals = list(weights.values())
            assert abs(vals[0] - vals[1]) < 0.05  # Approximately equal

    def test_weight_adaptation_with_history(self):
        combiner = AdaptiveWeightCombiner(
            min_confirming_sources=1, base_threshold=40.0,
        )
        # Strategy A consistently wins
        for _ in range(30):
            combiner.update_outcome('strategy', 'a', 'TRENDING', True, pnl_r=1.5)
        # Strategy B consistently loses
        for _ in range(30):
            combiner.update_outcome('strategy', 'b', 'TRENDING', False, pnl_r=-1.0)

        signals = [
            _sig('strategy', 'a', cal=0.6),
            _sig('strategy', 'b', cal=0.6),
        ]
        result = combiner.combine(signals, 'TRENDING', regime_confidence=0.8)
        # Strategy A should have higher weight
        w_a = result.source_weights.get('strategy:a', 0)
        w_b = result.source_weights.get('strategy:b', 0)
        assert w_a > w_b
        # But B should not be zero (weight floor)
        assert w_b >= WEIGHT_FLOOR

    def test_dynamic_threshold_regime_uncertainty(self):
        combiner = AdaptiveWeightCombiner(base_threshold=50.0)
        signals = [
            _sig('strategy', 'a', cal=0.55),
            _sig('pattern', 'b', cal=0.55),
        ]
        # High confidence regime
        r1 = combiner.combine(signals, 'TRENDING', regime_confidence=0.9)
        # Low confidence regime → higher threshold
        r2 = combiner.combine(signals, 'TRENDING', regime_confidence=0.4)
        assert r2.threshold_used > r1.threshold_used

    def test_time_of_day_boost(self):
        combiner = AdaptiveWeightCombiner(base_threshold=50.0)
        signals = [_sig('strategy', 'a', cal=0.55), _sig('pattern', 'b', cal=0.55)]
        r1 = combiner.combine(signals, 'TRENDING', regime_confidence=0.8, time_of_day_boost=0.0)
        r2 = combiner.combine(signals, 'TRENDING', regime_confidence=0.8, time_of_day_boost=5.0)
        assert r2.threshold_used > r1.threshold_used

    def test_uncertainty_penalty_is_warmup_aware(self):
        combiner = AdaptiveWeightCombiner(base_threshold=50.0)
        signals = [_sig('strategy', 'a', cal=0.55), _sig('pattern', 'b', cal=0.55)]
        early = combiner.combine(
            signals,
            'TRENDING',
            regime_confidence=0.34,
            regime_age_bars=1,
        )
        mature = combiner.combine(
            signals,
            'TRENDING',
            regime_confidence=0.34,
            regime_age_bars=60,
        )
        assert mature.threshold_used > early.threshold_used

    def test_to_dict(self):
        result = EnsembleScore(
            execute=True, direction='bullish', combined_score=72.5,
            calibrated_probability=0.68, confirming_sources=2, total_sources=3,
        )
        d = result.to_dict()
        assert d['execute'] is True
        assert d['combined_score'] == 72.5


class TestSourcePerformance:
    def test_default_win_rate(self):
        sp = SourcePerformance()
        assert sp.win_rate == 0.5  # Prior

    def test_sharpe_few_trades(self):
        sp = SourcePerformance()
        sp.recent_pnls.append(1.0)
        assert sp.sharpe == 0.0  # < 5 trades

    def test_sharpe_positive(self):
        sp = SourcePerformance()
        # Varying positive PnLs so std > 0
        pnls = [1.0, 2.0, 1.5, 0.8, 2.5, 1.2, 1.8, 0.9, 2.1, 1.3]
        for p in pnls:
            sp.trades += 1
            sp.wins += 1
            sp.total_pnl_r += p
            sp.recent_pnls.append(p)
        assert sp.sharpe > 0
