"""Tests for AdaptiveRegimeDetector: probabilistic regime detection."""
import pytest
from src.feature_store import FeatureStore, FeatureVector
from src.adaptive_regime import (
    AdaptiveRegimeDetector, RegimeState,
    RuleBasedClassifier, L2FlowClassifier, VolatilityClassifier,
    TRENDING, CHOPPY, MIXED,
)


def _make_fv(**kwargs) -> FeatureVector:
    """Create a FeatureVector with overrides."""
    defaults = {
        'bar_index': 1, 'close': 100.0, 'adx_14': 25.0,
        'trend_efficiency': 0.5, 'rsi_14': 50.0, 'atr_14': 1.0,
        'atr_z': 0.0, 'roc_5': 0.0, 'roc_10': 0.0,
        'momentum_z': 0.0, 'atr_pct_rank': 0.5, 'range_pct_rank': 0.5,
        'l2_has_coverage': False, 'l2_signed_aggression': 0.0,
        'l2_directional_consistency': 0.0, 'l2_sweep_intensity': 0.0,
        'l2_absorption_rate': 0.0, 'l2_book_pressure': 0.0,
        'l2_large_trader_activity': 0.0,
    }
    defaults.update(kwargs)
    return FeatureVector(**defaults)


class TestRuleBasedClassifier:
    def test_trending(self):
        clf = RuleBasedClassifier()
        probs = clf.classify(_make_fv(adx_14=40, trend_efficiency=0.8, roc_5=1.5))
        assert probs[TRENDING] > probs[CHOPPY]
        assert probs[TRENDING] > probs[MIXED]

    def test_choppy(self):
        clf = RuleBasedClassifier()
        probs = clf.classify(_make_fv(adx_14=12, trend_efficiency=0.2, roc_5=0.01))
        assert probs[CHOPPY] > probs[TRENDING]

    def test_probabilities_sum_to_one(self):
        clf = RuleBasedClassifier()
        probs = clf.classify(_make_fv())
        total = sum(probs.values())
        assert total == pytest.approx(1.0, abs=0.01)


class TestL2FlowClassifier:
    def test_no_l2_returns_uniform(self):
        clf = L2FlowClassifier()
        probs = clf.classify(_make_fv(l2_has_coverage=False))
        assert abs(probs[TRENDING] - probs[CHOPPY]) < 0.02

    def test_strong_flow_is_trending(self):
        clf = L2FlowClassifier()
        probs = clf.classify(_make_fv(
            l2_has_coverage=True, l2_signed_aggression=0.15,
            l2_directional_consistency=0.7, l2_sweep_intensity=0.3,
            l2_large_trader_activity=0.2,
        ))
        assert probs[TRENDING] > probs[CHOPPY]

    def test_absorption_is_choppy(self):
        clf = L2FlowClassifier()
        probs = clf.classify(_make_fv(
            l2_has_coverage=True, l2_signed_aggression=0.01,
            l2_directional_consistency=0.2, l2_absorption_rate=0.7,
        ))
        assert probs[CHOPPY] > probs[TRENDING]


class TestAdaptiveRegimeDetector:
    def test_basic_detection(self):
        detector = AdaptiveRegimeDetector()
        fv = _make_fv(adx_14=35, trend_efficiency=0.7, roc_5=1.0,
                       atr_pct_rank=0.8, momentum_z=1.5)
        state = detector.detect(fv)
        assert isinstance(state, RegimeState)
        assert state.primary in (TRENDING, CHOPPY, MIXED)
        assert 0.0 <= state.confidence <= 1.0
        assert sum(state.probabilities.values()) == pytest.approx(1.0, abs=0.01)

    def test_smoothing(self):
        detector = AdaptiveRegimeDetector(smoothing_alpha=0.3)
        # Feed trending signals
        for _ in range(10):
            state = detector.detect(_make_fv(adx_14=40, trend_efficiency=0.8,
                                              roc_5=1.5, momentum_z=2.0,
                                              atr_pct_rank=0.8))
        assert state.primary == TRENDING
        assert state.confidence > 0.5

    def test_transition_velocity(self):
        detector = AdaptiveRegimeDetector()
        # Alternate regimes → high velocity
        for i in range(10):
            if i % 2 == 0:
                fv = _make_fv(adx_14=40, trend_efficiency=0.8, roc_5=2.0,
                              momentum_z=2.0, atr_pct_rank=0.8)
            else:
                fv = _make_fv(adx_14=10, trend_efficiency=0.1, roc_5=0.0,
                              momentum_z=0.0, atr_pct_rank=0.2)
            state = detector.detect(fv)
        assert state.transition_velocity > 0.3

    def test_reset(self):
        detector = AdaptiveRegimeDetector()
        detector.detect(_make_fv())
        detector.reset()
        assert detector._smoothed_probs is None

    def test_is_confident(self):
        state = RegimeState(
            probabilities={TRENDING: 0.7, CHOPPY: 0.2, MIXED: 0.1},
            primary=TRENDING, confidence=0.7,
        )
        assert state.is_confident is True

        state2 = RegimeState(
            probabilities={TRENDING: 0.4, CHOPPY: 0.35, MIXED: 0.25},
            primary=TRENDING, confidence=0.4,
        )
        assert state2.is_confident is False
