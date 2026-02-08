"""
Adaptive Probabilistic Regime Detector.

Replaces binary TRENDING/CHOPPY/MIXED with probability distributions.
Uses an ensemble of rule-based, statistical, and L2-flow methods
to produce regime probabilities that strategies can weight against.

Anti-bias:
  - Ensemble of 3 independent methods reduces single-model overfitting
  - Probabilities eliminate boundary bias (ADX 24.9 vs 25.1)
  - Transition velocity metric adds caution during uncertain periods
  - Hysteresis built into probability smoothing, not binary flips
"""
import math
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .feature_store import FeatureVector

logger = logging.getLogger(__name__)

# Regime labels
TRENDING = "TRENDING"
CHOPPY = "CHOPPY"
MIXED = "MIXED"

# Micro-regime labels
MICRO_REGIMES = [
    "TRENDING_UP", "TRENDING_DOWN", "BREAKOUT",
    "ABSORPTION", "CHOPPY", "MIXED",
]


@dataclass
class RegimeState:
    """Probabilistic regime output."""
    probabilities: Dict[str, float] = field(
        default_factory=lambda: {TRENDING: 0.33, CHOPPY: 0.33, MIXED: 0.34}
    )
    primary: str = MIXED
    confidence: float = 0.34
    micro_regime: str = "MIXED"
    transition_velocity: float = 0.0   # How fast regime is changing (0=stable, 1=rapid)
    bar_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'probabilities': dict(self.probabilities),
            'primary': self.primary,
            'confidence': self.confidence,
            'micro_regime': self.micro_regime,
            'transition_velocity': self.transition_velocity,
            'bar_index': self.bar_index,
        }

    @property
    def is_confident(self) -> bool:
        """Whether the regime classification is confident (> 0.55)."""
        return self.confidence > 0.55

    @property
    def is_transition(self) -> bool:
        """Whether the market is in a transition/noise state.
        
        Returns True when:
          - micro_regime is TRANSITION
          - macro TRENDING but micro is CHOPPY/MIXED (divergence)
        
        Note: We don't check low confidence here because defaults are low.
        Low confidence is already penalized via regime_confidence in threshold.
        """
        if self.micro_regime == "TRANSITION":
            return True
        if self.primary == "TRENDING" and self.micro_regime in {"CHOPPY", "MIXED"}:
            return True
        return False

    def probability(self, regime: str) -> float:
        return self.probabilities.get(regime, 0.0)


class RuleBasedClassifier:
    """
    Rule-based regime classification using ADX, trend efficiency, and volatility.
    Outputs soft probabilities instead of hard categories.
    """

    def classify(self, fv: FeatureVector) -> Dict[str, float]:
        """Classify regime using normalized features."""
        adx = fv.adx_14
        trend_eff = fv.trend_efficiency
        atr_z = fv.atr_z
        roc_5 = abs(fv.roc_5)

        # Handle None ADX during warmup - use neutral value
        if adx is None:
            adx = 20.0  # Neutral: neither strongly trending nor choppy

        # Trending score: high ADX + high trend efficiency + directional momentum
        trending_score = 0.0
        if adx > 15:
            trending_score += min(1.0, (adx - 15) / 25)  # Sigmoid-like ramp
        trending_score += min(0.5, trend_eff)
        trending_score += min(0.3, roc_5 / 2.0)  # Directional momentum
        trending_score = min(1.0, trending_score / 1.5)  # Normalize

        # Choppy score: low ADX + low trend efficiency + high volatility
        choppy_score = 0.0
        if adx < 30:
            choppy_score += min(1.0, (30 - adx) / 20)
        choppy_score += max(0.0, 0.5 - trend_eff)
        if atr_z > 0.5:  # High relative volatility with no direction = choppy
            choppy_score += min(0.3, (atr_z - 0.5) * 0.3)
        choppy_score = min(1.0, choppy_score / 1.3)

        # Mixed = complement
        mixed_score = max(0.0, 1.0 - trending_score - choppy_score)

        # Normalize to probabilities
        total = trending_score + choppy_score + mixed_score
        if total < 1e-10:
            return {TRENDING: 0.33, CHOPPY: 0.33, MIXED: 0.34}

        return {
            TRENDING: trending_score / total,
            CHOPPY: choppy_score / total,
            MIXED: mixed_score / total,
        }


class L2FlowClassifier:
    """
    L2 order flow-based regime classification.
    Uses flow metrics to provide microstructure-informed regime probabilities.
    """

    def classify(self, fv: FeatureVector) -> Dict[str, float]:
        """Classify regime using L2 flow features."""
        if not fv.l2_has_coverage:
            # No L2 data → uniform prior (uninformative)
            return {TRENDING: 0.33, CHOPPY: 0.33, MIXED: 0.34}

        aggression = abs(fv.l2_signed_aggression)
        consistency = fv.l2_directional_consistency
        sweep = fv.l2_sweep_intensity
        absorption = fv.l2_absorption_rate
        pressure = abs(fv.l2_book_pressure)
        large_trader = fv.l2_large_trader_activity

        # Trending: strong directional flow
        trending_score = 0.0
        trending_score += min(0.4, aggression * 3.0)
        trending_score += min(0.3, consistency * 0.5)
        trending_score += min(0.2, sweep * 1.5)
        trending_score += min(0.1, large_trader * 0.5)

        # Choppy: high absorption, weak direction
        choppy_score = 0.0
        choppy_score += min(0.4, absorption * 0.7)
        choppy_score += min(0.3, max(0.0, 0.5 - aggression) * 1.5)
        choppy_score += min(0.2, max(0.0, 0.5 - consistency) * 0.6)

        # Mixed
        mixed_score = max(0.0, 1.0 - trending_score - choppy_score)

        total = trending_score + choppy_score + mixed_score
        if total < 1e-10:
            return {TRENDING: 0.33, CHOPPY: 0.33, MIXED: 0.34}

        return {
            TRENDING: trending_score / total,
            CHOPPY: choppy_score / total,
            MIXED: mixed_score / total,
        }

    def classify_micro(self, fv: FeatureVector) -> str:
        """Detailed micro-regime from L2 flow."""
        # Handle None ADX during warmup
        adx = fv.adx_14 if fv.adx_14 is not None else 20.0

        if not fv.l2_has_coverage:
            # Fallback to price-only micro
            if adx > 25 and fv.trend_efficiency > 0.5:
                return "TRENDING_UP" if fv.roc_5 >= 0 else "TRENDING_DOWN"
            if adx < 20:
                return "CHOPPY"
            return "MIXED"

        aggression = fv.l2_signed_aggression
        consistency = fv.l2_directional_consistency
        sweep = fv.l2_sweep_intensity
        absorption = fv.l2_absorption_rate
        pressure = fv.l2_book_pressure
        large_trader = fv.l2_large_trader_activity

        # Breakout: sweeps + large traders + directional
        if (sweep >= 0.25 and consistency >= 0.50
                and abs(aggression) >= 0.08 and large_trader >= 0.12):
            return "BREAKOUT"

        # Absorption: high absorption, constrained price
        if absorption >= 0.45 and abs(pressure) >= 0.08:
            return "ABSORPTION"

        # Trending: directional flow
        if abs(aggression) >= 0.06 and consistency >= 0.45 and adx >= 20:
            return "TRENDING_UP" if aggression > 0 else "TRENDING_DOWN"

        # Choppy
        if adx < 18 and fv.trend_efficiency < 0.35:
            return "CHOPPY"

        return "MIXED"


class VolatilityClassifier:
    """
    Volatility-regime classifier using normalized ATR and return distribution.
    Lightweight statistical method - no pandas/sklearn needed.
    """

    def classify(self, fv: FeatureVector) -> Dict[str, float]:
        """Classify based on volatility regime."""
        atr_pct = fv.atr_pct_rank  # 0-1 percentile rank
        range_pct = fv.range_pct_rank
        vol_z = fv.atr_z
        boll_w = fv.bollinger_width

        # High volatility + direction = trending
        # High volatility + no direction = mixed
        # Low volatility = choppy
        has_direction = abs(fv.momentum_z) > 0.5

        if atr_pct > 0.7 and has_direction:
            trending_score = 0.6 + (atr_pct - 0.7) * 0.5
            choppy_score = 0.1
        elif atr_pct > 0.7 and not has_direction:
            trending_score = 0.2
            choppy_score = 0.2
        elif atr_pct < 0.3:
            trending_score = 0.1
            choppy_score = 0.5 + (0.3 - atr_pct) * 0.5
        else:
            # Middle range
            trending_score = 0.25 + (0.25 if has_direction else 0.0)
            choppy_score = 0.25 + (0.0 if has_direction else 0.15)

        mixed_score = max(0.0, 1.0 - trending_score - choppy_score)

        total = trending_score + choppy_score + mixed_score
        if total < 1e-10:
            return {TRENDING: 0.33, CHOPPY: 0.33, MIXED: 0.34}

        return {
            TRENDING: trending_score / total,
            CHOPPY: choppy_score / total,
            MIXED: mixed_score / total,
        }


class AdaptiveRegimeDetector:
    """
    Probabilistic regime detection using ensemble of classifiers.

    Combines:
      1. Rule-based (ADX, trend efficiency, momentum)
      2. L2 flow (order flow microstructure)
      3. Volatility (ATR/range distribution)

    Outputs RegimeState with probabilities, not binary labels.
    """

    def __init__(
        self,
        smoothing_alpha: float = 0.3,
        transition_lookback: int = 10,
    ):
        self._rule_classifier = RuleBasedClassifier()
        self._l2_classifier = L2FlowClassifier()
        self._vol_classifier = VolatilityClassifier()

        # Smoothing: exponential moving average of probabilities
        self._smoothing_alpha = smoothing_alpha
        self._smoothed_probs: Optional[Dict[str, float]] = None

        # Transition tracking
        self._transition_lookback = transition_lookback
        self._recent_primaries: deque = deque(maxlen=transition_lookback)
        self._prev_state: Optional[RegimeState] = None

        # Classifier weights (adaptable based on accuracy)
        self._weights = {
            'rule': 0.35,
            'l2': 0.40,
            'vol': 0.25,
        }
        # When L2 unavailable, reweight
        self._weights_no_l2 = {
            'rule': 0.55,
            'l2': 0.0,
            'vol': 0.45,
        }

    def detect(self, fv: FeatureVector) -> RegimeState:
        """
        Detect regime from feature vector.
        Returns RegimeState with probabilities.
        """
        # Get probabilities from each classifier
        rule_probs = self._rule_classifier.classify(fv)
        l2_probs = self._l2_classifier.classify(fv)
        vol_probs = self._vol_classifier.classify(fv)

        # Select weights based on L2 availability
        weights = self._weights if fv.l2_has_coverage else self._weights_no_l2

        # Weighted ensemble combination
        combined = {}
        for regime in [TRENDING, CHOPPY, MIXED]:
            combined[regime] = (
                weights['rule'] * rule_probs[regime]
                + weights['l2'] * l2_probs[regime]
                + weights['vol'] * vol_probs[regime]
            )

        # Normalize
        total = sum(combined.values())
        if total > 0:
            combined = {k: v / total for k, v in combined.items()}

        # Apply exponential smoothing to reduce noise
        if self._smoothed_probs is None:
            self._smoothed_probs = dict(combined)
        else:
            alpha = self._smoothing_alpha
            for regime in combined:
                self._smoothed_probs[regime] = (
                    alpha * combined[regime]
                    + (1 - alpha) * self._smoothed_probs[regime]
                )
            # Re-normalize after smoothing
            total = sum(self._smoothed_probs.values())
            if total > 0:
                self._smoothed_probs = {
                    k: v / total for k, v in self._smoothed_probs.items()
                }

        # Primary regime = argmax
        primary = max(self._smoothed_probs, key=self._smoothed_probs.get)
        confidence = self._smoothed_probs[primary]

        # Micro-regime from L2
        micro = self._l2_classifier.classify_micro(fv)

        # Transition velocity: how much primary regime has changed recently
        self._recent_primaries.append(primary)
        transition_vel = self._compute_transition_velocity()

        state = RegimeState(
            probabilities=dict(self._smoothed_probs),
            primary=primary,
            confidence=confidence,
            micro_regime=micro,
            transition_velocity=transition_vel,
            bar_index=fv.bar_index,
        )

        self._prev_state = state
        return state

    def _compute_transition_velocity(self) -> float:
        """
        Measure how unstable the regime is.
        0.0 = perfectly stable (same regime for all lookback bars)
        1.0 = maximum instability (regime changes every bar)
        """
        if len(self._recent_primaries) < 2:
            return 0.0

        changes = sum(
            1 for i in range(1, len(self._recent_primaries))
            if self._recent_primaries[i] != self._recent_primaries[i - 1]
        )
        return changes / (len(self._recent_primaries) - 1)

    def to_legacy_regime(self, state: RegimeState) -> str:
        """Convert RegimeState to legacy Regime enum string for backward compatibility."""
        return state.primary

    def to_legacy_micro(self, state: RegimeState) -> str:
        """Convert to legacy micro-regime string."""
        return state.micro_regime

    def reset(self):
        """Reset state for new session."""
        self._smoothed_probs = None
        self._recent_primaries.clear()
        self._prev_state = None
