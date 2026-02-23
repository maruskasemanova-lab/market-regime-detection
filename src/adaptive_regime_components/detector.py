"""Adaptive probabilistic regime detector with pluggable classifier abstraction."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

from ..feature_store import FeatureVector
from .classifiers import L2FlowClassifier, RuleBasedClassifier, VolatilityClassifier
from .constants import CHOPPY, MIXED, REGIME_LABELS, TRENDING
from .interfaces import RegimeProbabilityClassifier
from .state import RegimeState


@dataclass(frozen=True)
class _WeightedClassifier:
    name: str
    classifier: RegimeProbabilityClassifier
    weight_with_l2: float
    weight_without_l2: float


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

        self._classifiers = (
            _WeightedClassifier(
                name="rule",
                classifier=self._rule_classifier,
                weight_with_l2=0.35,
                weight_without_l2=0.55,
            ),
            _WeightedClassifier(
                name="l2",
                classifier=self._l2_classifier,
                weight_with_l2=0.40,
                weight_without_l2=0.0,
            ),
            _WeightedClassifier(
                name="vol",
                classifier=self._vol_classifier,
                weight_with_l2=0.25,
                weight_without_l2=0.45,
            ),
        )

        # Kept for backward compatibility with existing callers/tests.
        self._weights = {
            spec.name: spec.weight_with_l2 for spec in self._classifiers
        }
        self._weights_no_l2 = {
            spec.name: spec.weight_without_l2 for spec in self._classifiers
        }

        self._smoothing_alpha = smoothing_alpha
        self._smoothed_probs: Optional[Dict[str, float]] = None

        self._transition_lookback = transition_lookback
        self._recent_primaries: deque = deque(maxlen=transition_lookback)
        self._prev_state: Optional[RegimeState] = None

    def _resolve_weights(self, has_l2_coverage: bool) -> Dict[str, float]:
        if has_l2_coverage:
            return dict(self._weights)
        return dict(self._weights_no_l2)

    def detect(self, fv: FeatureVector) -> RegimeState:
        """
        Detect regime from feature vector.
        Returns RegimeState with probabilities.
        """

        classifier_outputs: Dict[str, Dict[str, float]] = {
            spec.name: spec.classifier.classify(fv)
            for spec in self._classifiers
        }

        weights = self._resolve_weights(bool(fv.l2_has_coverage))

        combined: Dict[str, float] = {}
        for regime in REGIME_LABELS:
            combined[regime] = sum(
                weights[spec.name] * classifier_outputs[spec.name][regime]
                for spec in self._classifiers
            )

        total = sum(combined.values())
        if total > 0:
            combined = {k: v / total for k, v in combined.items()}

        if self._smoothed_probs is None:
            self._smoothed_probs = dict(combined)
        else:
            alpha = max(
                self._smoothing_alpha,
                2.0 / float(max(2, fv.bar_index + 1)),
            )
            for regime in combined:
                self._smoothed_probs[regime] = (
                    alpha * combined[regime]
                    + (1 - alpha) * self._smoothed_probs[regime]
                )
            total = sum(self._smoothed_probs.values())
            if total > 0:
                self._smoothed_probs = {
                    key: value / total for key, value in self._smoothed_probs.items()
                }

        primary = max(self._smoothed_probs, key=self._smoothed_probs.get)
        confidence = self._smoothed_probs[primary]
        micro = self._l2_classifier.classify_micro(fv)

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
        0.0 = perfectly stable
        1.0 = maximum instability
        """

        if len(self._recent_primaries) < 2:
            return 0.0

        changes = sum(
            1
            for idx in range(1, len(self._recent_primaries))
            if self._recent_primaries[idx] != self._recent_primaries[idx - 1]
        )
        return changes / (len(self._recent_primaries) - 1)

    def reset(self):
        """Reset state for new session."""

        self._smoothed_probs = None
        self._recent_primaries.clear()
        self._prev_state = None
