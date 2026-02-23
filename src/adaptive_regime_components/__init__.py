"""Adaptive regime detector internals split into focused modules."""

from .classifiers import L2FlowClassifier, RuleBasedClassifier, VolatilityClassifier
from .constants import CHOPPY, MICRO_REGIMES, MIXED, TRENDING
from .detector import AdaptiveRegimeDetector
from .state import RegimeState

__all__ = [
    "AdaptiveRegimeDetector",
    "RegimeState",
    "RuleBasedClassifier",
    "L2FlowClassifier",
    "VolatilityClassifier",
    "TRENDING",
    "CHOPPY",
    "MIXED",
    "MICRO_REGIMES",
]
