"""Classifier interfaces for adaptive regime components."""

from __future__ import annotations

from typing import Dict, Protocol

from ..feature_store import FeatureVector


class RegimeProbabilityClassifier(Protocol):
    """Any classifier that maps features to macro regime probabilities."""

    def classify(self, fv: FeatureVector) -> Dict[str, float]:
        ...
