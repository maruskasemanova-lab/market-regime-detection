"""Output state model for adaptive regime detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from .constants import MIXED, default_probabilities


@dataclass
class RegimeState:
    """Probabilistic regime output."""

    probabilities: Dict[str, float] = field(default_factory=default_probabilities)
    primary: str = MIXED
    confidence: float = 0.50
    micro_regime: str = "UNKNOWN"
    transition_velocity: float = 0.0
    bar_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probabilities": dict(self.probabilities),
            "primary": self.primary,
            "confidence": self.confidence,
            "micro_regime": self.micro_regime,
            "transition_velocity": self.transition_velocity,
            "bar_index": self.bar_index,
        }

    @property
    def is_confident(self) -> bool:
        """Whether the regime classification is confident (> 0.55)."""

        return self.confidence > 0.55

    @property
    def is_transition(self) -> bool:
        """Whether the market is in transition/noise state."""

        if self.micro_regime == "TRANSITION":
            return True
        if self.primary == "TRENDING" and self.micro_regime in {"CHOPPY", "MIXED"}:
            return True
        return False

    def probability(self, regime: str) -> float:
        return self.probabilities.get(regime, 0.0)
