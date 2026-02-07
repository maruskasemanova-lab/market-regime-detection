"""
Quality-Based Position Sizing.

Scales position size based on signal quality, regime confidence,
and strategy edge health. Higher-quality signals get larger positions.

Anti-bias:
  - Hard bounds [0.25%, 2.0%] prevent extreme concentration
  - Degraded edge health automatically reduces size
  - Regime uncertainty reduces size (not just threshold)
  - No scaling until sufficient calibration data exists
"""
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

from .adaptive_regime import RegimeState
from .edge_monitor import EdgeMonitor, EdgeStatus, EdgeHealth
from .ensemble_combiner import EnsembleScore

logger = logging.getLogger(__name__)

# Hard bounds on risk per trade
MIN_RISK_PCT = 0.25
MAX_RISK_PCT = 2.0

# Health multipliers
HEALTH_MULTIPLIERS = {
    EdgeStatus.STRONG: 1.2,
    EdgeStatus.NORMAL: 1.0,
    EdgeStatus.DEGRADED: 0.6,
    EdgeStatus.DEAD: 0.0,
    EdgeStatus.INSUFFICIENT: 0.8,  # Conservative when unknown
}


@dataclass
class SizingResult:
    """Position sizing calculation result."""
    base_risk_pct: float = 1.0
    quality_multiplier: float = 1.0
    regime_multiplier: float = 1.0
    health_multiplier: float = 1.0
    final_risk_pct: float = 1.0
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'base_risk_pct': round(self.base_risk_pct, 3),
            'quality_mult': round(self.quality_multiplier, 3),
            'regime_mult': round(self.regime_multiplier, 3),
            'health_mult': round(self.health_multiplier, 3),
            'final_risk_pct': round(self.final_risk_pct, 3),
            'reasoning': self.reasoning,
        }


class QualityPositionSizer:
    """
    Scales position size based on multi-dimensional quality assessment.

    final_risk = base_risk * quality_mult * regime_mult * health_mult

    Where:
      quality_mult ∈ [0.5, 1.5]: based on calibrated probability
      regime_mult ∈ [0.5, 1.0]: based on regime confidence
      health_mult ∈ [0.0, 1.2]: based on edge health status
    """

    def __init__(
        self,
        base_risk_pct: float = 1.0,
        edge_monitor: Optional[EdgeMonitor] = None,
    ):
        self._base_risk = base_risk_pct
        self._edge_monitor = edge_monitor

    def calculate_risk_pct(
        self,
        ensemble_score: Optional[EnsembleScore] = None,
        regime_state: Optional[RegimeState] = None,
        strategy_name: str = "",
        regime_label: str = "MIXED",
    ) -> SizingResult:
        """
        Calculate risk percentage for a trade.

        Returns SizingResult with final_risk_pct to use.
        """
        result = SizingResult(base_risk_pct=self._base_risk)

        # Quality multiplier: based on calibrated probability
        if ensemble_score is not None and ensemble_score.calibrated_probability > 0:
            prob = ensemble_score.calibrated_probability
            # Map P(profitable) to multiplier:
            # P=0.5 → 0.5x (break-even → minimum size)
            # P=0.6 → 1.0x (decent edge → normal size)
            # P=0.75 → 1.5x (strong edge → maximum size)
            result.quality_multiplier = max(0.5, min(1.5, 0.5 + (prob - 0.5) * 4.0))
        else:
            result.quality_multiplier = 0.8  # Conservative when no calibration

        # Regime multiplier: lower size in uncertain regimes
        if regime_state is not None:
            # Confident regime → normal size, uncertain → reduced
            # conf=0.33 (uniform) → 0.5x, conf=0.8+ → 1.0x
            result.regime_multiplier = max(0.5, min(1.0,
                0.5 + (regime_state.confidence - 0.33) * 1.5
            ))
            # Extra penalty during regime transitions
            if regime_state.transition_velocity > 0.3:
                result.regime_multiplier *= max(0.7, 1.0 - regime_state.transition_velocity * 0.3)
        else:
            result.regime_multiplier = 0.8

        # Health multiplier: based on edge monitor
        if self._edge_monitor is not None and strategy_name:
            health = self._edge_monitor.get_health(strategy_name, regime_label)
            result.health_multiplier = HEALTH_MULTIPLIERS.get(
                health.status, 1.0
            )
        else:
            result.health_multiplier = 1.0

        # Final calculation with hard bounds
        raw_risk = (
            result.base_risk_pct
            * result.quality_multiplier
            * result.regime_multiplier
            * result.health_multiplier
        )
        result.final_risk_pct = max(MIN_RISK_PCT, min(MAX_RISK_PCT, raw_risk))

        # Build reasoning
        parts = [f"base={result.base_risk_pct:.1f}%"]
        if result.quality_multiplier != 1.0:
            parts.append(f"quality={result.quality_multiplier:.2f}x")
        if result.regime_multiplier != 1.0:
            parts.append(f"regime={result.regime_multiplier:.2f}x")
        if result.health_multiplier != 1.0:
            parts.append(f"health={result.health_multiplier:.2f}x")
        parts.append(f"→ {result.final_risk_pct:.2f}%")
        result.reasoning = " × ".join(parts)

        return result
