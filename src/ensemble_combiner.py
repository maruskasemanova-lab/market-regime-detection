"""
Adaptive Ensemble Combiner with Bayesian Shrinkage.

Replaces fixed pattern_weight=0.4 / strategy_weight=0.6 with
adaptive weights that learn from recent strategy performance.

Anti-bias:
  - Bayesian shrinkage toward equal weights (1/N) with few trades
  - Regime-conditional weights (separate adaptation per regime)
  - Minimum weight floor (0.05) prevents total suppression
  - Softmax temperature controls adaptation speed
"""
import math
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# Minimum trades before weights adapt from prior
MIN_TRADES_FOR_ADAPTATION = 10
# Full adaptation threshold
FULL_ADAPTATION_TRADES = 50
# Floor: no signal source weight drops below this
WEIGHT_FLOOR = 0.05
# Softmax temperature for weight computation
SOFTMAX_TEMPERATURE = 2.0


@dataclass
class CalibratedSignal:
    """A signal with calibrated confidence."""
    source_type: str          # 'pattern', 'strategy', 'feature'
    source_name: str          # e.g. 'momentum_flow', 'hammer', 'rsi_oversold'
    direction: str            # 'bullish' or 'bearish'
    raw_confidence: float     # Original 0-100
    calibrated_confidence: float  # P(profitable) from calibrator (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleScore:
    """Combined ensemble decision."""
    execute: bool = False
    direction: Optional[str] = None
    combined_score: float = 0.0           # 0-100 normalized
    calibrated_probability: float = 0.0    # P(profitable) weighted
    confirming_sources: int = 0
    total_sources: int = 0
    source_weights: Dict[str, float] = field(default_factory=dict)
    source_contributions: Dict[str, float] = field(default_factory=dict)
    threshold_used: float = 0.0
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'execute': self.execute,
            'direction': self.direction,
            'combined_score': round(self.combined_score, 2),
            'calibrated_probability': round(self.calibrated_probability, 4),
            'confirming_sources': self.confirming_sources,
            'total_sources': self.total_sources,
            'source_weights': {k: round(v, 3) for k, v in self.source_weights.items()},
            'source_contributions': {k: round(v, 3) for k, v in self.source_contributions.items()},
            'threshold_used': round(self.threshold_used, 2),
            'reasoning': self.reasoning,
        }


@dataclass
class SourcePerformance:
    """Rolling performance tracker for a signal source."""
    trades: int = 0
    wins: int = 0
    total_pnl_r: float = 0.0
    recent_pnls: deque = field(default_factory=lambda: deque(maxlen=30))

    @property
    def win_rate(self) -> float:
        if self.trades == 0:
            return 0.5  # Prior
        return self.wins / self.trades

    @property
    def sharpe(self) -> float:
        """Rolling Sharpe from recent PnLs."""
        if len(self.recent_pnls) < 5:
            return 0.0
        pnls = list(self.recent_pnls)
        mean = sum(pnls) / len(pnls)
        var = sum((p - mean) ** 2 for p in pnls) / len(pnls)
        std = math.sqrt(var) if var > 0 else 0.0
        if std < 1e-10:
            return 0.0
        return mean / std

    @property
    def expectancy(self) -> float:
        """Average R per trade."""
        if self.trades == 0:
            return 0.0
        return self.total_pnl_r / self.trades


class AdaptiveWeightCombiner:
    """
    Combines multiple signal sources with adaptive weights.

    Weight computation:
      w_i = softmax(alpha * performance_score_i + (1-alpha) * prior_i)

    Where:
      alpha = min(n_trades / FULL_ADAPTATION_TRADES, 1.0)
      performance_score = sharpe * 0.5 + (win_rate - 0.5) * 0.3 + expectancy * 0.2

    Shrinkage:
      With 0 trades: weights = 1/N (equal)
      With 50+ trades: weights fully adapt to performance
      Between: linear interpolation
    """

    def __init__(
        self,
        min_confirming_sources: int = 2,
        base_threshold: float = 55.0,
    ):
        self._min_confirming = min_confirming_sources
        self._base_threshold = base_threshold

        # Performance tracking: (source_type, source_name, regime) → SourcePerformance
        self._performance: Dict[Tuple[str, str, str], SourcePerformance] = {}

        # Overall regime-level performance for threshold adaptation
        self._regime_performance: Dict[str, SourcePerformance] = {}

    def combine(
        self,
        signals: List[CalibratedSignal],
        regime: str,
        regime_confidence: float = 0.5,
        time_of_day_boost: float = 0.0,
        transition_velocity: float = 0.0,
        trend_efficiency: Optional[float] = None,
        is_transition: bool = False,
    ) -> EnsembleScore:
        """
        Combine calibrated signals into ensemble decision.

        Args:
            signals: List of calibrated signals from all sources
            regime: Current regime label
            regime_confidence: Probability of current regime (0-1)
            time_of_day_boost: Additional threshold penalty for quiet hours
            transition_velocity: Regime instability (0-1)
            trend_efficiency: Current trend efficiency (0-1), None if unavailable
            is_transition: True if in transition/noise state (macro/micro divergence)
        """
        if not signals:
            return EnsembleScore(reasoning="No signals")

        # Separate by direction
        bullish = [s for s in signals if s.direction == 'bullish']
        bearish = [s for s in signals if s.direction == 'bearish']

        # Determine dominant direction by count and average confidence
        if not bullish and not bearish:
            return EnsembleScore(reasoning="No directional signals")

        bull_score = (
            sum(s.calibrated_confidence for s in bullish) / len(bullish)
            if bullish else 0.0
        )
        bear_score = (
            sum(s.calibrated_confidence for s in bearish) / len(bearish)
            if bearish else 0.0
        )

        if bull_score >= bear_score and bullish:
            direction = 'bullish'
            aligned_signals = bullish
        elif bearish:
            direction = 'bearish'
            aligned_signals = bearish
        else:
            return EnsembleScore(reasoning="No clear directional consensus")

        # Compute adaptive weights for each signal source
        weights = self._compute_weights(aligned_signals, regime)

        # Weighted combination of calibrated confidences
        weighted_sum = 0.0
        weight_total = 0.0
        source_weights = {}
        source_contributions = {}

        for signal in aligned_signals:
            key = f"{signal.source_type}:{signal.source_name}"
            w = weights.get(key, 1.0 / len(aligned_signals))
            weighted_sum += w * signal.calibrated_confidence
            weight_total += w
            source_weights[key] = w
            source_contributions[key] = w * signal.calibrated_confidence

        if weight_total > 0:
            calibrated_prob = weighted_sum / weight_total
        else:
            calibrated_prob = 0.0

        # Convert to 0-100 scale for threshold comparison
        combined_score = calibrated_prob * 100.0

        # Dynamic threshold
        threshold = self._compute_dynamic_threshold(
            regime, regime_confidence, time_of_day_boost, transition_velocity,
            trend_efficiency, is_transition,
        )

        # Confirming sources check (lowered from 0.45 to 0.35 —
        # calibrated confidences typically range 0.35-0.65, so 0.45 was
        # too strict for most individual sources to qualify as "confirming")
        confirming = sum(
            1 for s in aligned_signals if s.calibrated_confidence > 0.35
        )

        # Execute decision
        execute = (
            combined_score >= threshold
            and confirming >= self._min_confirming
        )

        reasons = []
        if combined_score < threshold:
            reasons.append(f"score {combined_score:.1f} < threshold {threshold:.1f}")
        if confirming < self._min_confirming:
            reasons.append(
                f"confirming {confirming} < min {self._min_confirming}"
            )
        if execute:
            reasons.append(
                f"score {combined_score:.1f} >= threshold {threshold:.1f}, "
                f"{confirming} sources confirm"
            )

        return EnsembleScore(
            execute=execute,
            direction=direction,
            combined_score=combined_score,
            calibrated_probability=calibrated_prob,
            confirming_sources=confirming,
            total_sources=len(aligned_signals),
            source_weights=source_weights,
            source_contributions=source_contributions,
            threshold_used=threshold,
            reasoning="; ".join(reasons),
        )

    def update_outcome(
        self,
        source_type: str,
        source_name: str,
        regime: str,
        was_profitable: bool,
        pnl_r: float = 0.0,
    ):
        """Record trade outcome for weight adaptation."""
        key = (source_type, source_name, regime)
        if key not in self._performance:
            self._performance[key] = SourcePerformance()
        perf = self._performance[key]
        perf.trades += 1
        if was_profitable:
            perf.wins += 1
        perf.total_pnl_r += pnl_r
        perf.recent_pnls.append(pnl_r)

        # Regime-level tracking
        if regime not in self._regime_performance:
            self._regime_performance[regime] = SourcePerformance()
        rp = self._regime_performance[regime]
        rp.trades += 1
        if was_profitable:
            rp.wins += 1
        rp.total_pnl_r += pnl_r
        rp.recent_pnls.append(pnl_r)

    def _compute_weights(
        self,
        signals: List[CalibratedSignal],
        regime: str,
    ) -> Dict[str, float]:
        """
        Compute adaptive weights with Bayesian shrinkage.

        With few trades: equal weights (1/N).
        As trades accumulate: shift toward performance-based weights.
        """
        n = len(signals)
        if n == 0:
            return {}

        equal_weight = 1.0 / n
        keys = [f"{s.source_type}:{s.source_name}" for s in signals]

        # Get performance scores
        perf_scores = {}
        total_trades = 0
        for signal in signals:
            key = f"{signal.source_type}:{signal.source_name}"
            perf_key = (signal.source_type, signal.source_name, regime)
            perf = self._performance.get(perf_key)

            if perf and perf.trades >= 5:
                score = (
                    perf.sharpe * 0.5
                    + (perf.win_rate - 0.5) * 0.3
                    + perf.expectancy * 0.2
                )
                perf_scores[key] = score
                total_trades += perf.trades
            else:
                perf_scores[key] = 0.0

        # Adaptation factor: how much to trust performance vs prior
        avg_trades = total_trades / n if n > 0 else 0
        alpha = min(1.0, avg_trades / FULL_ADAPTATION_TRADES)

        # Softmax of performance scores
        if alpha > 0.01 and perf_scores:
            # Temperature-scaled softmax
            max_score = max(perf_scores.values())
            exp_scores = {}
            for key in keys:
                s = perf_scores.get(key, 0.0)
                # Shift by max for numerical stability
                exp_scores[key] = math.exp(
                    (s - max_score) / SOFTMAX_TEMPERATURE
                )
            exp_total = sum(exp_scores.values())
            if exp_total > 0:
                perf_weights = {
                    k: v / exp_total for k, v in exp_scores.items()
                }
            else:
                perf_weights = {k: equal_weight for k in keys}
        else:
            perf_weights = {k: equal_weight for k in keys}

        # Blend: alpha * performance_weights + (1-alpha) * equal_weights
        weights = {}
        for key in keys:
            w = alpha * perf_weights.get(key, equal_weight) + (1 - alpha) * equal_weight
            weights[key] = max(WEIGHT_FLOOR, w)

        # Re-normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def _compute_dynamic_threshold(
        self,
        regime: str,
        regime_confidence: float,
        time_of_day_boost: float,
        transition_velocity: float,
        trend_efficiency: Optional[float] = None,
        is_transition: bool = False,
    ) -> float:
        """
        Dynamic threshold based on context.

        Base threshold adjusted by:
          - Regime confidence (lower confidence → higher threshold)
          - Time of day (midday → higher threshold)
          - Transition velocity (unstable regime → higher threshold)
          - Recent regime performance (losing → higher threshold)
          - Transition/noise state (macro/micro divergence → higher threshold)
          - Low trend efficiency in TRENDING (TE < 0.15 → higher threshold)
        """
        threshold = self._base_threshold

        # Regime uncertainty penalty: +0 to +5 (reduced from 15x to 8x)
        # Previous 15x multiplier pushed threshold too high with typical
        # confidence values of 0.4-0.6, making trades nearly impossible.
        uncertainty_penalty = max(0.0, (1.0 - regime_confidence) * 8.0)
        threshold += uncertainty_penalty

        # Time of day
        threshold += time_of_day_boost

        # Transition instability: +0 to +4 (reduced from 8x to 4x)
        threshold += transition_velocity * 4.0

        # Transition/noise state (macro/micro divergence): +4 (reduced from +8)
        if is_transition:
            threshold += 4.0

        # Low trend efficiency in TRENDING regime: +3 (reduced from +5)
        if regime == "TRENDING" and trend_efficiency is not None and trend_efficiency < 0.15:
            threshold += 3.0

        # Recent regime performance penalty (reduced)
        rp = self._regime_performance.get(regime)
        if rp and rp.trades >= 10:
            if rp.win_rate < 0.40:
                threshold += 5.0  # Losing regime → raise bar
            elif rp.win_rate < 0.45:
                threshold += 3.0

        return min(threshold, 75.0)  # Hard cap (reduced from 90)

    def get_stats(self) -> Dict[str, Any]:
        """Get combiner statistics for observability."""
        stats = {}
        for (stype, sname, regime), perf in self._performance.items():
            key = f"{stype}:{sname}@{regime}"
            stats[key] = {
                'trades': perf.trades,
                'win_rate': round(perf.win_rate, 3),
                'sharpe': round(perf.sharpe, 3),
                'expectancy': round(perf.expectancy, 3),
            }
        return stats

    def reset(self):
        """Reset for new context."""
        self._performance.clear()
        self._regime_performance.clear()
