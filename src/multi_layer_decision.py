"""
Multi-Layer Decision Engine.

Layer 1: Candlestick Pattern Detection (entry signal generation)
Layer 2: Strategy Confirmation (existing strategies confirm the pattern)
Layer 3: Combined scoring with threshold to trigger trades.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from .strategies.candlestick_patterns import (
    CandlestickPatternDetector,
    DetectedPattern,
    PatternDirection,
)
from .strategies.base_strategy import BaseStrategy, Signal, SignalType, Regime

logger = logging.getLogger(__name__)


@dataclass
class DecisionResult:
    """Result of multi-layer evaluation."""
    execute: bool = False
    direction: Optional[str] = None  # 'bullish' or 'bearish'
    signal: Optional[Signal] = None
    patterns: List[DetectedPattern] = field(default_factory=list)
    primary_pattern: Optional[str] = None  # strongest directional pattern name
    confirming_signals: List[Signal] = field(default_factory=list)
    pattern_score: float = 0.0
    strategy_score: float = 0.0
    # combined_raw: weighted sum before normalization (can exceed 100)
    combined_raw: float = 0.0
    # combined_norm_0_100: normalized to 0-100 scale for comparison with threshold
    combined_norm_0_100: float = 0.0
    # combined_score: the value compared against threshold (= combined_norm_0_100)
    combined_score: float = 0.0
    threshold: float = 65.0
    pattern_threshold: float = 65.0
    trade_gate_threshold: float = 65.0
    threshold_used_reason: str = "base_threshold"
    # pattern_confirmed: True IFF directional patterns exist AND pattern_score >= pattern_threshold
    pattern_confirmation: bool = False
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'execute': self.execute,
            'direction': self.direction,
            'signal': self.signal.to_dict() if self.signal else None,
            'patterns': [p.to_dict() for p in self.patterns],
            'primary_pattern': self.primary_pattern,
            'confirming_signals': [s.to_dict() for s in self.confirming_signals],
            'pattern_score': round(self.pattern_score, 1),
            'strategy_score': round(self.strategy_score, 1),
            'combined_raw': round(self.combined_raw, 2),
            'combined_norm_0_100': round(self.combined_norm_0_100, 1),
            'combined_score': round(self.combined_score, 1),
            'threshold': self.threshold,
            'pattern_threshold': self.pattern_threshold,
            'trade_gate_threshold': self.trade_gate_threshold,
            'threshold_used_reason': self.threshold_used_reason,
            'pattern_confirmation': self.pattern_confirmation,
            'reasoning': self.reasoning,
        }


class MultiLayerDecision:
    """
    Orchestrates multi-layer trade decision making.

    Layer 1 (Candlestick Patterns): Generates entry signal candidates.
    Layer 2 (Strategy Confirmation): Existing strategies confirm the direction.
    Layer 3 (Scoring): Weighted combination decides whether to trade.
    """

    def __init__(
        self,
        pattern_weight: float = 0.4,
        strategy_weight: float = 0.6,
        threshold: float = 65.0,
        require_pattern: bool = True,
        strategy_only_threshold: float = 0.0,
    ):
        """
        Args:
            pattern_weight: Weight for candlestick pattern score (0-1).
            strategy_weight: Weight for strategy confirmation score (0-1).
            threshold: Minimum combined score to execute a trade (0-100).
            require_pattern: If True, a pattern must be detected before strategies
                            are even consulted. If False, strategies can still
                            generate signals on their own (backward compatible).
            strategy_only_threshold: Higher threshold applied when no candlestick
                            pattern confirms the signal (strategy-only mode).
                            If 0, falls back to the base threshold.
                            Filters borderline strategy-only entries.
        """
        self.pattern_detector = CandlestickPatternDetector()
        self.pattern_weight = pattern_weight
        self.strategy_weight = strategy_weight
        self.threshold = threshold
        self.require_pattern = require_pattern
        self.strategy_only_threshold = strategy_only_threshold

    def evaluate(
        self,
        ohlcv: Dict[str, List[float]],
        indicators: Dict[str, Any],
        regime: Regime,
        strategies: Dict[str, BaseStrategy],
        active_strategy_names: List[str],
        current_price: float,
        timestamp: datetime,
        ticker: str = "",
        generate_signal_fn=None,
        is_long_only: bool = False,
    ) -> DecisionResult:
        """
        Run multi-layer evaluation.

        Returns:
            DecisionResult with execution decision and full reasoning chain.
        """
        result = DecisionResult(
            threshold=self.threshold,
            pattern_threshold=self.threshold,
            trade_gate_threshold=(
                self.strategy_only_threshold if self.strategy_only_threshold > 0
                else self.threshold
            ),
        )

        # ── Layer 1: Candlestick Pattern Detection ──────────────────
        patterns = self.pattern_detector.detect(ohlcv, indicators)
        result.patterns = patterns

        if not patterns:
            if self.require_pattern:
                result.reasoning = "No candlestick pattern detected"
                return result
            else:
                # Fall through to strategy-only mode
                pass

        # Determine dominant direction from patterns
        bullish_patterns = [p for p in patterns if p.direction == PatternDirection.BULLISH]
        bearish_patterns = [p for p in patterns if p.direction == PatternDirection.BEARISH]

        if bullish_patterns and bearish_patterns:
            # Conflicting patterns - use the stronger set
            bull_avg = sum(p.strength for p in bullish_patterns) / len(bullish_patterns)
            bear_avg = sum(p.strength for p in bearish_patterns) / len(bearish_patterns)
            if bull_avg >= bear_avg:
                dominant_direction = PatternDirection.BULLISH
                dominant_patterns = bullish_patterns
                result.pattern_score = bull_avg
            else:
                dominant_direction = PatternDirection.BEARISH
                dominant_patterns = bearish_patterns
                result.pattern_score = bear_avg
        elif bullish_patterns:
            dominant_direction = PatternDirection.BULLISH
            dominant_patterns = bullish_patterns
            result.pattern_score = max(p.strength for p in bullish_patterns)
        elif bearish_patterns:
            dominant_direction = PatternDirection.BEARISH
            dominant_patterns = bearish_patterns
            result.pattern_score = max(p.strength for p in bearish_patterns)
        else:
            # Only neutral patterns (doji) - not enough for entry
            if self.require_pattern:
                result.reasoning = "Only neutral patterns detected (Doji), no directional bias"
                return result
            dominant_direction = None
            dominant_patterns = []
            result.pattern_score = 0

        result.direction = dominant_direction.value if dominant_direction else None

        # Identify primary (strongest) pattern for logging clarity
        if dominant_patterns:
            primary = max(dominant_patterns, key=lambda p: p.strength)
            result.primary_pattern = primary.name

        # Long-only filter
        if is_long_only and dominant_direction == PatternDirection.BEARISH:
            result.reasoning = "Bearish pattern filtered (long-only mode)"
            return result

        # ── Layer 2: Strategy Confirmation ──────────────────────────
        confirming_signals: List[Signal] = []

        if generate_signal_fn is not None:
            # Use the existing signal generation logic
            signal = generate_signal_fn()
            if signal:
                # Check direction alignment
                signal_direction = (
                    PatternDirection.BULLISH if signal.signal_type == SignalType.BUY
                    else PatternDirection.BEARISH
                )
                if dominant_direction is None or signal_direction == dominant_direction:
                    confirming_signals.append(signal)
                    result.strategy_score = signal.confidence
        else:
            # Generate signals from each active strategy directly
            for strat_name in active_strategy_names:
                strategy = strategies.get(strat_name)
                if not strategy:
                    continue
                signal = strategy.generate_signal(
                    current_price=current_price,
                    ohlcv=ohlcv,
                    indicators=indicators,
                    regime=regime,
                    timestamp=timestamp,
                )
                if signal:
                    sig_direction = (
                        PatternDirection.BULLISH if signal.signal_type == SignalType.BUY
                        else PatternDirection.BEARISH
                    )
                    if dominant_direction is None or sig_direction == dominant_direction:
                        confirming_signals.append(signal)

            if confirming_signals:
                result.strategy_score = max(s.confidence for s in confirming_signals)

        result.confirming_signals = confirming_signals

        # ── Layer 3: Combined Scoring ───────────────────────────────
        # Compute weighted raw score and normalize to 0-100.
        #
        # Decomposition (logged in reasoning):
        #   pattern_component = pattern_weight * pattern_score
        #   strategy_component = strategy_weight * strategy_score  (or 1.0 * score in strategy-only)
        #   combined_raw = sum of active components
        #   max_possible_raw = sum of (weight * 100) for active layers
        #   combined_norm = (combined_raw / max_possible_raw) * 100, clamped to [0, 100]
        #   combined_score = combined_norm  (this is what gets compared to threshold)

        has_patterns = bool(dominant_patterns and result.pattern_score > 0)
        has_strategy = bool(confirming_signals and result.strategy_score > 0)

        pattern_component = 0.0
        strategy_component = 0.0
        max_possible_raw = 0.0

        if has_patterns and has_strategy:
            pattern_component = self.pattern_weight * result.pattern_score
            strategy_component = self.strategy_weight * result.strategy_score
            max_possible_raw = (self.pattern_weight + self.strategy_weight) * 100.0
        elif has_patterns and not has_strategy:
            pattern_component = self.pattern_weight * result.pattern_score
            max_possible_raw = self.pattern_weight * 100.0
        elif not has_patterns and has_strategy:
            # Strategy-only mode: use raw strategy confidence (weight=1.0)
            strategy_component = result.strategy_score
            max_possible_raw = 100.0

        result.combined_raw = pattern_component + strategy_component

        if max_possible_raw > 0:
            result.combined_norm_0_100 = max(0.0, min(100.0,
                (result.combined_raw / max_possible_raw) * 100.0
            ))
        else:
            result.combined_norm_0_100 = 0.0

        # The score compared against threshold is the normalized score
        result.combined_score = result.combined_norm_0_100

        # ── Pattern confirmation (strict boolean) ──────────────────
        # A pattern "confirms" IFF:
        #   1) directional patterns exist (not just neutral Doji)
        #   2) pattern_score >= base threshold
        # This determines which threshold gate to use.
        result.pattern_confirmation = (
            has_patterns
            and result.pattern_score >= self.threshold
        )

        if result.pattern_confirmation:
            effective_threshold = self.threshold
            result.threshold_used_reason = "pattern_confirmation"
        elif self.strategy_only_threshold > 0:
            effective_threshold = self.strategy_only_threshold
            result.threshold_used_reason = "no_pattern_confirmation"
        else:
            effective_threshold = self.threshold
            result.threshold_used_reason = "base_threshold"

        result.threshold = effective_threshold
        result.pattern_threshold = self.threshold
        result.trade_gate_threshold = (
            self.strategy_only_threshold if self.strategy_only_threshold > 0
            else self.threshold
        )

        # ── Build reasoning ────────────────────────────────────────
        all_pattern_names = [p.name for p in patterns] if patterns else []
        strat_names = [s.strategy_name for s in confirming_signals]

        reasoning_parts = []

        # Show all patterns with primary highlighted
        if all_pattern_names:
            if result.primary_pattern and len(all_pattern_names) > 1:
                reasoning_parts.append(
                    f"Patterns: [{', '.join(all_pattern_names)}] primary={result.primary_pattern} "
                    f"(score: {result.pattern_score:.0f})"
                )
            else:
                reasoning_parts.append(
                    f"Pattern: {', '.join(all_pattern_names)} (score: {result.pattern_score:.0f})"
                )

        if strat_names:
            reasoning_parts.append(
                f"Strategy: {', '.join(strat_names)} (score: {result.strategy_score:.0f})"
            )

        # Decomposed combined score
        component_parts = []
        if pattern_component > 0:
            component_parts.append(f"pattern={pattern_component:.1f} (w={self.pattern_weight})")
        if strategy_component > 0:
            sw = self.strategy_weight if has_patterns else 1.0
            component_parts.append(f"strategy={strategy_component:.1f} (w={sw})")
        if component_parts:
            reasoning_parts.append(
                f"Combined: {' + '.join(component_parts)} => "
                f"raw={result.combined_raw:.1f} | norm={result.combined_norm_0_100:.1f}/100"
            )

        # Threshold info
        reasoning_parts.append(
            f"Threshold: {effective_threshold} "
            f"(reason={result.threshold_used_reason}, "
            f"pattern_confirmed={result.pattern_confirmation})"
        )

        # ── Decision ────────────────────────────────────────────────
        if result.combined_score >= effective_threshold:
            result.execute = True
            # Pick the best confirming signal to use for the trade
            if confirming_signals:
                best_signal = max(confirming_signals, key=lambda s: s.confidence)
                # Enrich signal metadata with pattern info
                best_signal.metadata['patterns'] = [p.to_dict() for p in patterns]
                best_signal.metadata['layer_scores'] = self._build_layer_scores_dict(result)
                best_signal.reasoning = (
                    f"[ML] {' | '.join(reasoning_parts)} | {best_signal.reasoning}"
                )
                result.signal = best_signal
            elif dominant_direction and dominant_patterns:
                # Patterns strong enough on their own (no strategy confirmation needed)
                best_pattern = max(dominant_patterns, key=lambda p: p.strength)
                atr_list = indicators.get('atr', [])
                atr_val = atr_list[-1] if atr_list else current_price * 0.005

                if dominant_direction == PatternDirection.BULLISH:
                    sig_type = SignalType.BUY
                    stop_loss = current_price - atr_val * 2
                    take_profit = current_price + atr_val * 3
                else:
                    sig_type = SignalType.SELL
                    stop_loss = current_price + atr_val * 2
                    take_profit = current_price - atr_val * 3

                result.signal = Signal(
                    strategy_name=f"Pattern:{best_pattern.name}",
                    signal_type=sig_type,
                    price=current_price,
                    timestamp=timestamp,
                    confidence=result.combined_score,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    trailing_stop=True,
                    trailing_stop_pct=0.8,
                    reasoning=f"[ML] {' | '.join(reasoning_parts)}",
                    metadata={
                        'patterns': [p.to_dict() for p in patterns],
                        'layer_scores': self._build_layer_scores_dict(result),
                    },
                )
            reasoning_parts.append("EXECUTE")
        else:
            reasoning_parts.append("SKIP (below threshold)")

        result.reasoning = " | ".join(reasoning_parts)
        return result

    def _build_layer_scores_dict(self, result: DecisionResult) -> Dict[str, Any]:
        """Build a consistent layer_scores metadata dict from DecisionResult."""
        return {
            'pattern_score': round(result.pattern_score, 1),
            'strategy_score': round(result.strategy_score, 1),
            'combined_raw': round(result.combined_raw, 2),
            'combined_norm_0_100': round(result.combined_norm_0_100, 1),
            'combined_score': round(result.combined_score, 1),
            'threshold': result.threshold,
            'pattern_threshold': result.pattern_threshold,
            'trade_gate_threshold': result.trade_gate_threshold,
            'threshold_used': result.threshold,
            'threshold_used_reason': result.threshold_used_reason,
            'pattern_confirmation': result.pattern_confirmation,
            'primary_pattern': result.primary_pattern,
            'pattern_direction': result.direction,
            'pattern_weight': self.pattern_weight,
            'strategy_weight': self.strategy_weight,
        }
