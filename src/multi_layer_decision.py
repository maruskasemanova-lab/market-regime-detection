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
    confirming_signals: List[Signal] = field(default_factory=list)
    pattern_score: float = 0.0
    strategy_score: float = 0.0
    combined_score: float = 0.0
    threshold: float = 65.0
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'execute': self.execute,
            'direction': self.direction,
            'signal': self.signal.to_dict() if self.signal else None,
            'patterns': [p.to_dict() for p in self.patterns],
            'confirming_signals': [s.to_dict() for s in self.confirming_signals],
            'pattern_score': round(self.pattern_score, 1),
            'strategy_score': round(self.strategy_score, 1),
            'combined_score': round(self.combined_score, 1),
            'threshold': self.threshold,
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
    ):
        """
        Args:
            pattern_weight: Weight for candlestick pattern score (0-1).
            strategy_weight: Weight for strategy confirmation score (0-1).
            threshold: Minimum combined score to execute a trade (0-100).
            require_pattern: If True, a pattern must be detected before strategies
                            are even consulted. If False, strategies can still
                            generate signals on their own (backward compatible).
        """
        self.pattern_detector = CandlestickPatternDetector()
        self.pattern_weight = pattern_weight
        self.strategy_weight = strategy_weight
        self.threshold = threshold
        self.require_pattern = require_pattern

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

        Args:
            ohlcv: OHLCV data arrays
            indicators: Calculated indicator values
            regime: Current market regime
            strategies: Dict of strategy_name -> BaseStrategy instances
            active_strategy_names: Names of strategies active for current regime
            current_price: Current bar close price
            timestamp: Current timestamp
            ticker: Ticker symbol (for logging)
            generate_signal_fn: Optional callback that returns the best Signal
                               from active strategies (reuses existing DayTradingManager
                               ranking logic). Signature: () -> Optional[Signal]
            is_long_only: Whether only long trades are allowed

        Returns:
            DecisionResult with execution decision and full reasoning chain.
        """
        result = DecisionResult(threshold=self.threshold)

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
        if patterns and confirming_signals:
            # Both layers contribute
            result.combined_score = (
                self.pattern_weight * result.pattern_score
                + self.strategy_weight * result.strategy_score
            )
        elif patterns and not confirming_signals:
            # Only patterns, no strategy confirmation
            result.combined_score = self.pattern_weight * result.pattern_score
        elif not patterns and confirming_signals:
            # Only strategy (require_pattern=False mode)
            # Use raw strategy confidence when pattern gate is disabled so
            # flow-first strategies are not artificially down-weighted.
            result.combined_score = result.strategy_score
        else:
            result.combined_score = 0

        # Build reasoning
        pattern_names = [p.name for p in dominant_patterns] if dominant_patterns else []
        strat_names = [s.strategy_name for s in confirming_signals]

        reasoning_parts = []
        if pattern_names:
            reasoning_parts.append(f"Patterns: {', '.join(pattern_names)} (score: {result.pattern_score:.0f})")
        if strat_names:
            reasoning_parts.append(f"Confirmed by: {', '.join(strat_names)} (score: {result.strategy_score:.0f})")
        reasoning_parts.append(f"Combined: {result.combined_score:.1f} vs threshold {self.threshold}")

        # ── Decision ────────────────────────────────────────────────
        if result.combined_score >= self.threshold:
            result.execute = True
            # Pick the best confirming signal to use for the trade
            if confirming_signals:
                best_signal = max(confirming_signals, key=lambda s: s.confidence)
                # Enrich signal metadata with pattern info
                best_signal.metadata['patterns'] = [p.to_dict() for p in patterns]
                best_signal.metadata['layer_scores'] = {
                    'pattern_score': round(result.pattern_score, 1),
                    'strategy_score': round(result.strategy_score, 1),
                    'combined_score': round(result.combined_score, 1),
                    'threshold': self.threshold,
                }
                best_signal.reasoning = (
                    f"[ML] {' | '.join(reasoning_parts)} | {best_signal.reasoning}"
                )
                result.signal = best_signal
            elif dominant_direction and dominant_patterns:
                # Patterns strong enough on their own (no strategy confirmation needed)
                # Create a signal from the pattern
                best_pattern = max(dominant_patterns, key=lambda p: p.strength)
                atr_list = indicators.get('atr', [])
                atr_val = atr_list[-1] if atr_list else current_price * 0.005
                vwap_list = indicators.get('vwap', [])
                vwap_val = vwap_list[-1] if isinstance(vwap_list, list) and vwap_list else current_price

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
                        'layer_scores': {
                            'pattern_score': round(result.pattern_score, 1),
                            'strategy_score': 0,
                            'combined_score': round(result.combined_score, 1),
                            'threshold': self.threshold,
                        },
                    },
                )
            reasoning_parts.append("EXECUTE")
        else:
            reasoning_parts.append("SKIP (below threshold)")

        result.reasoning = " | ".join(reasoning_parts)
        return result
