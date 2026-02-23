"""
Evidence-Based Decision Engine.

Evolves MultiLayerDecision from pattern-gated to multi-evidence approach.
No single evidence source can block entry alone; minimum 2 confirming
sources required.

Uses:
  - FeatureVector (normalized indicators)
  - RegimeState (probabilistic regime)
  - ConfidenceCalibrator (raw → P(profitable))
  - AdaptiveWeightCombiner (adaptive signal weights)
  - EdgeMonitor (strategy health)
  - CrossAssetState (index context)

Maintains backward compatibility with DecisionResult for frontend/tracker.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

from .feature_store import FeatureVector
from .adaptive_regime import RegimeState, TRENDING, CHOPPY, MIXED
from .confidence_calibrator import ConfidenceCalibrator
from .ensemble_combiner import (
    AdaptiveWeightCombiner, CalibratedSignal, EnsembleScore,
)
from .edge_monitor import EdgeMonitor, EdgeStatus, RecommendedAction
from .cross_asset import CrossAssetState
from .multi_layer_decision import DecisionResult
from .strategies.base_strategy import BaseStrategy, Signal, SignalType, Regime

logger = logging.getLogger(__name__)


@dataclass
class EvidenceSource:
    """A single piece of evidence for/against a trade."""
    source_type: str       # 'strategy', 'feature', 'l2_flow', 'cross_asset', 'regime'
    source_name: str       # e.g. 'hammer', 'momentum_flow', 'rsi_extreme'
    direction: str         # 'bullish', 'bearish', 'neutral'
    strength: float        # 0-100 raw strength
    calibrated: float      # 0-1 P(profitable) after calibration
    reasoning: str = ""


class EvidenceDecisionEngine:
    """
    Multi-evidence decision engine.

    Evidence sources:
      1. Strategy signals (existing 10 strategies)
      2. Feature signals (normalized indicator extremes)
      3. L2 flow signals (order flow alignment)
      4. Cross-asset signals (index/sector context)
      5. Regime alignment (regime probability support)

    Decision logic:
      - Collect evidence from all sources
      - Calibrate each source's confidence
      - Combine with adaptive weights
      - Require minimum 2 confirming sources
      - Apply dynamic threshold
      - Execute only when an aligned strategy signal exists
    """

    def __init__(
        self,
        calibrator: Optional[ConfidenceCalibrator] = None,
        combiner: Optional[AdaptiveWeightCombiner] = None,
        edge_monitor: Optional[EdgeMonitor] = None,
        min_confirming_sources: int = 2,
        base_threshold: float = 55.0,
        strategy_weight: float = 0.6,
        strategy_only_threshold: float = 0.0,
    ):
        self.calibrator = calibrator or ConfidenceCalibrator()
        self.combiner = combiner or AdaptiveWeightCombiner(
            min_confirming_sources=min_confirming_sources,
            base_threshold=base_threshold,
        )
        self.edge_monitor = edge_monitor or EdgeMonitor()
        self._min_confirming = min_confirming_sources
        self._base_threshold = base_threshold
        # Legacy fields kept for schema compatibility.
        self._strategy_weight = 1.0
        self._strategy_only_threshold = strategy_only_threshold

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
        # New evidence-based params
        feature_vector: Optional[FeatureVector] = None,
        regime_state: Optional[RegimeState] = None,
        cross_asset_state: Optional[CrossAssetState] = None,
        time_of_day_boost: float = 0.0,
    ) -> DecisionResult:
        """
        Multi-evidence evaluation.

        Maintains DecisionResult interface for backward compatibility.
        Internally uses evidence collection → calibration → ensemble → decision.
        """
        result = DecisionResult(
            threshold=self._base_threshold,
            trade_gate_threshold=self._base_threshold,
        )

        evidence_sources: List[EvidenceSource] = []
        calibrated_signals: List[CalibratedSignal] = []

        # ── Evidence 1: Strategy Signals ──────────────────────────────
        confirming_signals: List[Signal] = []

        if generate_signal_fn is not None:
            signal = generate_signal_fn()
            if signal:
                confirming_signals.append(signal)
        else:
            for strat_name in active_strategy_names:
                strategy = strategies.get(strat_name)
                if not strategy:
                    continue
                # Check edge health before running strategy
                regime_label = regime_state.primary if regime_state else regime.value
                health = self.edge_monitor.get_health(strat_name, regime_label)
                if health.action == RecommendedAction.PAUSE:
                    continue  # Skip paused strategies

                signal = strategy.generate_signal(
                    current_price=current_price,
                    ohlcv=ohlcv,
                    indicators=indicators,
                    regime=regime,
                    timestamp=timestamp,
                )
                if signal:
                    confirming_signals.append(signal)

        for signal in confirming_signals:
            sig_dir = 'bullish' if signal.signal_type == SignalType.BUY else 'bearish'
            regime_label = regime_state.primary if regime_state else regime.value

            cal_conf = self.calibrator.calibrate(
                signal.strategy_name, signal.confidence, regime_label,
            )

            evidence_sources.append(EvidenceSource(
                source_type='strategy',
                source_name=signal.strategy_name,
                direction=sig_dir,
                strength=signal.confidence,
                calibrated=cal_conf,
                reasoning=signal.reasoning,
            ))

            calibrated_signals.append(CalibratedSignal(
                source_type='strategy',
                source_name=signal.strategy_name,
                direction=sig_dir,
                raw_confidence=signal.confidence,
                calibrated_confidence=cal_conf,
                metadata={
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'trailing_stop': signal.trailing_stop,
                    'trailing_stop_pct': signal.trailing_stop_pct,
                },
            ))

        result.confirming_signals = confirming_signals
        if confirming_signals:
            result.strategy_score = max(s.confidence for s in confirming_signals)

        # ── Evidence 2: Feature-Based Signals ─────────────────────────
        if feature_vector is not None:
            feat_evidence = self._extract_feature_evidence(feature_vector)
            for ev in feat_evidence:
                evidence_sources.append(ev)
                calibrated_signals.append(CalibratedSignal(
                    source_type=ev.source_type,
                    source_name=ev.source_name,
                    direction=ev.direction,
                    raw_confidence=ev.strength,
                    calibrated_confidence=ev.calibrated,
                ))

        # ── Evidence 3: L2 Flow Evidence ──────────────────────────────
        if feature_vector is not None and feature_vector.l2_has_coverage:
            l2_evidence = self._extract_l2_evidence(feature_vector)
            for ev in l2_evidence:
                evidence_sources.append(ev)
                calibrated_signals.append(CalibratedSignal(
                    source_type=ev.source_type,
                    source_name=ev.source_name,
                    direction=ev.direction,
                    raw_confidence=ev.strength,
                    calibrated_confidence=ev.calibrated,
                ))

        # ── Evidence 4: Cross-Asset Evidence ──────────────────────────
        if cross_asset_state is not None and cross_asset_state.index_available:
            ca_evidence = self._extract_cross_asset_evidence(cross_asset_state)
            if ca_evidence:
                evidence_sources.append(ca_evidence)
                calibrated_signals.append(CalibratedSignal(
                    source_type=ca_evidence.source_type,
                    source_name=ca_evidence.source_name,
                    direction=ca_evidence.direction,
                    raw_confidence=ca_evidence.strength,
                    calibrated_confidence=ca_evidence.calibrated,
                ))

        # ── Evidence 5: Regime Alignment ──────────────────────────────
        if regime_state is not None:
            regime_evidence = self._extract_regime_evidence(regime_state)
            if regime_evidence:
                evidence_sources.append(regime_evidence)
                calibrated_signals.append(CalibratedSignal(
                    source_type=regime_evidence.source_type,
                    source_name=regime_evidence.source_name,
                    direction=regime_evidence.direction,
                    raw_confidence=regime_evidence.strength,
                    calibrated_confidence=regime_evidence.calibrated,
                ))

        # ── Long-Only Filter ─────────────────────────────────────────
        if is_long_only:
            calibrated_signals = [
                s for s in calibrated_signals if s.direction != 'bearish'
            ]

        # ── Ensemble Combination ──────────────────────────────────────
        if not calibrated_signals:
            result.reasoning = "No evidence signals generated"
            return result

        regime_conf = regime_state.confidence if regime_state else 0.5
        transition_vel = regime_state.transition_velocity if regime_state else 0.0
        is_trans = regime_state.is_transition if regime_state else False
        regime_age_bars = regime_state.bar_index if regime_state else None
        te = feature_vector.trend_efficiency if feature_vector else None

        ensemble = self.combiner.combine(
            signals=calibrated_signals,
            regime=regime_state.primary if regime_state else regime.value,
            regime_confidence=regime_conf,
            time_of_day_boost=time_of_day_boost,
            transition_velocity=transition_vel,
            trend_efficiency=te,
            is_transition=is_trans,
            regime_age_bars=regime_age_bars,
        )

        # ── Map to DecisionResult ─────────────────────────────────────
        result.direction = ensemble.direction
        result.combined_score = ensemble.combined_score
        result.combined_norm_0_100 = ensemble.combined_score
        result.combined_raw = ensemble.combined_score  # Simplified for compatibility
        result.threshold = ensemble.threshold_used
        te_str = f"{te:.2f}" if te is not None else "N/A"
        result.threshold_used_reason = (
            f"dynamic(regime_conf={regime_conf:.2f}, "
            f"tod_boost={time_of_day_boost}, "
            f"transition_vel={transition_vel:.2f}, "
            f"is_trans={is_trans}, te={te_str})"
        )

        # ── Execute Decision ──────────────────────────────────────────
        no_aligned_strategy_signal = False
        if ensemble.execute:
            result.execute = True
            # Pick the best strategy signal for trade execution
            strategy_signals = [
                s for s in confirming_signals
                if (s.signal_type == SignalType.BUY and ensemble.direction == 'bullish')
                or (s.signal_type == SignalType.SELL and ensemble.direction == 'bearish')
            ]

            if strategy_signals:
                best_signal = max(strategy_signals, key=lambda s: s.confidence)
                if isinstance(best_signal.metadata, dict):
                    best_signal.metadata.pop('patterns', None)
                best_signal.metadata['layer_scores'] = self._build_layer_scores(
                    result, ensemble,
                )
                best_signal.metadata['evidence_sources'] = [
                    {'type': e.source_type, 'name': e.source_name,
                     'direction': e.direction, 'strength': round(e.strength, 1),
                     'calibrated': round(e.calibrated, 3)}
                    for e in evidence_sources
                ]
                best_signal.metadata['ensemble'] = ensemble.to_dict()
                best_signal.reasoning = (
                    f"[EV] {ensemble.reasoning} | {best_signal.reasoning}"
                )
                result.signal = best_signal
            else:
                # Evidence by itself cannot open positions; execution needs a
                # strategy-produced signal aligned with ensemble direction.
                result.execute = False
                no_aligned_strategy_signal = True

        # ── Reasoning ─────────────────────────────────────────────────
        evidence_summary = ", ".join(
            f"{e.source_type}:{e.source_name}({e.direction[:1]},{e.strength:.0f}→{e.calibrated:.2f})"
            for e in evidence_sources
        )
        result.reasoning = (
            f"Evidence: [{evidence_summary}] | "
            f"Ensemble: score={ensemble.combined_score:.1f} "
            f"thresh={ensemble.threshold_used:.1f} "
            f"confirming={ensemble.confirming_sources}/{ensemble.total_sources} | "
            f"{'EXECUTE' if result.execute else 'SKIP'}"
        )
        if no_aligned_strategy_signal:
            result.reasoning += " | no aligned strategy signal"

        return result

    # ──────────────────────────────────────────────────────────────────
    # Evidence extraction helpers
    # ──────────────────────────────────────────────────────────────────

    def _extract_feature_evidence(
        self, fv: FeatureVector,
    ) -> List[EvidenceSource]:
        """Extract directional evidence from normalized features."""
        evidence = []

        # RSI extreme: z-score > 1.0 (lowered from 1.5 for OHLCV-only environments
        # where fewer evidence sources are available)
        if fv.rsi_z < -1.0:
            evidence.append(EvidenceSource(
                source_type='feature', source_name='rsi_oversold',
                direction='bullish', strength=min(80, 50 + abs(fv.rsi_z) * 12),
                calibrated=min(0.7, 0.40 + abs(fv.rsi_z) * 0.10),
                reasoning=f"RSI z-score={fv.rsi_z:.1f} (oversold)",
            ))
        elif fv.rsi_z > 1.0:
            evidence.append(EvidenceSource(
                source_type='feature', source_name='rsi_overbought',
                direction='bearish', strength=min(80, 50 + abs(fv.rsi_z) * 12),
                calibrated=min(0.7, 0.40 + abs(fv.rsi_z) * 0.10),
                reasoning=f"RSI z-score={fv.rsi_z:.1f} (overbought)",
            ))

        # Strong momentum: z-score > 0.8 (lowered from 1.2)
        if fv.momentum_z > 0.8:
            evidence.append(EvidenceSource(
                source_type='feature', source_name='momentum_strong',
                direction='bullish', strength=min(75, 50 + fv.momentum_z * 10),
                calibrated=min(0.65, 0.38 + fv.momentum_z * 0.09),
                reasoning=f"Momentum z={fv.momentum_z:.1f}",
            ))
        elif fv.momentum_z < -0.8:
            evidence.append(EvidenceSource(
                source_type='feature', source_name='momentum_strong',
                direction='bearish', strength=min(75, 50 + abs(fv.momentum_z) * 10),
                calibrated=min(0.65, 0.38 + abs(fv.momentum_z) * 0.09),
                reasoning=f"Momentum z={fv.momentum_z:.1f}",
            ))

        # VWAP deviation (lowered from 2.0 to 1.5)
        if fv.vwap_dist_z < -1.5:
            evidence.append(EvidenceSource(
                source_type='feature', source_name='vwap_below',
                direction='bullish', strength=min(70, 45 + abs(fv.vwap_dist_z) * 8),
                calibrated=min(0.6, 0.35 + abs(fv.vwap_dist_z) * 0.06),
                reasoning=f"VWAP distance z={fv.vwap_dist_z:.1f}",
            ))
        elif fv.vwap_dist_z > 1.5:
            evidence.append(EvidenceSource(
                source_type='feature', source_name='vwap_above',
                direction='bearish', strength=min(70, 45 + abs(fv.vwap_dist_z) * 8),
                calibrated=min(0.6, 0.35 + abs(fv.vwap_dist_z) * 0.06),
                reasoning=f"VWAP distance z={fv.vwap_dist_z:.1f}",
            ))

        # Volume spike with direction (lowered from z>2.0/roc>0.2 to z>1.5/roc>0.15)
        if fv.volume_z > 1.5 and abs(fv.roc_5) > 0.15:
            vol_dir = 'bullish' if fv.roc_5 > 0 else 'bearish'
            evidence.append(EvidenceSource(
                source_type='feature', source_name='volume_spike',
                direction=vol_dir, strength=min(70, 45 + fv.volume_z * 6),
                calibrated=min(0.6, 0.35 + fv.volume_z * 0.05),
                reasoning=f"Volume z={fv.volume_z:.1f} with {vol_dir} price",
            ))

        return evidence

    def _extract_l2_evidence(
        self, fv: FeatureVector,
    ) -> List[EvidenceSource]:
        """Extract evidence from L2 order flow.

        Uses z-score-based thresholds when available (adapts to current market
        conditions automatically). Falls back to static thresholds when z-scores
        are not warmed up (value == 0.0 with non-zero raw value).
        """
        evidence = []

        # Directional aggression: use z-score threshold when available
        agg_z = abs(fv.l2_aggression_z)
        agg_raw = abs(fv.l2_signed_aggression)
        # Z-score available and warmed up: use adaptive threshold
        agg_fires = (agg_z > 1.0) if (agg_z > 0 or agg_raw == 0) else (agg_raw > 0.06)
        if agg_fires:
            flow_dir = 'bullish' if fv.l2_signed_aggression > 0 else 'bearish'
            agg_strength = min(80, 50 + agg_z * 10)
            evidence.append(EvidenceSource(
                source_type='l2_flow', source_name='aggression',
                direction=flow_dir, strength=agg_strength,
                calibrated=min(0.7, 0.4 + agg_z * 0.08),
                reasoning=f"L2 aggression={fv.l2_signed_aggression:.3f} z={fv.l2_aggression_z:.1f}",
            ))

        # Delta divergence: use z-score threshold when available
        div_z = abs(getattr(fv, 'l2_divergence_z', 0.0) or 0.0)
        div_raw = abs(fv.l2_delta_price_divergence)
        div_fires = (div_z > 1.2) if (div_z > 0 or div_raw == 0) else (div_raw > 0.5)
        if div_fires:
            div_dir = 'bullish' if fv.l2_delta_price_divergence > 0 else 'bearish'
            div_strength = min(75, 45 + div_raw * 12)
            evidence.append(EvidenceSource(
                source_type='l2_flow', source_name='delta_divergence',
                direction=div_dir, strength=div_strength,
                calibrated=min(0.65, 0.35 + div_raw * 0.06),
                reasoning=f"L2 delta-price divergence={fv.l2_delta_price_divergence:.2f} z={div_z:.1f}",
            ))

        # Absorption (price absorbed, flow neutral → potential reversal)
        if fv.l2_absorption_rate > 0.45 and abs(fv.l2_book_pressure) > 0.08:
            abs_dir = 'bullish' if fv.l2_book_pressure < 0 else 'bearish'
            evidence.append(EvidenceSource(
                source_type='l2_flow', source_name='absorption',
                direction=abs_dir,
                strength=min(70, 45 + fv.l2_absorption_rate * 30),
                calibrated=min(0.6, 0.35 + fv.l2_absorption_rate * 0.15),
                reasoning=f"L2 absorption={fv.l2_absorption_rate:.2f} pressure={fv.l2_book_pressure:.3f}",
            ))

        return evidence

    def _extract_cross_asset_evidence(
        self, ca: CrossAssetState,
    ) -> Optional[EvidenceSource]:
        """Extract evidence from cross-asset context."""
        if not ca.index_available:
            return None

        # Only generate evidence when there's a meaningful signal
        if abs(ca.index_momentum_5) < 0.1:
            return None

        idx_dir = 'bullish' if ca.index_momentum_5 > 0 else 'bearish'

        # Strength based on correlation-weighted index momentum
        strength = min(65, 40 + abs(ca.index_momentum_5) * 8 * max(0, ca.correlation_20))
        cal = min(0.6, 0.35 + abs(ca.index_momentum_5) * 0.04 * max(0, ca.correlation_20))

        return EvidenceSource(
            source_type='cross_asset', source_name='index_context',
            direction=idx_dir, strength=strength, calibrated=cal,
            reasoning=f"Index mom={ca.index_momentum_5:.2f}% corr={ca.correlation_20:.2f}",
        )

    def _extract_regime_evidence(
        self, rs: RegimeState,
    ) -> Optional[EvidenceSource]:
        """
        Regime can provide directional bias when confident.
        TRENDING_UP/DOWN micro-regimes carry direction.
        """
        if rs.confidence < 0.55:
            return None

        micro = rs.micro_regime
        if micro in ('TRENDING_UP', 'BREAKOUT'):
            return EvidenceSource(
                source_type='regime', source_name='regime_direction',
                direction='bullish', strength=min(65, rs.confidence * 80),
                calibrated=min(0.6, rs.confidence * 0.65),
                reasoning=f"Regime {micro} conf={rs.confidence:.2f}",
            )
        elif micro == 'TRENDING_DOWN':
            return EvidenceSource(
                source_type='regime', source_name='regime_direction',
                direction='bearish', strength=min(65, rs.confidence * 80),
                calibrated=min(0.6, rs.confidence * 0.65),
                reasoning=f"Regime {micro} conf={rs.confidence:.2f}",
            )

        return None

    # ──────────────────────────────────────────────────────────────────
    # Legacy compatibility
    # ──────────────────────────────────────────────────────────────────

    def _build_layer_scores(
        self, result: DecisionResult, ensemble: EnsembleScore,
    ) -> Dict[str, Any]:
        """Build layer_scores dict for legacy compatibility."""
        return {
            'strategy_score': round(result.strategy_score, 1),
            'combined_raw': round(result.combined_raw, 2),
            'combined_norm_0_100': round(result.combined_norm_0_100, 1),
            'combined_score': round(result.combined_score, 1),
            'threshold': result.threshold,
            'threshold_used': result.threshold,
            'threshold_used_reason': result.threshold_used_reason,
            'strategy_weight': self._strategy_weight,
            # New fields
            'calibrated_probability': round(ensemble.calibrated_probability, 4),
            'confirming_sources': ensemble.confirming_sources,
            'total_sources': ensemble.total_sources,
            'source_weights': ensemble.source_weights,
            'engine': 'evidence_v1',
        }
