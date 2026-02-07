"""
Trading Orchestrator: Integrates all new modules into the existing pipeline.

This module bridges the new architecture (FeatureStore, AdaptiveRegimeDetector,
ConfidenceCalibrator, EnsembleCombiner, EdgeMonitor, CrossAssetContext,
QualityPositionSizer, EvidenceDecisionEngine) with the existing
DayTradingManager and session_runner flow.

Usage in DayTradingManager:
    self.orchestrator = TradingOrchestrator()

    # In process_bar:
    fv = self.orchestrator.update_bar(bar_dict)
    regime_state = self.orchestrator.detect_regime(fv)
    # ... use evidence_decision_engine for signal evaluation ...
    # After trade close:
    self.orchestrator.record_trade_outcome(...)
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .feature_store import FeatureStore, FeatureVector
from .adaptive_regime import AdaptiveRegimeDetector, RegimeState
from .confidence_calibrator import ConfidenceCalibrator
from .ensemble_combiner import AdaptiveWeightCombiner
from .edge_monitor import EdgeMonitor, EdgeHealth, EdgeStatus, RecommendedAction
from .cross_asset import CrossAssetContext, CrossAssetState
from .position_sizing import QualityPositionSizer, SizingResult
from .evidence_decision import EvidenceDecisionEngine

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    # Feature Store
    zscore_window: int = 100
    l2_zscore_window: int = 50
    percentile_window: int = 200

    # Regime Detection
    regime_smoothing_alpha: float = 0.3
    regime_transition_lookback: int = 10

    # Calibration
    calibration_lookback_trades: int = 50

    # Ensemble Combiner
    min_confirming_sources: int = 2
    base_threshold: float = 55.0

    # Position Sizing
    base_risk_pct: float = 1.0

    # Cross-Asset
    reference_tickers: List[str] = field(
        default_factory=lambda: ['QQQ']
    )

    # Legacy compatibility weights (used in DecisionResult)
    pattern_weight: float = 0.4
    strategy_weight: float = 0.6
    strategy_only_threshold: float = 0.0

    # Enable/disable new modules (for gradual rollout)
    use_evidence_engine: bool = True
    use_adaptive_regime: bool = True
    use_calibration: bool = True
    use_quality_sizing: bool = True
    use_cross_asset: bool = True
    use_edge_monitor: bool = True


class TradingOrchestrator:
    """
    Central orchestrator for the new trading architecture.

    Holds all new module instances and coordinates data flow between them.
    Designed to be attached to DayTradingManager as self.orchestrator.

    Lifecycle:
      1. __init__() or configure() → set up all modules
      2. new_session() → reset per-session state
      3. update_bar() → called per bar, returns FeatureVector
      4. detect_regime() → returns RegimeState
      5. update_cross_asset() → feed reference ticker bar
      6. get_evidence_engine() → get engine for evaluate()
      7. get_sizing() → get position sizing for a signal
      8. record_trade_outcome() → update calibration + edge monitor
      9. get_system_health() → observability
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()

        # Core modules
        self.feature_store = FeatureStore(
            zscore_window=self.config.zscore_window,
            l2_zscore_window=self.config.l2_zscore_window,
            percentile_window=self.config.percentile_window,
        )

        self.regime_detector = AdaptiveRegimeDetector(
            smoothing_alpha=self.config.regime_smoothing_alpha,
            transition_lookback=self.config.regime_transition_lookback,
        )

        self.calibrator = ConfidenceCalibrator(
            lookback_trades=self.config.calibration_lookback_trades,
        )

        self.combiner = AdaptiveWeightCombiner(
            min_confirming_sources=self.config.min_confirming_sources,
            base_threshold=self.config.base_threshold,
        )

        self.edge_monitor = EdgeMonitor()

        self.cross_asset = CrossAssetContext(
            reference_tickers=self.config.reference_tickers,
        )

        self.position_sizer = QualityPositionSizer(
            base_risk_pct=self.config.base_risk_pct,
            edge_monitor=self.edge_monitor,
        )

        self.evidence_engine = EvidenceDecisionEngine(
            calibrator=self.calibrator,
            combiner=self.combiner,
            edge_monitor=self.edge_monitor,
            min_confirming_sources=self.config.min_confirming_sources,
            base_threshold=self.config.base_threshold,
            pattern_weight=self.config.pattern_weight,
            strategy_weight=self.config.strategy_weight,
            strategy_only_threshold=self.config.strategy_only_threshold,
        )

        # State
        self._current_fv: Optional[FeatureVector] = None
        self._current_regime: Optional[RegimeState] = None
        self._current_cross_asset: Optional[CrossAssetState] = None
        self._bar_count = 0

    def configure(self, config: OrchestratorConfig):
        """Reconfigure orchestrator (e.g. from AOS config)."""
        self.__init__(config)

    def new_session(self):
        """Reset per-session state. Call at start of each trading day."""
        self.feature_store.reset()
        self.regime_detector.reset()
        self.cross_asset.reset()
        self._current_fv = None
        self._current_regime = None
        self._current_cross_asset = None
        self._bar_count = 0
        # Note: calibrator, combiner, edge_monitor persist across sessions
        # (they learn from historical trades)

    def reset_learning_state(self):
        """
        Reset learned state for isolated backtest reruns.

        Keeps orchestrator config intact, but clears components that accumulate
        trade outcomes across sessions.
        """
        self.calibrator.reset()
        self.combiner.reset()
        self.edge_monitor.reset()

    def full_reset(self):
        """
        Reset both session pipeline and learned state.

        Intended for deterministic backtest runs where each replay should start
        from a cold state.
        """
        self.reset_learning_state()
        self.new_session()

    # ──────────────────────────────────────────────────────────────────
    # Per-bar pipeline
    # ──────────────────────────────────────────────────────────────────

    def update_bar(self, bar: Dict[str, Any]) -> FeatureVector:
        """
        Process a new bar through the feature store.
        Returns normalized FeatureVector.
        """
        self._bar_count += 1
        self._current_fv = self.feature_store.update(bar)
        return self._current_fv

    def detect_regime(
        self, fv: Optional[FeatureVector] = None,
    ) -> RegimeState:
        """
        Detect regime from current feature vector.
        Returns probabilistic RegimeState.
        """
        fv = fv or self._current_fv
        if fv is None:
            return RegimeState()

        if self.config.use_adaptive_regime:
            self._current_regime = self.regime_detector.detect(fv)
        else:
            # Fallback: create RegimeState from legacy detection
            self._current_regime = RegimeState()

        return self._current_regime

    def update_cross_asset(
        self, ticker: str, bar: Dict[str, Any],
    ):
        """
        Feed a reference ticker bar (e.g. QQQ).
        Call this before update_target_cross_asset.
        """
        if self.config.use_cross_asset:
            self.cross_asset.update_reference(ticker, bar)

    def update_target_cross_asset(
        self, bar: Dict[str, Any],
    ) -> CrossAssetState:
        """
        Update cross-asset context for the target ticker.
        """
        if self.config.use_cross_asset:
            self._current_cross_asset = self.cross_asset.update_target(bar)
        else:
            self._current_cross_asset = CrossAssetState()
        return self._current_cross_asset

    def get_time_of_day_boost(self, bar_time) -> float:
        """
        Calculate time-of-day threshold boost.
        Midday quiet hours (10:30-14:00 ET) → higher threshold.
        """
        from datetime import time
        if isinstance(bar_time, str):
            return 0.0
        try:
            t = bar_time if isinstance(bar_time, time) else bar_time.time()
            if time(10, 30) <= t <= time(14, 0):
                return 5.0
        except (AttributeError, TypeError):
            pass
        return 0.0

    # ──────────────────────────────────────────────────────────────────
    # Decision support
    # ──────────────────────────────────────────────────────────────────

    def get_legacy_indicators(self, order_flow: Dict) -> Dict[str, Any]:
        """
        Convert current FeatureVector to legacy indicators dict
        for backward compatibility with existing strategies.
        """
        if self._current_fv is None:
            return {'order_flow': order_flow}
        return self.feature_store.to_legacy_indicators(self._current_fv, order_flow)

    def get_sizing(
        self,
        ensemble_score=None,
        strategy_name: str = "",
        regime_label: str = "MIXED",
    ) -> SizingResult:
        """
        Calculate position sizing based on signal quality.
        """
        if not self.config.use_quality_sizing:
            return SizingResult(final_risk_pct=self.config.base_risk_pct)

        return self.position_sizer.calculate_risk_pct(
            ensemble_score=ensemble_score,
            regime_state=self._current_regime,
            strategy_name=strategy_name,
            regime_label=regime_label,
        )

    def get_edge_health(
        self, strategy: str, regime: str,
    ) -> EdgeHealth:
        """Get edge health for a strategy."""
        return self.edge_monitor.get_health(strategy, regime)

    def should_skip_strategy(
        self, strategy: str, regime: str,
    ) -> bool:
        """Check if strategy should be skipped due to edge degradation."""
        if not self.config.use_edge_monitor:
            return False
        health = self.edge_monitor.get_health(strategy, regime)
        return health.action == RecommendedAction.PAUSE

    def get_threshold_adjustment(
        self, strategy: str, regime: str,
    ) -> float:
        """Get threshold adjustment for a strategy based on edge health."""
        if not self.config.use_edge_monitor:
            return 0.0
        health = self.edge_monitor.get_health(strategy, regime)
        return health.threshold_adjustment

    # ──────────────────────────────────────────────────────────────────
    # Trade outcome recording
    # ──────────────────────────────────────────────────────────────────

    def record_trade_outcome(
        self,
        strategy: str,
        regime: str,
        raw_confidence: float,
        was_profitable: bool,
        pnl_r: float = 0.0,
        bar_index: int = 0,
        confirming_sources: Optional[List[str]] = None,
    ):
        """
        Record a closed trade for calibration and edge monitoring.
        Call this after every trade close.
        """
        # Update calibrator
        if self.config.use_calibration:
            self.calibrator.update(
                strategy=strategy,
                raw_confidence=raw_confidence,
                regime=regime,
                was_profitable=was_profitable,
            )

        # Update edge monitor
        if self.config.use_edge_monitor:
            self.edge_monitor.update_trade(
                strategy=strategy,
                regime=regime,
                pnl_r=pnl_r,
                was_profitable=was_profitable,
                bar_index=bar_index,
                confidence=raw_confidence,
            )

        # Update ensemble combiner
        self.combiner.update_outcome(
            source_type='strategy',
            source_name=strategy,
            regime=regime,
            was_profitable=was_profitable,
            pnl_r=pnl_r,
        )

        # Also update for any confirming sources
        if confirming_sources:
            for source in confirming_sources:
                stype, sname = (source.split(':', 1) + [''])[:2]
                if stype and sname:
                    self.combiner.update_outcome(
                        source_type=stype,
                        source_name=sname,
                        regime=regime,
                        was_profitable=was_profitable,
                        pnl_r=pnl_r,
                    )

    # ──────────────────────────────────────────────────────────────────
    # Observability
    # ──────────────────────────────────────────────────────────────────

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get complete system health for observability / frontend.
        """
        return {
            'bar_count': self._bar_count,
            'regime': self._current_regime.to_dict() if self._current_regime else None,
            'cross_asset': self._current_cross_asset.to_dict() if self._current_cross_asset else None,
            'edge_monitor': self.edge_monitor.get_stats(),
            'calibrator': self.calibrator.get_stats(),
            'combiner': self.combiner.get_stats(),
            'should_refit': self.edge_monitor.should_refit(),
            'config': {
                'use_evidence_engine': self.config.use_evidence_engine,
                'use_adaptive_regime': self.config.use_adaptive_regime,
                'use_calibration': self.config.use_calibration,
                'use_quality_sizing': self.config.use_quality_sizing,
                'use_cross_asset': self.config.use_cross_asset,
                'use_edge_monitor': self.config.use_edge_monitor,
                'min_confirming_sources': self.config.min_confirming_sources,
                'base_threshold': self.config.base_threshold,
            },
        }

    # ──────────────────────────────────────────────────────────────────
    # Checkpoint persistence
    # ──────────────────────────────────────────────────────────────────

    def save_checkpoint(
        self,
        path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save learning state (calibrator + combiner + edge monitor) to JSON.

        Returns the path written.  If *path* is None a timestamped file is
        created under ``data/checkpoints/``.
        """
        from .checkpoint import save_checkpoint, DEFAULT_CHECKPOINT_DIR

        if path is None:
            DEFAULT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            ts = __import__("datetime").datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            path = str(DEFAULT_CHECKPOINT_DIR / f"checkpoint_{ts}.json")

        save_checkpoint(
            self.calibrator, self.edge_monitor, self.combiner,
            path, metadata,
        )
        return path

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load learning state from a checkpoint file.

        Restores calibrator, combiner, and edge monitor in-place.
        Returns checkpoint metadata (source info, trade counts).
        """
        from .checkpoint import apply_checkpoint
        return apply_checkpoint(self, path)

    def warmup_feature_store(self, bars: List[Dict[str, Any]]) -> int:
        """
        Feed historical bars through the feature store only (no trading).

        Used for live mode to build z-score baselines before the session's
        first real bar.  Call *after* ``new_session()`` and ``load_checkpoint()``.

        Returns number of bars processed.
        """
        for bar in bars:
            self._current_fv = self.feature_store.update(bar)
        self._bar_count += len(bars)
        return len(bars)

    # ──────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────

    @property
    def current_feature_vector(self) -> Optional[FeatureVector]:
        return self._current_fv

    @property
    def current_regime_state(self) -> Optional[RegimeState]:
        return self._current_regime

    @property
    def current_cross_asset_state(self) -> Optional[CrossAssetState]:
        return self._current_cross_asset
