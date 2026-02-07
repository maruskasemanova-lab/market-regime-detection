"""
Edge Monitor: Real-time strategy edge tracking with degradation detection.

Monitors rolling Sharpe, win rate, and expectancy per strategy/regime.
Detects when a strategy's edge has degraded and recommends actions:
  TRADE → normal operation
  TIGHTEN → raise thresholds
  PAUSE → stop trading this strategy
  REFIT → trigger WFO re-optimization

Anti-bias:
  - Minimum 20 trades before any action (small sample protection)
  - Rolling window (30 trades) prevents ancient results from dominating
  - Separate tracking per strategy × regime avoids cross-contamination
  - Degradation requires persistent underperformance, not single bad trade
"""
import math
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

# Minimum trades before edge assessment
MIN_TRADES_FOR_ASSESSMENT = 20
# Rolling window for edge computation
ROLLING_WINDOW = 30


class EdgeStatus(Enum):
    STRONG = "STRONG"
    NORMAL = "NORMAL"
    DEGRADED = "DEGRADED"
    DEAD = "DEAD"
    INSUFFICIENT = "INSUFFICIENT"  # Not enough data


class RecommendedAction(Enum):
    TRADE = "TRADE"
    TIGHTEN = "TIGHTEN"
    PAUSE = "PAUSE"
    REFIT = "REFIT"


@dataclass
class EdgeHealth:
    """Health assessment for a strategy in a regime."""
    strategy: str
    regime: str
    status: EdgeStatus = EdgeStatus.INSUFFICIENT
    action: RecommendedAction = RecommendedAction.TRADE
    rolling_sharpe: float = 0.0
    rolling_win_rate: float = 0.5
    rolling_expectancy: float = 0.0
    total_trades: int = 0
    threshold_adjustment: float = 0.0  # Points to add to threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy': self.strategy,
            'regime': self.regime,
            'status': self.status.value,
            'action': self.action.value,
            'rolling_sharpe': round(self.rolling_sharpe, 3),
            'rolling_win_rate': round(self.rolling_win_rate, 3),
            'rolling_expectancy': round(self.rolling_expectancy, 3),
            'total_trades': self.total_trades,
            'threshold_adjustment': round(self.threshold_adjustment, 1),
        }


@dataclass
class TradeRecord:
    """Minimal trade record for edge tracking."""
    strategy: str
    regime: str
    pnl_r: float  # P&L in R-multiples
    was_profitable: bool
    bar_index: int = 0
    confidence: float = 0.0


class StrategyEdgeTracker:
    """Rolling edge tracker for a single strategy × regime pair."""

    def __init__(self, window: int = ROLLING_WINDOW):
        self._window = window
        self._trades: deque = deque(maxlen=window)
        self._total_trades = 0
        self._total_wins = 0

    @property
    def n_trades(self) -> int:
        return len(self._trades)

    @property
    def total_trades(self) -> int:
        return self._total_trades

    def add_trade(self, pnl_r: float, was_profitable: bool):
        """Record a trade result."""
        self._trades.append((pnl_r, was_profitable))
        self._total_trades += 1
        if was_profitable:
            self._total_wins += 1

    @property
    def rolling_win_rate(self) -> float:
        if not self._trades:
            return 0.5
        wins = sum(1 for _, won in self._trades if won)
        return wins / len(self._trades)

    @property
    def rolling_sharpe(self) -> float:
        if len(self._trades) < 5:
            return 0.0
        pnls = [p for p, _ in self._trades]
        mean = sum(pnls) / len(pnls)
        var = sum((p - mean) ** 2 for p in pnls) / len(pnls)
        std = math.sqrt(var) if var > 0 else 0.0
        if std < 1e-10:
            return 0.0 if abs(mean) < 1e-10 else (10.0 if mean > 0 else -10.0)
        return mean / std

    @property
    def rolling_expectancy(self) -> float:
        if not self._trades:
            return 0.0
        pnls = [p for p, _ in self._trades]
        return sum(pnls) / len(pnls)

    @property
    def profit_factor(self) -> float:
        if not self._trades:
            return 1.0
        gross_profit = sum(p for p, _ in self._trades if p > 0)
        gross_loss = abs(sum(p for p, _ in self._trades if p < 0))
        if gross_loss < 1e-10:
            return 10.0 if gross_profit > 0 else 1.0
        return gross_profit / gross_loss

    @property
    def max_drawdown_r(self) -> float:
        """Max drawdown in R-multiples."""
        if not self._trades:
            return 0.0
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl, _ in self._trades:
            cumulative += pnl
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)
        return max_dd


class EdgeMonitor:
    """
    Real-time edge monitoring across all strategies and regimes.

    Usage:
        monitor = EdgeMonitor()

        # After each trade closes:
        monitor.update_trade(strategy='momentum', regime='TRENDING',
                             pnl_r=1.5, was_profitable=True)

        # Before generating signals:
        health = monitor.get_health('momentum', 'TRENDING')
        if health.action == RecommendedAction.PAUSE:
            skip_strategy()
        elif health.action == RecommendedAction.TIGHTEN:
            threshold += health.threshold_adjustment
    """

    def __init__(self):
        # Key: (strategy, regime) → StrategyEdgeTracker
        self._trackers: Dict[Tuple[str, str], StrategyEdgeTracker] = {}
        # Global tracker (all strategies, all regimes)
        self._global_tracker = StrategyEdgeTracker(window=ROLLING_WINDOW * 3)
        # Degradation events log
        self._degradation_events: List[Dict[str, Any]] = []

    def update_trade(
        self,
        strategy: str,
        regime: str,
        pnl_r: float,
        was_profitable: bool,
        bar_index: int = 0,
        confidence: float = 0.0,
    ):
        """Record a closed trade for edge tracking."""
        key = (strategy, regime)
        if key not in self._trackers:
            self._trackers[key] = StrategyEdgeTracker()

        self._trackers[key].add_trade(pnl_r, was_profitable)
        self._global_tracker.add_trade(pnl_r, was_profitable)

        # Check for degradation events
        health = self._assess_health(strategy, regime)
        if health.status in (EdgeStatus.DEGRADED, EdgeStatus.DEAD):
            event = {
                'bar_index': bar_index,
                'strategy': strategy,
                'regime': regime,
                'status': health.status.value,
                'action': health.action.value,
                'sharpe': health.rolling_sharpe,
                'win_rate': health.rolling_win_rate,
            }
            self._degradation_events.append(event)
            logger.warning(
                "Edge degradation: %s in %s → %s (Sharpe=%.2f, WR=%.1f%%)",
                strategy, regime, health.status.value,
                health.rolling_sharpe, health.rolling_win_rate * 100,
            )

    def get_health(self, strategy: str, regime: str) -> EdgeHealth:
        """Get current edge health for a strategy × regime pair."""
        return self._assess_health(strategy, regime)

    def get_all_health(self) -> List[EdgeHealth]:
        """Get health for all tracked strategy × regime pairs."""
        return [
            self._assess_health(strategy, regime)
            for strategy, regime in self._trackers.keys()
        ]

    def get_global_health(self) -> EdgeHealth:
        """Get overall system health."""
        tracker = self._global_tracker
        health = EdgeHealth(strategy='_global_', regime='ALL')
        health.total_trades = tracker.total_trades

        if tracker.n_trades < MIN_TRADES_FOR_ASSESSMENT:
            health.status = EdgeStatus.INSUFFICIENT
            health.action = RecommendedAction.TRADE
            return health

        health.rolling_sharpe = tracker.rolling_sharpe
        health.rolling_win_rate = tracker.rolling_win_rate
        health.rolling_expectancy = tracker.rolling_expectancy

        health.status, health.action, health.threshold_adjustment = (
            self._classify_edge(tracker)
        )
        return health

    def should_refit(self) -> bool:
        """
        Should we trigger WFO refit?
        True when multiple strategies show degradation.
        """
        degraded_count = 0
        total_assessed = 0

        for (strategy, regime), tracker in self._trackers.items():
            if tracker.n_trades >= MIN_TRADES_FOR_ASSESSMENT:
                total_assessed += 1
                status, _, _ = self._classify_edge(tracker)
                if status in (EdgeStatus.DEGRADED, EdgeStatus.DEAD):
                    degraded_count += 1

        if total_assessed == 0:
            return False

        # Refit if >50% of assessed strategies are degraded
        return degraded_count / total_assessed > 0.5

    def get_degradation_events(self) -> List[Dict[str, Any]]:
        """Get history of degradation events."""
        return list(self._degradation_events)

    def _assess_health(self, strategy: str, regime: str) -> EdgeHealth:
        """Assess edge health for a strategy × regime pair."""
        key = (strategy, regime)
        tracker = self._trackers.get(key)
        health = EdgeHealth(strategy=strategy, regime=regime)

        if tracker is None or tracker.n_trades < MIN_TRADES_FOR_ASSESSMENT:
            health.status = EdgeStatus.INSUFFICIENT
            health.action = RecommendedAction.TRADE
            health.total_trades = tracker.total_trades if tracker else 0
            if tracker:
                health.rolling_win_rate = tracker.rolling_win_rate
                health.rolling_sharpe = tracker.rolling_sharpe
                health.rolling_expectancy = tracker.rolling_expectancy
            return health

        health.total_trades = tracker.total_trades
        health.rolling_sharpe = tracker.rolling_sharpe
        health.rolling_win_rate = tracker.rolling_win_rate
        health.rolling_expectancy = tracker.rolling_expectancy

        health.status, health.action, health.threshold_adjustment = (
            self._classify_edge(tracker)
        )
        return health

    @staticmethod
    def _classify_edge(
        tracker: StrategyEdgeTracker,
    ) -> Tuple[EdgeStatus, RecommendedAction, float]:
        """
        Classify edge status and recommend action.

        Returns: (status, action, threshold_adjustment)
        """
        sharpe = tracker.rolling_sharpe
        wr = tracker.rolling_win_rate
        expect = tracker.rolling_expectancy
        pf = tracker.profit_factor

        # STRONG: clear positive edge
        if sharpe > 1.5 and wr > 0.55 and expect > 0.5:
            return EdgeStatus.STRONG, RecommendedAction.TRADE, 0.0

        # DEAD: clear negative edge
        if sharpe < 0 and wr < 0.35 and tracker.n_trades >= MIN_TRADES_FOR_ASSESSMENT:
            return EdgeStatus.DEAD, RecommendedAction.PAUSE, 15.0

        # DEGRADED: weak or deteriorating edge
        if sharpe < 0.5 or wr < 0.40:
            return EdgeStatus.DEGRADED, RecommendedAction.TIGHTEN, 10.0

        # NORMAL: reasonable edge
        return EdgeStatus.NORMAL, RecommendedAction.TRADE, 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get monitor statistics for observability."""
        stats = {
            'global': {
                'total_trades': self._global_tracker.total_trades,
                'rolling_sharpe': round(self._global_tracker.rolling_sharpe, 3),
                'rolling_win_rate': round(self._global_tracker.rolling_win_rate, 3),
            },
            'strategies': {},
            'degradation_events': len(self._degradation_events),
            'should_refit': self.should_refit(),
        }
        for (strategy, regime), tracker in self._trackers.items():
            key = f"{strategy}@{regime}"
            stats['strategies'][key] = {
                'trades': tracker.total_trades,
                'rolling_trades': tracker.n_trades,
                'sharpe': round(tracker.rolling_sharpe, 3),
                'win_rate': round(tracker.rolling_win_rate, 3),
                'expectancy': round(tracker.rolling_expectancy, 3),
                'profit_factor': round(tracker.profit_factor, 3),
                'max_dd_r': round(tracker.max_drawdown_r, 3),
            }
        return stats

    def reset(self):
        """Reset for new context."""
        self._trackers.clear()
        self._global_tracker = StrategyEdgeTracker(window=ROLLING_WINDOW * 3)
        self._degradation_events.clear()
