"""
Cross-Asset Context: Index and sector relative strength.

Provides context about broader market conditions so the system
doesn't generate bullish signals during sector-wide selloffs.

Anti-bias:
  - Headwind score is a penalty, not a gate (doesn't block signals outright)
  - Rolling correlation adapts to changing inter-asset relationships
  - Minimum lookback prevents premature correlation estimates
"""
import math
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class CrossAssetState:
    """Cross-asset context at a point in time."""
    index_trend: float = 0.0            # Index (QQQ) 5-bar momentum direction
    index_momentum_5: float = 0.0       # QQQ 5-bar ROC (%)
    index_momentum_20: float = 0.0      # QQQ 20-bar ROC (%)
    sector_relative: float = 0.0        # Ticker RS vs index
    correlation_20: float = 0.0         # 20-bar rolling correlation with index
    headwind_score: float = 0.0         # 0-1 (0=tailwind/neutral, 1=strong headwind)
    index_available: bool = False       # Whether index data is loaded
    bar_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'index_trend': round(self.index_trend, 4),
            'index_momentum_5': round(self.index_momentum_5, 3),
            'index_momentum_20': round(self.index_momentum_20, 3),
            'sector_relative': round(self.sector_relative, 4),
            'correlation_20': round(self.correlation_20, 3),
            'headwind_score': round(self.headwind_score, 3),
            'index_available': self.index_available,
        }


class CrossAssetContext:
    """
    Tracks reference tickers (index/sector ETFs) alongside the target ticker
    to provide relative strength and headwind/tailwind context.

    Usage:
        ctx = CrossAssetContext(reference_tickers=['QQQ', 'SPY'])

        # Feed reference bars (must be time-aligned with target)
        ctx.update_reference('QQQ', qqq_bar)
        ctx.update_reference('SPY', spy_bar)

        # Feed target bar and get context
        state = ctx.update_target(target_bar)
        # state.headwind_score → 0.8 means strong index-against-signal headwind
    """

    def __init__(
        self,
        reference_tickers: Optional[List[str]] = None,
        correlation_window: int = 20,
        momentum_short: int = 5,
        momentum_long: int = 20,
    ):
        self._reference_tickers = reference_tickers or ['QQQ']
        self._correlation_window = correlation_window
        self._momentum_short = momentum_short
        self._momentum_long = momentum_long

        # Reference ticker data
        self._ref_closes: Dict[str, deque] = {
            t: deque(maxlen=200) for t in self._reference_tickers
        }
        self._ref_returns: Dict[str, deque] = {
            t: deque(maxlen=200) for t in self._reference_tickers
        }

        # Target ticker data
        self._target_closes: deque = deque(maxlen=200)
        self._target_returns: deque = deque(maxlen=200)

        self._bar_index = 0

    def update_reference(self, ticker: str, bar: Dict[str, Any]):
        """
        Update a reference ticker with a new bar.
        Call this before update_target() for each bar.
        """
        if ticker not in self._ref_closes:
            self._ref_closes[ticker] = deque(maxlen=200)
            self._ref_returns[ticker] = deque(maxlen=200)

        close = float(bar.get('close', 0))
        closes = self._ref_closes[ticker]

        if closes and closes[-1] > 0:
            ret = (close - closes[-1]) / closes[-1]
        else:
            ret = 0.0

        closes.append(close)
        self._ref_returns[ticker].append(ret)

    def update_target(self, bar: Dict[str, Any]) -> CrossAssetState:
        """
        Update target ticker and compute cross-asset state.
        """
        self._bar_index += 1

        close = float(bar.get('close', 0))
        closes = self._target_closes

        if closes and closes[-1] > 0:
            ret = (close - closes[-1]) / closes[-1]
        else:
            ret = 0.0

        closes.append(close)
        self._target_returns.append(ret)

        # Primary reference (usually QQQ)
        primary_ref = self._reference_tickers[0]
        ref_closes = self._ref_closes.get(primary_ref, deque())

        if not ref_closes or len(ref_closes) < 2:
            return CrossAssetState(
                bar_index=self._bar_index,
                index_available=False,
            )

        # Index momentum
        idx_mom_5 = self._compute_roc(ref_closes, self._momentum_short)
        idx_mom_20 = self._compute_roc(ref_closes, self._momentum_long)
        idx_trend = 1.0 if idx_mom_5 > 0 else (-1.0 if idx_mom_5 < 0 else 0.0)

        # Relative strength: target ROC - index ROC
        target_roc_5 = self._compute_roc(closes, self._momentum_short)
        sector_relative = target_roc_5 - idx_mom_5

        # Rolling correlation
        ref_returns = self._ref_returns.get(primary_ref, deque())
        correlation = self._compute_correlation(
            self._target_returns, ref_returns, self._correlation_window
        )

        # Headwind score
        headwind = self._compute_headwind(
            idx_mom_5, idx_mom_20, target_roc_5, correlation
        )

        return CrossAssetState(
            index_trend=idx_trend,
            index_momentum_5=idx_mom_5,
            index_momentum_20=idx_mom_20,
            sector_relative=sector_relative,
            correlation_20=correlation,
            headwind_score=headwind,
            index_available=True,
            bar_index=self._bar_index,
        )

    @staticmethod
    def _compute_roc(closes: deque, period: int) -> float:
        """Rate of change (%)."""
        if len(closes) <= period:
            return 0.0
        prev = closes[-period - 1]
        curr = closes[-1]
        if abs(prev) < 1e-10:
            return 0.0
        return (curr - prev) / prev * 100.0

    @staticmethod
    def _compute_correlation(
        returns_a: deque, returns_b: deque, window: int
    ) -> float:
        """Rolling Pearson correlation between two return series."""
        n = min(len(returns_a), len(returns_b), window)
        if n < 5:
            return 0.0

        a = list(returns_a)[-n:]
        b = list(returns_b)[-n:]

        mean_a = sum(a) / n
        mean_b = sum(b) / n

        cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n)) / n
        var_a = sum((x - mean_a) ** 2 for x in a) / n
        var_b = sum((x - mean_b) ** 2 for x in b) / n

        denom = math.sqrt(var_a * var_b) if var_a > 0 and var_b > 0 else 0
        if denom < 1e-10:
            return 0.0

        return max(-1.0, min(1.0, cov / denom))

    @staticmethod
    def _compute_headwind(
        idx_mom_5: float,
        idx_mom_20: float,
        target_roc_5: float,
        correlation: float,
    ) -> float:
        """
        Compute headwind score (0-1).

        0.0 = tailwind or neutral (index supports signal direction)
        1.0 = strong headwind (index moving against likely signal direction)

        Headwind is stronger when:
          - Index momentum is strongly negative (for bullish signals)
          - Correlation is high (stock follows index)
          - Both short and long-term index momentum agree on direction
        """
        if abs(idx_mom_5) < 0.05:
            # Index is flat → no headwind
            return 0.0

        # Base headwind: how strongly is index moving against?
        # For headwind to matter, the stock must be correlated with index
        corr_factor = max(0.0, correlation)  # Only positive correlation creates headwind

        # Index negativity score (for bullish headwind)
        # Positive idx_mom → bullish tailwind → low headwind for bulls, high for bears
        # We compute bidirectional headwind magnitude
        idx_strength = min(1.0, abs(idx_mom_5) / 1.0)  # Normalize: 1% move = full strength

        # Multi-timeframe agreement amplifies headwind
        if idx_mom_5 * idx_mom_20 > 0:
            # Both agree → stronger signal
            agreement_mult = 1.2
        else:
            agreement_mult = 0.8

        # Divergence: if target moves opposite to index, headwind is real
        if target_roc_5 * idx_mom_5 < 0:
            # Target diverging from index → headwind detected
            divergence_mult = 1.3
        else:
            divergence_mult = 0.5  # Moving with index → tailwind

        headwind = idx_strength * corr_factor * agreement_mult * divergence_mult
        return max(0.0, min(1.0, headwind))

    def reset(self):
        """Reset for new session."""
        for t in self._reference_tickers:
            self._ref_closes[t] = deque(maxlen=200)
            self._ref_returns[t] = deque(maxlen=200)
        self._target_closes = deque(maxlen=200)
        self._target_returns = deque(maxlen=200)
        self._bar_index = 0
