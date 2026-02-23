"""
Confidence Calibration: Maps raw strategy scores to P(profitable).

Uses isotonic regression on a rolling window of trade outcomes to
learn the mapping: raw_confidence → actual_win_probability.

Anti-bias:
  - Per-strategy, per-regime calibration avoids cross-contamination
  - Minimum sample requirement (20 trades) before calibration activates
  - Falls back to identity mapping with insufficient data
  - Rolling window (50 trades) prevents stale calibration
"""
import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Minimum trades before calibration kicks in
MIN_CALIBRATION_TRADES = 20
# Conservative bias while calibrator is still warming up.
UNCALIBRATED_CONFIDENCE_MULTIPLIER = 0.8


def _conservative_identity(raw_confidence: float) -> float:
    base = max(0.0, min(1.0, raw_confidence / 100.0))
    return max(0.0, min(1.0, base * UNCALIBRATED_CONFIDENCE_MULTIPLIER))


@dataclass
class TradeOutcome:
    """Record of a trade for calibration."""
    strategy: str
    regime: str
    raw_confidence: float  # Original strategy confidence (0-100)
    was_profitable: bool
    pnl_r: float = 0.0  # P&L in R-multiples


class IsotonicCalibrator:
    """
    Simple isotonic regression for monotone calibration.
    Maps raw confidence bins to empirical win rates.

    Lightweight: no sklearn dependency, hand-rolled for streaming use.
    """

    def __init__(self, n_bins: int = 10, lookback: int = 50):
        self.n_bins = n_bins
        self.lookback = lookback
        self._outcomes: deque = deque(maxlen=lookback)

    @property
    def n_trades(self) -> int:
        return len(self._outcomes)

    def update(self, raw_confidence: float, was_profitable: bool):
        """Record a new trade outcome."""
        self._outcomes.append((raw_confidence, was_profitable))

    def calibrate(self, raw_confidence: float) -> float:
        """
        Map raw confidence to calibrated P(profitable).
        Returns value in [0, 1].
        """
        if self.n_trades < MIN_CALIBRATION_TRADES:
            return _conservative_identity(raw_confidence)

        # Bin outcomes by confidence range
        bin_width = 100.0 / self.n_bins
        bin_wins: Dict[int, int] = {}
        bin_totals: Dict[int, int] = {}

        for conf, won in self._outcomes:
            b = min(self.n_bins - 1, int(conf / bin_width))
            bin_totals[b] = bin_totals.get(b, 0) + 1
            if won:
                bin_wins[b] = bin_wins.get(b, 0) + 1

        # Compute empirical win rate per bin
        bin_rates = {}
        for b in range(self.n_bins):
            total = bin_totals.get(b, 0)
            if total >= 3:  # Minimum per-bin samples
                bin_rates[b] = bin_wins.get(b, 0) / total
            else:
                bin_rates[b] = None

        # Isotonic pass: ensure monotonically non-decreasing
        rates = []
        for b in range(self.n_bins):
            r = bin_rates.get(b)
            if r is not None:
                rates.append((b, r))

        if not rates:
            return _conservative_identity(raw_confidence)

        # Pool-adjacent-violators for monotonicity
        isotonic_rates = self._pava([r for _, r in rates])
        rate_map = {}
        for i, (b, _) in enumerate(rates):
            rate_map[b] = isotonic_rates[i]

        # Interpolate for the query bin
        query_bin = min(self.n_bins - 1, int(raw_confidence / bin_width))
        if query_bin in rate_map:
            return max(0.0, min(1.0, rate_map[query_bin]))

        # Linear interpolation between nearest known bins
        known_bins = sorted(rate_map.keys())
        if not known_bins:
            return _conservative_identity(raw_confidence)

        if query_bin <= known_bins[0]:
            return max(0.0, min(1.0, rate_map[known_bins[0]]))
        if query_bin >= known_bins[-1]:
            return max(0.0, min(1.0, rate_map[known_bins[-1]]))

        # Find surrounding bins
        lower_bin = max(b for b in known_bins if b <= query_bin)
        upper_bin = min(b for b in known_bins if b >= query_bin)
        if lower_bin == upper_bin:
            return max(0.0, min(1.0, rate_map[lower_bin]))

        # Interpolate
        frac = (query_bin - lower_bin) / (upper_bin - lower_bin)
        result = rate_map[lower_bin] + frac * (rate_map[upper_bin] - rate_map[lower_bin])
        return max(0.0, min(1.0, result))

    @staticmethod
    def _pava(values: List[float]) -> List[float]:
        """Pool Adjacent Violators Algorithm for isotonic regression."""
        if not values:
            return values
        result = list(values)
        n = len(result)
        # Forward pass: enforce non-decreasing
        i = 0
        while i < n - 1:
            if result[i] > result[i + 1]:
                # Pool: average the block
                j = i + 1
                while j < n and result[j] < result[i]:
                    j += 1
                # Average from i to j-1
                block_avg = sum(result[i:j]) / (j - i)
                for k in range(i, j):
                    result[k] = block_avg
                # Re-check from the start of block
                i = max(0, i - 1)
            else:
                i += 1
        return result


class ConfidenceCalibrator:
    """
    Per-strategy, per-regime confidence calibration.

    Maps raw strategy confidence scores to calibrated P(profitable).
    Uses rolling isotonic regression per (strategy, regime) pair.
    """

    def __init__(self, lookback_trades: int = 50):
        self._lookback = lookback_trades
        # Key: (strategy_name, regime) → IsotonicCalibrator
        self._calibrators: Dict[Tuple[str, str], IsotonicCalibrator] = {}
        # Global fallback calibrator (all strategies, all regimes)
        self._global_calibrator = IsotonicCalibrator(lookback=lookback_trades * 3)

    def calibrate(
        self,
        strategy: str,
        raw_confidence: float,
        regime: str,
    ) -> float:
        """
        Map raw confidence to calibrated P(profitable).

        Falls back gracefully:
          1. (strategy, regime) calibrator if enough trades
          2. (strategy, 'ALL') calibrator if enough trades
          3. Global calibrator
          4. Identity mapping (raw / 100)
        """
        # Try specific (strategy, regime) calibrator
        key = (strategy, regime)
        if key in self._calibrators and self._calibrators[key].n_trades >= MIN_CALIBRATION_TRADES:
            return self._calibrators[key].calibrate(raw_confidence)

        # Try strategy-level (all regimes)
        key_all = (strategy, 'ALL')
        if key_all in self._calibrators and self._calibrators[key_all].n_trades >= MIN_CALIBRATION_TRADES:
            return self._calibrators[key_all].calibrate(raw_confidence)

        # Try global
        if self._global_calibrator.n_trades >= MIN_CALIBRATION_TRADES:
            return self._global_calibrator.calibrate(raw_confidence)

        # Conservative fallback while calibration has insufficient support.
        return _conservative_identity(raw_confidence)

    def update(
        self,
        strategy: str,
        raw_confidence: float,
        regime: str,
        was_profitable: bool,
    ):
        """Record a trade outcome for calibration updates."""
        # Update (strategy, regime) calibrator
        key = (strategy, regime)
        if key not in self._calibrators:
            self._calibrators[key] = IsotonicCalibrator(lookback=self._lookback)
        self._calibrators[key].update(raw_confidence, was_profitable)

        # Update (strategy, ALL) calibrator
        key_all = (strategy, 'ALL')
        if key_all not in self._calibrators:
            self._calibrators[key_all] = IsotonicCalibrator(lookback=self._lookback)
        self._calibrators[key_all].update(raw_confidence, was_profitable)

        # Update global calibrator
        self._global_calibrator.update(raw_confidence, was_profitable)

    def get_stats(self) -> Dict[str, Any]:
        """Get calibration statistics for observability."""
        stats = {}
        for (strategy, regime), cal in self._calibrators.items():
            key = f"{strategy}_{regime}"
            stats[key] = {
                'n_trades': cal.n_trades,
                'calibrated': cal.n_trades >= MIN_CALIBRATION_TRADES,
            }
        stats['global'] = {
            'n_trades': self._global_calibrator.n_trades,
            'calibrated': self._global_calibrator.n_trades >= MIN_CALIBRATION_TRADES,
        }
        return stats

    def reset(self):
        """Reset all calibrators for new context."""
        self._calibrators.clear()
        self._global_calibrator = IsotonicCalibrator(
            lookback=self._lookback * 3
        )
