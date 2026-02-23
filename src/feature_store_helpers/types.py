"""Shared data structures for the feature store."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class FeatureVector:
    """Immutable snapshot of all features at a point in time."""

    bar_index: int = 0

    # --- Raw price indicators ---
    close: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: float = 0.0
    vwap: float = 0.0

    # --- Technical indicators (raw) ---
    sma_20: float = 0.0
    ema_10: float = 0.0
    ema_20: float = 0.0
    rsi_14: float = 50.0
    atr_14: float = 0.0
    adx_14: Optional[float] = None
    bollinger_width: float = 0.0
    roc_5: float = 0.0
    roc_10: float = 0.0
    roc_20: float = 0.0
    obv: float = 0.0
    obv_slope: float = 0.0
    vwap_distance_pct: float = 0.0
    range_atr_ratio: float = 0.0
    trend_efficiency: float = 0.0

    # --- Normalized (z-score) indicators ---
    rsi_z: float = 0.0
    atr_z: float = 0.0
    adx_z: float = 0.0
    volume_z: float = 0.0
    vwap_dist_z: float = 0.0
    roc_5_z: float = 0.0
    roc_10_z: float = 0.0
    momentum_z: float = 0.0

    # --- Percentile ranks (0-1) ---
    volume_pct_rank: float = 0.5
    atr_pct_rank: float = 0.5
    range_pct_rank: float = 0.5

    # --- L2 Flow features (raw) ---
    l2_has_coverage: bool = False
    l2_delta: float = 0.0
    l2_signed_aggression: float = 0.0
    l2_directional_consistency: float = 0.0
    l2_imbalance: float = 0.0
    l2_absorption_rate: float = 0.0
    l2_sweep_intensity: float = 0.0
    l2_book_pressure: float = 0.0
    l2_large_trader_activity: float = 0.0
    l2_delta_zscore: float = 0.0
    l2_flow_score: float = 0.0
    l2_iceberg_bias: float = 0.0
    l2_participation_ratio: float = 0.0
    l2_delta_acceleration: float = 0.0
    l2_delta_price_divergence: float = 0.0

    # --- L2 Normalized (z-score over rolling window) ---
    l2_delta_z: float = 0.0
    l2_aggression_z: float = 0.0
    l2_imbalance_z: float = 0.0
    l2_book_pressure_z: float = 0.0
    l2_sweep_z: float = 0.0
    l2_flow_score_z: float = 0.0

    # --- Multi-timeframe ---
    tf5_trend_slope: float = 0.0
    tf5_rsi: float = 50.0
    tf15_trend_slope: float = 0.0
    tf15_rsi: float = 50.0
    tf5_volume_ratio: float = 1.0
    tf15_volume_ratio: float = 1.0

    # --- Cross-asset ---
    index_trend: float = 0.0
    sector_relative: float = 0.0
    correlation_20: float = 0.0
    headwind_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class RollingStats:
    """Efficient rolling statistics (mean, std, percentile rank)."""

    def __init__(self, window: int = 100):
        self.window = window
        self._values: deque = deque(maxlen=window)

    def update(self, value: float):
        self._values.append(value)

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def mean(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    @property
    def std(self) -> float:
        n = len(self._values)
        if n < 2:
            return 0.0
        m = self.mean
        variance = sum((x - m) ** 2 for x in self._values) / n
        return math.sqrt(variance) if variance > 0 else 0.0

    def z_score(self, value: float) -> float:
        s = self.std
        if s < 1e-10:
            return 0.0
        return (value - self.mean) / s

    def percentile_rank(self, value: float) -> float:
        if not self._values:
            return 0.5
        count_below = sum(1 for v in self._values if v <= value)
        return count_below / len(self._values)


__all__ = ["FeatureVector", "RollingStats"]
