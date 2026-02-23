"""Multi-timeframe feature helpers for FeatureStore."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List


def aggregate_bars(bars: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate minute bars into a single higher-TF bar."""
    return {
        'open': bars[0]['open'],
        'high': max(bar['high'] for bar in bars),
        'low': min(bar['low'] for bar in bars),
        'close': bars[-1]['close'],
        'volume': sum(bar['volume'] for bar in bars),
    }


def compute_tf_ema_slope(tf_bars: deque, period: int = 5) -> float:
    """EMA slope on higher timeframe bars."""
    if len(tf_bars) < period + 1:
        return 0.0
    closes = [bar['close'] for bar in list(tf_bars)[-period - 1:]]
    mult = 2.0 / (period + 1)
    ema = closes[0]
    prev_ema = ema
    for close in closes[1:]:
        prev_ema = ema
        ema = (close - ema) * mult + ema
    return ema - prev_ema


def compute_tf_rsi(tf_bars: deque, period: int = 14) -> float:
    """Simple RSI on higher timeframe bars."""
    if len(tf_bars) < period + 1:
        return 50.0
    closes = [bar['close'] for bar in list(tf_bars)[-(period + 1):]]
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0))
        losses.append(abs(min(change, 0)))
    avg_g = sum(gains) / len(gains) if gains else 0
    avg_l = sum(losses) / len(losses) if losses else 0
    if avg_l < 1e-10:
        return 100.0
    rs = avg_g / avg_l
    return 100.0 - (100.0 / (1.0 + rs))


def compute_tf_volume_ratio(tf_bars: deque) -> float:
    """Current TF bar volume vs rolling average."""
    if len(tf_bars) < 2:
        return 1.0
    volumes = [bar['volume'] for bar in tf_bars]
    avg = sum(volumes[:-1]) / len(volumes[:-1]) if len(volumes) > 1 else 1.0
    if avg < 1e-10:
        return 1.0
    return volumes[-1] / avg


def compute_multi_timeframe(store: Any, bar: Dict[str, Any]) -> Dict[str, float]:
    """Aggregate 1-min bars into 5-min and 15-min timeframes."""
    bar_entry = {
        'open': float(bar.get('open', 0)),
        'high': float(bar.get('high', 0)),
        'low': float(bar.get('low', 0)),
        'close': float(bar.get('close', 0)),
        'volume': float(bar.get('volume', 0)),
    }
    store._tf5_accumulator.append(bar_entry)
    store._tf15_accumulator.append(bar_entry)

    if len(store._tf5_accumulator) >= 5:
        aggregated = aggregate_bars(store._tf5_accumulator)
        store._tf5_bars.append(aggregated)
        store._tf5_accumulator = []

    if len(store._tf15_accumulator) >= 15:
        aggregated = aggregate_bars(store._tf15_accumulator)
        store._tf15_bars.append(aggregated)
        store._tf15_accumulator = []

    return {
        'tf5_trend_slope': compute_tf_ema_slope(store._tf5_bars, 5),
        'tf5_rsi': compute_tf_rsi(store._tf5_bars),
        'tf15_trend_slope': compute_tf_ema_slope(store._tf15_bars, 5),
        'tf15_rsi': compute_tf_rsi(store._tf15_bars),
        'tf5_volume_ratio': compute_tf_volume_ratio(store._tf5_bars),
        'tf15_volume_ratio': compute_tf_volume_ratio(store._tf15_bars),
    }


__all__ = [
    "aggregate_bars",
    "compute_multi_timeframe",
    "compute_tf_ema_slope",
    "compute_tf_rsi",
    "compute_tf_volume_ratio",
]
