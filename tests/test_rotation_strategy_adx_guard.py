"""Regression tests for RotationStrategy ADX handling."""

from datetime import datetime, timezone
from pathlib import Path
import sys

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.strategies.base_strategy import Regime, SignalType
from src.strategies.rotation import RotationStrategy


def _ts() -> datetime:
    return datetime(2026, 2, 3, 14, 31, tzinfo=timezone.utc)


def test_rotation_strategy_handles_none_adx_value_for_buy_path() -> None:
    strategy = RotationStrategy(
        lookback_period=3,
        rotation_threshold=0.2,
        max_performance=5.0,
        min_confidence=0.0,
    )
    ohlcv = {
        "open": [99.7, 100.0, 100.4],
        "high": [100.0, 100.6, 101.2],
        "low": [99.5, 99.8, 100.1],
        "close": [99.8, 100.2, 100.8],
        "volume": [900.0, 1200.0, 1700.0],
    }
    indicators = {
        "vwap": [100.7],
        "rsi": [56.0],
        "sma": [100.6],
        "ema": [100.9],
        "adx": [None],  # Previously could raise TypeError during trend filter comparison.
        "order_flow": {"signed_aggression": 0.05},
    }

    signal = strategy.generate_signal(
        current_price=100.9,
        ohlcv=ohlcv,
        indicators=indicators,
        regime=Regime.MIXED,
        timestamp=_ts(),
    )

    assert signal is not None
    assert signal.signal_type == SignalType.BUY


def test_rotation_strategy_handles_none_adx_value_for_sell_path() -> None:
    strategy = RotationStrategy(
        lookback_period=3,
        rotation_threshold=0.2,
        max_performance=5.0,
        min_confidence=0.0,
    )
    ohlcv = {
        "open": [100.8, 100.3, 99.8],
        "high": [101.0, 100.5, 100.0],
        "low": [100.2, 99.9, 99.3],
        "close": [100.7, 100.1, 99.6],
        "volume": [950.0, 1200.0, 1800.0],
    }
    indicators = {
        "vwap": [99.7],
        "rsi": [44.0],
        "sma": [99.8],
        "ema": [99.5],
        "adx": [None],  # Regression: must default safely instead of crashing.
        "order_flow": {"signed_aggression": -0.05},
    }

    signal = strategy.generate_signal(
        current_price=99.6,
        ohlcv=ohlcv,
        indicators=indicators,
        regime=Regime.MIXED,
        timestamp=_ts(),
    )

    assert signal is not None
    assert signal.signal_type == SignalType.SELL
