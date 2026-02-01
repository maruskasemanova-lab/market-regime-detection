"""
Trading Strategies Module.
Regime-based strategies with trailing stop-loss support.
"""
from .base_strategy import BaseStrategy, Signal, Position
from .trailing_stop import TrailingStopManager, StopType
from .mean_reversion import MeanReversionStrategy
from .pullback import PullbackStrategy
from .momentum import MomentumStrategy
from .rotation import RotationStrategy
from .vwap_magnet import VWAPMagnetStrategy

__all__ = [
    'BaseStrategy',
    'Signal',
    'Position',
    'TrailingStopManager',
    'StopType',
    'MeanReversionStrategy',
    'PullbackStrategy',
    'MomentumStrategy',
    'RotationStrategy',
    'VWAPMagnetStrategy',
]
