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
from .volume_profile import VolumeProfileStrategy
from .gap_liquidity import GapLiquidityStrategy
from .absorption_reversal import AbsorptionReversalStrategy
from .momentum_flow import MomentumFlowStrategy
from .exhaustion_fade import ExhaustionFadeStrategy
from .iceberg_defense import IcebergDefenseStrategy

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
    'VolumeProfileStrategy',
    'GapLiquidityStrategy',
    'AbsorptionReversalStrategy',
    'MomentumFlowStrategy',
    'ExhaustionFadeStrategy',
    'IcebergDefenseStrategy',
]
