"""
Shared strategy registry builder used by runtime components.
"""
from __future__ import annotations

from typing import Dict

from .strategies.base_strategy import BaseStrategy
from .strategies.mean_reversion import MeanReversionStrategy
from .strategies.pullback import PullbackStrategy
from .strategies.momentum import MomentumStrategy
from .strategies.rotation import RotationStrategy
from .strategies.vwap_magnet import VWAPMagnetStrategy
from .strategies.volume_profile import VolumeProfileStrategy
from .strategies.gap_liquidity import GapLiquidityStrategy
from .strategies.absorption_reversal import AbsorptionReversalStrategy
from .strategies.momentum_flow import MomentumFlowStrategy
from .strategies.exhaustion_fade import ExhaustionFadeStrategy


def build_strategy_registry() -> Dict[str, BaseStrategy]:
    """Create a fresh strategy registry with all supported strategy instances."""
    return {
        "mean_reversion": MeanReversionStrategy(),
        "pullback": PullbackStrategy(),
        "momentum": MomentumStrategy(),
        "rotation": RotationStrategy(),
        "vwap_magnet": VWAPMagnetStrategy(),
        "volume_profile": VolumeProfileStrategy(),
        "gap_liquidity": GapLiquidityStrategy(),
        "absorption_reversal": AbsorptionReversalStrategy(),
        "momentum_flow": MomentumFlowStrategy(),
        "exhaustion_fade": ExhaustionFadeStrategy(),
    }

