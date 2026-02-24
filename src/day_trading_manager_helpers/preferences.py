"""Default strategy preference maps for DayTradingManager."""

from __future__ import annotations

from typing import Dict, List

from ..strategies.base_strategy import Regime


def bootstrap_ticker_preferences() -> Dict[str, Dict[Regime, List[str]]]:
    """Bootstrap preferences used during early manager initialization."""

    return {
        "NVDA": {
            Regime.TRENDING: ["pullback", "momentum", "volume_profile"],
            Regime.CHOPPY: [],
            Regime.MIXED: ["volume_profile", "vwap_magnet"],
        },
        "TSLA": {
            Regime.TRENDING: ["momentum", "gap_liquidity"],
            Regime.CHOPPY: [],
            Regime.MIXED: [],
        },
        "AAPL": {
            Regime.TRENDING: ["mean_reversion", "vwap_magnet", "pullback"],
            Regime.CHOPPY: ["mean_reversion", "vwap_magnet"],
            Regime.MIXED: ["mean_reversion", "vwap_magnet"],
        },
        "SPY": {
            Regime.TRENDING: ["momentum_flow", "gap_liquidity"],
            Regime.CHOPPY: ["vwap_magnet", "mean_reversion"],
            Regime.MIXED: ["vwap_magnet"],
        },
        "QQQ": {
            Regime.TRENDING: ["momentum_flow", "scalp_l2_intrabar", "gap_liquidity"],
            Regime.CHOPPY: ["scalp_l2_intrabar", "absorption_reversal", "mean_reversion"],
            Regime.MIXED: ["scalp_l2_intrabar", "vwap_magnet"],
        },
    }


def default_regime_preferences() -> Dict[Regime, List[str]]:
    return {
        Regime.TRENDING: [
            "momentum_flow",
            "momentum",
            "pullback",
            "gap_liquidity",
            "scalp_l2_intrabar",
            "volume_profile",
            "vwap_magnet",
            "evidence_scalp",
        ],
        Regime.CHOPPY: [
            "absorption_reversal",
            "exhaustion_fade",
            "mean_reversion",
            "vwap_magnet",
            "scalp_l2_intrabar",
            "volume_profile",
            "evidence_scalp",
        ],
        Regime.MIXED: [
            "exhaustion_fade",
            "absorption_reversal",
            "volume_profile",
            "gap_liquidity",
            "scalp_l2_intrabar",
            "mean_reversion",
            "vwap_magnet",
            "rotation",
            "evidence_scalp",
        ],
    }


def default_micro_regime_preferences() -> Dict[str, List[str]]:
    return {
        "TRENDING_UP": [
            "momentum_flow",
            "momentum",
            "pullback",
            "gap_liquidity",
            "scalp_l2_intrabar",
            "evidence_scalp",
        ],
        "TRENDING_DOWN": [
            "momentum_flow",
            "momentum",
            "gap_liquidity",
            "pullback",
            "scalp_l2_intrabar",
            "evidence_scalp",
        ],
        "CHOPPY": [
            "absorption_reversal",
            "exhaustion_fade",
            "mean_reversion",
            "vwap_magnet",
            "evidence_scalp",
        ],
        "ABSORPTION": [
            "absorption_reversal",
            "exhaustion_fade",
            "vwap_magnet",
            "evidence_scalp",
        ],
        "BREAKOUT": [
            "momentum_flow",
            "momentum",
            "gap_liquidity",
            "scalp_l2_intrabar",
            "evidence_scalp",
        ],
        "MIXED": ["exhaustion_fade", "volume_profile", "rotation", "evidence_scalp"],
        "TRANSITION": ["vwap_magnet", "volume_profile", "rotation", "evidence_scalp"],
        "UNKNOWN": ["vwap_magnet", "volume_profile", "evidence_scalp"],
    }


def default_ticker_preferences() -> Dict[str, Dict[Regime, List[str]]]:
    return {
        "NVDA": {
            Regime.TRENDING: ["pullback", "momentum", "volume_profile"],
            Regime.CHOPPY: [],
            Regime.MIXED: ["volume_profile", "vwap_magnet"],
        },
        "TSLA": {
            Regime.TRENDING: ["momentum", "gap_liquidity"],
            Regime.CHOPPY: [],
            Regime.MIXED: [],
        },
        "AAPL": {
            Regime.TRENDING: ["mean_reversion", "vwap_magnet", "pullback"],
            Regime.CHOPPY: ["mean_reversion", "vwap_magnet"],
            Regime.MIXED: ["mean_reversion", "vwap_magnet"],
        },
        "AMD": {
            Regime.TRENDING: ["momentum", "volume_profile", "pullback"],
            Regime.CHOPPY: [],
            Regime.MIXED: ["volume_profile", "vwap_magnet"],
        },
        "GOOGL": {
            Regime.TRENDING: ["mean_reversion", "vwap_magnet"],
            Regime.CHOPPY: ["mean_reversion", "vwap_magnet"],
            Regime.MIXED: ["mean_reversion", "vwap_magnet"],
        },
        "META": {
            Regime.TRENDING: ["pullback", "vwap_magnet", "volume_profile"],
            Regime.CHOPPY: ["mean_reversion"],
            Regime.MIXED: ["pullback", "vwap_magnet"],
        },
        "MSFT": {
            Regime.TRENDING: ["mean_reversion", "vwap_magnet"],
            Regime.CHOPPY: ["mean_reversion", "vwap_magnet"],
            Regime.MIXED: ["mean_reversion", "vwap_magnet"],
        },
        "MU": {
            Regime.TRENDING: ["momentum", "gap_liquidity", "volume_profile"],
            Regime.CHOPPY: [],
            Regime.MIXED: ["volume_profile"],
        },
        "AMZN": {
            Regime.TRENDING: ["pullback", "mean_reversion", "vwap_magnet"],
            Regime.CHOPPY: ["mean_reversion"],
            Regime.MIXED: ["pullback", "mean_reversion"],
        },
    }
