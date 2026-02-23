"""Classifier implementations used by the adaptive regime detector."""

from __future__ import annotations

from typing import Dict

from ..feature_store import FeatureVector
from .constants import CHOPPY, MIXED, TRENDING, default_probabilities


class RuleBasedClassifier:
    """
    Rule-based regime classification using ADX, trend efficiency, and volatility.
    Outputs soft probabilities instead of hard categories.
    """

    def classify(self, fv: FeatureVector) -> Dict[str, float]:
        """Classify regime using normalized features."""

        adx = fv.adx_14
        trend_eff = fv.trend_efficiency
        atr_z = fv.atr_z
        roc_5 = abs(fv.roc_5)

        if adx is None:
            adx = 20.0

        trending_score = 0.0
        if adx > 15:
            trending_score += min(1.0, (adx - 15) / 25)
        trending_score += min(0.5, trend_eff)
        trending_score += min(0.3, roc_5 / 2.0)
        trending_score = min(1.0, trending_score / 1.5)

        choppy_score = 0.0
        if adx < 30:
            choppy_score += min(1.0, (30 - adx) / 20)
        choppy_score += max(0.0, 0.5 - trend_eff)
        if atr_z > 0.5:
            choppy_score += min(0.3, (atr_z - 0.5) * 0.3)
        choppy_score = min(1.0, choppy_score / 1.3)

        mixed_score = max(0.0, 1.0 - trending_score - choppy_score)

        total = trending_score + choppy_score + mixed_score
        if total < 1e-10:
            return default_probabilities()

        return {
            TRENDING: trending_score / total,
            CHOPPY: choppy_score / total,
            MIXED: mixed_score / total,
        }


class L2FlowClassifier:
    """
    L2 order flow-based regime classification.
    Uses flow metrics to provide microstructure-informed regime probabilities.
    """

    def classify(self, fv: FeatureVector) -> Dict[str, float]:
        """Classify regime using L2 flow features."""

        if not fv.l2_has_coverage:
            return default_probabilities()

        aggression = abs(fv.l2_signed_aggression)
        consistency = fv.l2_directional_consistency
        sweep = fv.l2_sweep_intensity
        absorption = fv.l2_absorption_rate
        pressure = abs(fv.l2_book_pressure)
        large_trader = fv.l2_large_trader_activity

        trending_score = 0.0
        trending_score += min(0.4, aggression * 3.0)
        trending_score += min(0.3, consistency * 0.5)
        trending_score += min(0.2, sweep * 1.5)
        trending_score += min(0.1, large_trader * 0.5)

        choppy_score = 0.0
        choppy_score += min(0.4, absorption * 0.7)
        choppy_score += min(0.3, max(0.0, 0.5 - aggression) * 1.5)
        choppy_score += min(0.2, max(0.0, 0.5 - consistency) * 0.6)

        mixed_score = max(0.0, 1.0 - trending_score - choppy_score)

        total = trending_score + choppy_score + mixed_score
        if total < 1e-10:
            return default_probabilities()

        return {
            TRENDING: trending_score / total,
            CHOPPY: choppy_score / total,
            MIXED: mixed_score / total,
        }

    def classify_micro(self, fv: FeatureVector) -> str:
        """Detailed micro-regime from L2 flow."""

        if fv.adx_14 is None:
            return "MIXED"
        adx = float(fv.adx_14)
        trend_eff = max(0.0, min(1.0, float(fv.trend_efficiency)))
        price_change_pct = float(fv.roc_5)
        directional_price_change = price_change_pct if abs(price_change_pct) >= 0.05 else 0.0

        if not fv.l2_has_coverage:
            if trend_eff < 0.15 and adx >= 20.0:
                return "TRANSITION"
            if trend_eff >= 0.62 and adx > 30.0 and directional_price_change != 0.0:
                return "TRENDING_UP" if directional_price_change > 0.0 else "TRENDING_DOWN"
            if adx < 20 and trend_eff < 0.40:
                return "CHOPPY"
            return "MIXED"

        aggression = fv.l2_signed_aggression
        consistency = fv.l2_directional_consistency
        sweep = fv.l2_sweep_intensity
        absorption = fv.l2_absorption_rate
        pressure = fv.l2_book_pressure
        large_trader = fv.l2_large_trader_activity

        if (
            sweep >= 0.25
            and consistency >= 0.50
            and abs(aggression) >= 0.08
            and large_trader >= 0.12
        ):
            return "BREAKOUT"

        if absorption >= 0.45 and abs(pressure) >= 0.08:
            return "ABSORPTION"

        if trend_eff < 0.15 and adx >= 15.0:
            return "TRANSITION"

        if (
            abs(aggression) >= 0.06
            and consistency >= 0.45
            and adx >= 20.0
            and trend_eff >= 0.45
            and directional_price_change != 0.0
        ):
            return "TRENDING_UP" if aggression > 0 else "TRENDING_DOWN"

        if adx < 18.0 and trend_eff < 0.35:
            return "CHOPPY"

        return "MIXED"


class VolatilityClassifier:
    """
    Volatility-regime classifier using normalized ATR and return distribution.
    Lightweight statistical method - no pandas/sklearn needed.
    """

    def classify(self, fv: FeatureVector) -> Dict[str, float]:
        """Classify based on volatility regime."""

        atr_pct = fv.atr_pct_rank
        range_pct = fv.range_pct_rank
        vol_z = fv.atr_z
        boll_w = fv.bollinger_width

        has_direction = abs(fv.momentum_z) > 0.5

        if atr_pct > 0.7 and has_direction:
            trending_score = 0.6 + (atr_pct - 0.7) * 0.5
            choppy_score = 0.1
        elif atr_pct > 0.7 and not has_direction:
            trending_score = 0.2
            choppy_score = 0.2
        elif atr_pct < 0.3:
            trending_score = 0.1
            choppy_score = 0.5 + (0.3 - atr_pct) * 0.5
        else:
            trending_score = 0.25 + (0.25 if has_direction else 0.0)
            choppy_score = 0.25 + (0.0 if has_direction else 0.15)

        mixed_score = max(0.0, 1.0 - trending_score - choppy_score)

        total = trending_score + choppy_score + mixed_score
        if total < 1e-10:
            return default_probabilities()

        return {
            TRENDING: trending_score / total,
            CHOPPY: choppy_score / total,
            MIXED: mixed_score / total,
        }
