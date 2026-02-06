"""
Exhaustion Fade Strategy.

Fades one-sided flow extremes when delta diverges from price progress.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_strategy import BaseStrategy, Regime, Signal, SignalType


class ExhaustionFadeStrategy(BaseStrategy):
    """Reversal strategy for exhausted aggressive flow."""

    def __init__(
        self,
        min_absorption_rate: float = 0.50,
        min_divergence: float = 0.20,
        min_delta_zscore: float = 1.4,
        max_sweep_intensity: float = 0.8,
        min_book_pressure: float = 0.0,
        min_confidence: float = 63.0,
        atr_stop_multiplier: float = 1.3,
        rr_ratio: float = 1.7,
        trailing_stop_pct: float = 0.6,
    ):
        super().__init__(
            name="ExhaustionFade",
            regimes=[Regime.TRENDING, Regime.CHOPPY, Regime.MIXED],
        )
        self.min_absorption_rate = min_absorption_rate
        self.min_divergence = min_divergence
        self.min_delta_zscore = min_delta_zscore
        self.max_sweep_intensity = max_sweep_intensity
        self.min_book_pressure = min_book_pressure
        self.min_confidence = min_confidence
        self.atr_stop_multiplier = atr_stop_multiplier
        self.rr_ratio = rr_ratio
        self.trailing_stop_pct = trailing_stop_pct

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def generate_signal(
        self,
        current_price: float,
        ohlcv: Dict[str, List[float]],
        indicators: Dict[str, Any],
        regime: Regime,
        timestamp: datetime,
    ) -> Optional[Signal]:
        if not self.is_allowed_in_regime(regime):
            return None
        if len(self.get_open_positions()) > 0:
            return None

        flow = indicators.get("order_flow") or {}
        if not flow.get("has_l2_coverage", False):
            return None

        absorption = float(flow.get("absorption_rate", 0.0) or 0.0)
        divergence = float(flow.get("delta_price_divergence", 0.0) or 0.0)
        delta_z = float(flow.get("delta_zscore", 0.0) or 0.0)
        sweep_intensity = float(flow.get("sweep_intensity", 0.0) or 0.0)
        signed_aggr = float(flow.get("signed_aggression", 0.0) or 0.0)
        book_pressure = float(flow.get("book_pressure_avg", 0.0) or 0.0)
        price_change_pct = float(flow.get("price_change_pct", 0.0) or 0.0)

        atr_series = indicators.get("atr") or []
        atr_val = float(atr_series[-1]) if atr_series else max(current_price * 0.004, 0.01)

        # BUY: selling became extreme, but price no longer follows lower (bullish divergence).
        long_trigger = (
            delta_z <= -self.min_delta_zscore
            and divergence >= self.min_divergence
            and absorption >= self.min_absorption_rate
            and signed_aggr <= -0.04
            and book_pressure >= self.min_book_pressure
            and sweep_intensity <= self.max_sweep_intensity
            and price_change_pct <= 0.0
        )
        # SELL: buying became extreme, but price no longer follows higher (bearish divergence).
        short_trigger = (
            delta_z >= self.min_delta_zscore
            and divergence <= -self.min_divergence
            and absorption >= self.min_absorption_rate
            and signed_aggr >= 0.04
            and book_pressure <= -self.min_book_pressure
            and sweep_intensity <= self.max_sweep_intensity
            and price_change_pct >= 0.0
        )
        if not long_trigger and not short_trigger:
            return None

        exhaustion_score = self._clamp01(abs(delta_z) / max(self.min_delta_zscore * 2.0, 1e-6))
        divergence_score = self._clamp01(abs(divergence) / max(self.min_divergence * 2.0, 1e-6))
        absorption_score = self._clamp01(absorption / max(self.min_absorption_rate, 1e-6))
        sweep_penalty = self._clamp01(sweep_intensity / max(self.max_sweep_intensity, 1e-6))
        book_score = self._clamp01(
            abs(book_pressure) / max(max(self.min_book_pressure, 0.02) * 3.0, 1e-6)
        )
        confidence = 100.0 * (
            0.30 * exhaustion_score
            + 0.26 * divergence_score
            + 0.20 * absorption_score
            + 0.10 * (1.0 - sweep_penalty)
            + 0.14 * book_score
        )
        if confidence < self.min_confidence:
            return None

        if long_trigger:
            signal_type = SignalType.BUY
            stop_loss = current_price - (atr_val * self.atr_stop_multiplier)
            take_profit = self.calculate_take_profit(
                current_price, stop_loss, self.rr_ratio, side="long"
            )
            direction_label = "bullish_fade"
        else:
            signal_type = SignalType.SELL
            stop_loss = current_price + (atr_val * self.atr_stop_multiplier)
            take_profit = self.calculate_take_profit(
                current_price, stop_loss, self.rr_ratio, side="short"
            )
            direction_label = "bearish_fade"

        signal = Signal(
            strategy_name=self.name,
            signal_type=signal_type,
            price=current_price,
            timestamp=timestamp,
            confidence=min(confidence, 100.0),
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=True,
            trailing_stop_pct=self.trailing_stop_pct,
            reasoning=(
                f"Exhaustion {direction_label}: delta_z {delta_z:+.2f}, "
                f"divergence {divergence:+.2f}, absorption {absorption:.2f}, "
                f"sweep {sweep_intensity:.2f}, book {book_pressure:+.2f}"
            ),
            metadata={
                "order_flow": {
                    "absorption_rate": absorption,
                    "delta_price_divergence": divergence,
                    "delta_zscore": delta_z,
                    "signed_aggression": signed_aggr,
                    "book_pressure_avg": book_pressure,
                    "sweep_intensity": sweep_intensity,
                    "price_change_pct": price_change_pct,
                }
            },
        )
        self.add_signal(signal)
        return signal

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "min_absorption_rate": self.min_absorption_rate,
                "min_divergence": self.min_divergence,
                "min_delta_zscore": self.min_delta_zscore,
                "max_sweep_intensity": self.max_sweep_intensity,
                "min_book_pressure": self.min_book_pressure,
                "min_confidence": self.min_confidence,
                "atr_stop_multiplier": self.atr_stop_multiplier,
                "rr_ratio": self.rr_ratio,
                "trailing_stop_pct": self.trailing_stop_pct,
            }
        )
        return base
