"""
Absorption Reversal Strategy.

Trades reversal when aggressive flow is absorbed at an extreme and price stops
advancing in the aggressor direction.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_strategy import BaseStrategy, Regime, Signal, SignalType


class AbsorptionReversalStrategy(BaseStrategy):
    """Flow-first reversal strategy for absorption regimes."""

    def __init__(
        self,
        min_absorption_rate: float = 0.55,
        min_divergence: float = 0.15,
        min_signed_aggression: float = 0.08,
        min_book_pressure: float = 0.05,
        min_price_extension_pct: float = 0.12,
        min_confidence: float = 62.0,
        atr_stop_multiplier: float = 0.9,  # Reduced from 1.4 for tighter SL
        rr_ratio: float = 1.8,
        trailing_stop_pct: float = 0.7,
    ):
        super().__init__(
            name="AbsorptionReversal",
            regimes=[Regime.TRENDING, Regime.CHOPPY, Regime.MIXED],
        )
        self._uses_l2_internally = True
        self.min_absorption_rate = min_absorption_rate
        self.min_divergence = min_divergence
        self.min_signed_aggression = min_signed_aggression
        self.min_book_pressure = min_book_pressure
        self.min_price_extension_pct = min_price_extension_pct
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
        signed_aggr = float(flow.get("signed_aggression", 0.0) or 0.0)
        book_pressure = float(flow.get("book_pressure_avg", 0.0) or 0.0)
        price_extension = float(flow.get("price_change_pct", 0.0) or 0.0)
        imbalance = float(flow.get("imbalance_avg", 0.0) or 0.0)
        consistency = float(flow.get("directional_consistency", 0.0) or 0.0)

        atr_series = indicators.get("atr") or []
        atr_val = float(atr_series[-1]) if atr_series else max(current_price * 0.004, 0.01)

        # BUY: aggressive selling got absorbed and downside extension stalls.
        long_trigger = (
            price_extension <= -self.min_price_extension_pct
            and absorption >= self.min_absorption_rate
            and divergence >= self.min_divergence
            and signed_aggr <= -self.min_signed_aggression
            and book_pressure >= self.min_book_pressure
            and imbalance >= -0.05
        )
        # SELL: aggressive buying got absorbed and upside extension stalls.
        short_trigger = (
            price_extension >= self.min_price_extension_pct
            and absorption >= self.min_absorption_rate
            and divergence <= -self.min_divergence
            and signed_aggr >= self.min_signed_aggression
            and book_pressure <= -self.min_book_pressure
            and imbalance <= 0.05
        )

        if not long_trigger and not short_trigger:
            return None

        absorption_score = self._clamp01(absorption / max(self.min_absorption_rate, 1e-6))
        divergence_score = self._clamp01(abs(divergence) / max(self.min_divergence * 2.0, 1e-6))
        aggression_score = self._clamp01(abs(signed_aggr) / max(self.min_signed_aggression * 2.0, 1e-6))
        consistency_score = self._clamp01(consistency)
        book_score = self._clamp01(abs(book_pressure) / max(self.min_book_pressure * 2.0, 1e-6))
        confidence = 100.0 * (
            0.30 * absorption_score
            + 0.22 * divergence_score
            + 0.22 * aggression_score
            + 0.10 * consistency_score
            + 0.16 * book_score
        )
        if confidence < self.min_confidence:
            return None

        if long_trigger:
            signal_type = SignalType.BUY
            stop_loss = current_price - (atr_val * self.atr_stop_multiplier)
            take_profit = self.calculate_take_profit(
                current_price, stop_loss, self.rr_ratio, side="long"
            )
            direction_label = "bullish"
        else:
            signal_type = SignalType.SELL
            stop_loss = current_price + (atr_val * self.atr_stop_multiplier)
            take_profit = self.calculate_take_profit(
                current_price, stop_loss, self.rr_ratio, side="short"
            )
            direction_label = "bearish"

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
                f"Absorption {absorption:.2f}, divergence {divergence:+.2f}, "
                f"signed aggression {signed_aggr:+.2f}, book {book_pressure:+.2f}, "
                f"consistency {consistency:.2f}"
            ),
            metadata={
                "flow_direction": direction_label,
                "order_flow": {
                    "absorption_rate": absorption,
                    "delta_price_divergence": divergence,
                    "signed_aggression": signed_aggr,
                    "book_pressure_avg": book_pressure,
                    "price_change_pct": price_extension,
                    "imbalance_avg": imbalance,
                    "directional_consistency": consistency,
                },
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
                "min_signed_aggression": self.min_signed_aggression,
                "min_book_pressure": self.min_book_pressure,
                "min_price_extension_pct": self.min_price_extension_pct,
                "min_confidence": self.min_confidence,
                "atr_stop_multiplier": self.atr_stop_multiplier,
                "rr_ratio": self.rr_ratio,
                "trailing_stop_pct": self.trailing_stop_pct,
            }
        )
        return base
