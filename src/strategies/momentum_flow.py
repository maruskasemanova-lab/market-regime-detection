"""
Momentum Flow Strategy.

Follows directional moves only when aggressive order flow confirms the move.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_strategy import BaseStrategy, Regime, Signal, SignalType


class MomentumFlowStrategy(BaseStrategy):
    """Flow-confirmed momentum strategy for trend phases."""

    def __init__(
        self,
        min_signed_aggression: float = 0.08,
        min_directional_consistency: float = 0.52,
        min_imbalance: float = 0.03,
        min_sweep_intensity: float = 0.08,
        min_book_pressure: float = 0.0,
        min_confidence: float = 54.0,
        atr_stop_multiplier: float = 1.8,
        rr_ratio: float = 2.2,
        trailing_stop_pct: float = 0.9,
    ):
        super().__init__(
            name="MomentumFlow",
            regimes=[Regime.TRENDING, Regime.CHOPPY, Regime.MIXED],
        )
        self.min_signed_aggression = min_signed_aggression
        self.min_directional_consistency = min_directional_consistency
        self.min_imbalance = min_imbalance
        self.min_sweep_intensity = min_sweep_intensity
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
        if len(self.get_open_positions()) >= 2:
            return None

        flow = indicators.get("order_flow") or {}
        if not flow.get("has_l2_coverage", False):
            return None

        signed_aggr = float(flow.get("signed_aggression", 0.0) or 0.0)
        consistency = float(flow.get("directional_consistency", 0.0) or 0.0)
        imbalance = float(flow.get("imbalance_avg", 0.0) or 0.0)
        sweep_intensity = float(flow.get("sweep_intensity", 0.0) or 0.0)
        book_pressure = float(flow.get("book_pressure_avg", 0.0) or 0.0)
        book_pressure_trend = float(flow.get("book_pressure_trend", 0.0) or 0.0)
        price_change_pct = float(flow.get("price_change_pct", 0.0) or 0.0)
        delta_acceleration = float(flow.get("delta_acceleration", 0.0) or 0.0)

        atr_series = indicators.get("atr") or []
        atr_val = float(atr_series[-1]) if atr_series else max(current_price * 0.005, 0.01)

        long_trigger = (
            signed_aggr >= self.min_signed_aggression
            and consistency >= self.min_directional_consistency
            and imbalance >= self.min_imbalance
            and sweep_intensity >= self.min_sweep_intensity
            and book_pressure >= self.min_book_pressure
            and book_pressure_trend >= 0.0
        )
        short_trigger = (
            signed_aggr <= -self.min_signed_aggression
            and consistency >= self.min_directional_consistency
            and imbalance <= -self.min_imbalance
            and sweep_intensity >= self.min_sweep_intensity
            and book_pressure <= -self.min_book_pressure
            and book_pressure_trend <= 0.0
        )
        if not long_trigger and not short_trigger:
            return None

        aggression_score = self._clamp01(abs(signed_aggr) / max(self.min_signed_aggression * 2.0, 1e-6))
        consistency_score = self._clamp01(consistency)
        imbalance_score = self._clamp01(abs(imbalance) / max(self.min_imbalance * 3.0, 1e-6))
        sweep_score = self._clamp01(sweep_intensity / max(self.min_sweep_intensity * 3.0, 1e-6))
        book_score = self._clamp01(
            abs(book_pressure) / max(max(self.min_book_pressure, 0.02) * 3.0, 1e-6)
        )
        confidence = 100.0 * (
            0.30 * aggression_score
            + 0.24 * consistency_score
            + 0.20 * imbalance_score
            + 0.12 * sweep_score
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
            direction_label = "up"
        else:
            signal_type = SignalType.SELL
            stop_loss = current_price + (atr_val * self.atr_stop_multiplier)
            take_profit = self.calculate_take_profit(
                current_price, stop_loss, self.rr_ratio, side="short"
            )
            direction_label = "down"

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
                f"Flow momentum {direction_label}: aggression {signed_aggr:+.2f}, "
                f"consistency {consistency:.2f}, imbalance {imbalance:+.2f}, "
                f"sweeps {sweep_intensity:.2f}, book {book_pressure:+.2f}"
            ),
            metadata={
                "order_flow": {
                    "signed_aggression": signed_aggr,
                    "directional_consistency": consistency,
                    "imbalance_avg": imbalance,
                    "sweep_intensity": sweep_intensity,
                    "book_pressure_avg": book_pressure,
                    "book_pressure_trend": book_pressure_trend,
                    "price_change_pct": price_change_pct,
                    "delta_acceleration": delta_acceleration,
                }
            },
        )
        self.add_signal(signal)
        return signal

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "min_signed_aggression": self.min_signed_aggression,
                "min_directional_consistency": self.min_directional_consistency,
                "min_imbalance": self.min_imbalance,
                "min_sweep_intensity": self.min_sweep_intensity,
                "min_book_pressure": self.min_book_pressure,
                "min_confidence": self.min_confidence,
                "atr_stop_multiplier": self.atr_stop_multiplier,
                "rr_ratio": self.rr_ratio,
                "trailing_stop_pct": self.trailing_stop_pct,
            }
        )
        return base
