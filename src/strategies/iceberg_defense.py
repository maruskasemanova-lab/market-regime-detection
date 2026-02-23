"""
Iceberg Defense Strategy.

Trades in the direction of detected iceberg order accumulation.
Iceberg orders are large hidden institutional orders that repeatedly refill
at a price level, revealing intent to defend support/resistance.

This strategy is orthogonal to existing flow strategies (which use
aggression/divergence/absorption) because it reads *hidden* liquidity
rather than visible aggression.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_strategy import BaseStrategy, Regime, Signal, SignalType


class IcebergDefenseStrategy(BaseStrategy):
    """Institutional iceberg-based strategy.

    Enters when iceberg orders accumulate on one side, confirming institutional
    intent to defend a price level. Confirmations: book pressure alignment and
    no extreme opposing flow.
    """

    def __init__(
        self,
        min_iceberg_bias: float = 0.5,
        min_book_pressure: float = 0.0,
        max_opposing_aggression: float = 0.03,
        min_absorption_rate: float = 0.30,
        min_confidence: float = 60.0,
        atr_stop_multiplier: float = 1.2,
        rr_ratio: float = 2.0,
        trailing_stop_pct: float = 0.8,
    ):
        super().__init__(
            name="IcebergDefense",
            regimes=[Regime.TRENDING, Regime.CHOPPY, Regime.MIXED],
        )
        self._uses_l2_internally = True
        self.min_iceberg_bias = min_iceberg_bias
        self.min_book_pressure = min_book_pressure
        self.max_opposing_aggression = max_opposing_aggression
        self.min_absorption_rate = min_absorption_rate
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

        iceberg_bias = float(flow.get("iceberg_bias", 0.0) or 0.0)
        book_pressure = float(flow.get("book_pressure_avg", 0.0) or 0.0)
        signed_aggr = float(flow.get("signed_aggression", 0.0) or 0.0)
        absorption = float(flow.get("absorption_rate", 0.0) or 0.0)
        consistency = float(flow.get("directional_consistency", 0.0) or 0.0)
        price_change_pct = float(flow.get("price_change_pct", 0.0) or 0.0)

        atr_series = indicators.get("atr") or []
        atr_val = float(atr_series[-1]) if atr_series else max(current_price * 0.005, 0.01)
        effective_atr_stop_multiplier = (
            self.get_effective_atr_stop_multiplier() or self.atr_stop_multiplier
        )
        effective_rr_ratio = self.get_effective_rr_ratio() or self.rr_ratio

        # BUY: buy-side icebergs dominate → institutions defending support
        long_trigger = (
            iceberg_bias >= self.min_iceberg_bias
            and book_pressure >= self.min_book_pressure
            and signed_aggr >= -self.max_opposing_aggression  # not strong selling against
            and absorption >= self.min_absorption_rate
            and price_change_pct <= 0.30  # not already extended up
        )

        # SELL: sell-side icebergs dominate → institutions defending resistance
        short_trigger = (
            iceberg_bias <= -self.min_iceberg_bias
            and book_pressure <= -self.min_book_pressure
            and signed_aggr <= self.max_opposing_aggression  # not strong buying against
            and absorption >= self.min_absorption_rate
            and price_change_pct >= -0.30  # not already extended down
        )

        if not long_trigger and not short_trigger:
            return None

        # Confidence from multiple factors
        iceberg_score = self._clamp01(
            abs(iceberg_bias) / max(self.min_iceberg_bias * 3.0, 1e-6)
        )
        book_score = self._clamp01(
            abs(book_pressure) / max(max(self.min_book_pressure, 0.02) * 3.0, 1e-6)
        )
        absorption_score = self._clamp01(absorption)
        consistency_score = self._clamp01(consistency)
        # Bonus for low opposing aggression (clean defense)
        opposing = abs(signed_aggr) if (
            (long_trigger and signed_aggr < 0) or
            (short_trigger and signed_aggr > 0)
        ) else 0.0
        clean_entry_score = self._clamp01(1.0 - opposing / max(self.max_opposing_aggression * 3.0, 1e-6))

        confidence = 100.0 * (
            0.35 * iceberg_score        # Primary: iceberg strength
            + 0.20 * book_score          # Confirming book pressure
            + 0.15 * absorption_score    # Market absorbing opposing flow
            + 0.15 * consistency_score   # Directional consistency
            + 0.15 * clean_entry_score   # Lack of opposing aggression
        )

        if confidence < self.min_confidence:
            return None

        if long_trigger:
            signal_type = SignalType.BUY
            stop_loss = current_price - (atr_val * effective_atr_stop_multiplier)
            take_profit = self.calculate_take_profit(
                current_price, stop_loss, effective_rr_ratio, side="long"
            )
            direction_label = "buy_defense"
        else:
            signal_type = SignalType.SELL
            stop_loss = current_price + (atr_val * effective_atr_stop_multiplier)
            take_profit = self.calculate_take_profit(
                current_price, stop_loss, effective_rr_ratio, side="short"
            )
            direction_label = "sell_defense"

        signal = Signal(
            strategy_name=self.name,
            signal_type=signal_type,
            price=current_price,
            timestamp=timestamp,
            confidence=min(confidence, 100.0),
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=True,
            trailing_stop_pct=self.get_effective_trailing_stop_pct(),
            reasoning=(
                f"Iceberg {direction_label}: bias {iceberg_bias:+.2f}, "
                f"book {book_pressure:+.2f}, aggr {signed_aggr:+.2f}, "
                f"absorption {absorption:.2f}"
            ),
            metadata={
                "order_flow": {
                    "iceberg_bias": iceberg_bias,
                    "book_pressure_avg": book_pressure,
                    "signed_aggression": signed_aggr,
                    "absorption_rate": absorption,
                    "directional_consistency": consistency,
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
                "min_iceberg_bias": self.min_iceberg_bias,
                "min_book_pressure": self.min_book_pressure,
                "max_opposing_aggression": self.max_opposing_aggression,
                "min_absorption_rate": self.min_absorption_rate,
                "min_confidence": self.min_confidence,
                "atr_stop_multiplier": self.atr_stop_multiplier,
                "rr_ratio": self.rr_ratio,
                "trailing_stop_pct": self.trailing_stop_pct,
            }
        )
        return base
