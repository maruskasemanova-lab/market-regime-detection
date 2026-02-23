"""
Scalp L2 Intrabar Strategy.

Fast scalp strategy that combines minute-level L2 flow with optional
1-second intrabar quote microstructure confirmation.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_strategy import BaseStrategy, Regime, Signal, SignalType


class ScalpL2IntrabarStrategy(BaseStrategy):
    """L2-first scalp strategy with optional 1-second intrabar confirmation."""

    def __init__(
        self,
        min_flow_score: float = 48.0,
        min_signed_aggression: float = 0.045,
        min_directional_consistency: float = 0.58,
        min_imbalance: float = 0.03,
        min_book_pressure: float = 0.02,
        min_participation_ratio: float = 0.05,
        min_flow_score_trend_3bar: float = -2.0,
        min_intrabar_move_pct: float = 0.035,
        min_intrabar_push_ratio: float = 0.12,
        min_intrabar_coverage_points: int = 4,
        min_intrabar_directional_consistency: float = 0.12,
        intrabar_eval_window_seconds: int = 5,
        min_intrabar_window_move_pct: float = 0.015,
        min_intrabar_window_push_ratio: float = 0.08,
        min_intrabar_window_directional_consistency: float = 0.08,
        max_intrabar_micro_volatility_bps: float = 18.0,
        max_intrabar_spread_bps: float = 8.0,
        spread_penalty_floor_bps: float = 4.0,
        spread_flow_score_penalty_per_bps: float = 0.45,
        min_round_trip_cost_bps: float = 6.5,
        spread_cost_multiplier: float = 1.1,
        min_reward_to_cost_ratio: float = 1.7,
        min_flow_signal_margin: float = 0.01,
        max_abs_price_extension_pct: float = 1.8,
        require_intrabar_confirmation: bool = False,
        no_intrabar_flow_buffer: float = 10.0,
        min_confidence: float = 55.0,
        atr_stop_multiplier: float = 0.66,
        min_stop_loss_pct: float = 0.05,
        rr_ratio: float = 1.35,
        trailing_stop_pct: float = 0.28,
    ):
        super().__init__(
            name="ScalpL2Intrabar",
            regimes=[Regime.TRENDING, Regime.CHOPPY, Regime.MIXED],
        )
        self._uses_l2_internally = True
        self.min_flow_score = min_flow_score
        self.min_signed_aggression = min_signed_aggression
        self.min_directional_consistency = min_directional_consistency
        self.min_imbalance = min_imbalance
        self.min_book_pressure = min_book_pressure
        self.min_participation_ratio = min_participation_ratio
        self.min_flow_score_trend_3bar = min_flow_score_trend_3bar
        self.min_intrabar_move_pct = min_intrabar_move_pct
        self.min_intrabar_push_ratio = min_intrabar_push_ratio
        self.min_intrabar_coverage_points = max(1, int(min_intrabar_coverage_points))
        self.min_intrabar_directional_consistency = min_intrabar_directional_consistency
        self.intrabar_eval_window_seconds = max(2, min(30, int(intrabar_eval_window_seconds)))
        self.min_intrabar_window_move_pct = max(0.0, float(min_intrabar_window_move_pct))
        self.min_intrabar_window_push_ratio = max(0.0, float(min_intrabar_window_push_ratio))
        self.min_intrabar_window_directional_consistency = max(
            0.0, min(1.0, float(min_intrabar_window_directional_consistency))
        )
        self.max_intrabar_micro_volatility_bps = max_intrabar_micro_volatility_bps
        self.max_intrabar_spread_bps = max_intrabar_spread_bps
        self.spread_penalty_floor_bps = spread_penalty_floor_bps
        self.spread_flow_score_penalty_per_bps = spread_flow_score_penalty_per_bps
        self.min_round_trip_cost_bps = min_round_trip_cost_bps
        self.spread_cost_multiplier = spread_cost_multiplier
        self.min_reward_to_cost_ratio = min_reward_to_cost_ratio
        self.min_flow_signal_margin = min_flow_signal_margin
        self.max_abs_price_extension_pct = max_abs_price_extension_pct
        self.require_intrabar_confirmation = require_intrabar_confirmation
        self.no_intrabar_flow_buffer = no_intrabar_flow_buffer
        self.min_confidence = min_confidence
        self.atr_stop_multiplier = atr_stop_multiplier
        self.min_stop_loss_pct = min_stop_loss_pct
        self.rr_ratio = rr_ratio
        self.trailing_stop_pct = trailing_stop_pct

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _flow_signal_flags(
        self,
        flow: Dict[str, Any],
        *,
        effective_min_flow_score: Optional[float] = None,
    ) -> tuple[bool, bool, float]:
        signed_aggr = float(flow.get("signed_aggression", 0.0) or 0.0)
        consistency = float(flow.get("directional_consistency", 0.0) or 0.0)
        imbalance = float(flow.get("imbalance_avg", 0.0) or 0.0)
        book_pressure = float(flow.get("book_pressure_avg", 0.0) or 0.0)
        flow_score = float(flow.get("flow_score", 0.0) or 0.0)
        min_flow_score = float(
            self.min_flow_score if effective_min_flow_score is None else effective_min_flow_score
        )
        direction_margin = signed_aggr + 0.75 * imbalance + 0.45 * book_pressure

        long_flow = (
            flow_score >= min_flow_score
            and signed_aggr >= self.min_signed_aggression
            and consistency >= self.min_directional_consistency
            and imbalance >= self.min_imbalance
            and book_pressure >= self.min_book_pressure
            and direction_margin >= self.min_flow_signal_margin
        )
        short_flow = (
            flow_score >= min_flow_score
            and signed_aggr <= -self.min_signed_aggression
            and consistency >= self.min_directional_consistency
            and imbalance <= -self.min_imbalance
            and book_pressure <= -self.min_book_pressure
            and direction_margin <= -self.min_flow_signal_margin
        )
        return long_flow, short_flow, direction_margin

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

        intrabar = indicators.get("intrabar_1s") or {}
        has_intrabar = bool(intrabar.get("has_intrabar_coverage", False))

        flow_score = float(flow.get("flow_score", 0.0) or 0.0)
        signed_aggr = float(flow.get("signed_aggression", 0.0) or 0.0)
        consistency = float(flow.get("directional_consistency", 0.0) or 0.0)
        imbalance = float(flow.get("imbalance_avg", 0.0) or 0.0)
        book_pressure = float(flow.get("book_pressure_avg", 0.0) or 0.0)
        participation_ratio = float(flow.get("participation_ratio", 0.0) or 0.0)
        sweep_intensity = float(flow.get("sweep_intensity", 0.0) or 0.0)
        price_change_pct = float(flow.get("price_change_pct", 0.0) or 0.0)
        flow_score_trend = float(flow.get("flow_score_trend_3bar", 0.0) or 0.0)

        intrabar_move_pct = float(intrabar.get("mid_move_pct", 0.0) or 0.0)
        intrabar_push_ratio = float(intrabar.get("push_ratio", 0.0) or 0.0)
        intrabar_dir_consistency = float(intrabar.get("directional_consistency", 0.0) or 0.0)
        intrabar_micro_volatility_bps = float(intrabar.get("micro_volatility_bps", 0.0) or 0.0)
        intrabar_spread_bps = float(intrabar.get("spread_bps_avg", 0.0) or 0.0)
        intrabar_points = int(intrabar.get("coverage_points", 0) or 0)
        window_eval_seconds = int(intrabar.get("window_eval_seconds", 0) or 0)
        window_long_move_pct = float(intrabar.get("window_long_move_pct", 0.0) or 0.0)
        window_long_push_ratio = float(intrabar.get("window_long_push_ratio", 0.0) or 0.0)
        window_long_directional_consistency = float(
            intrabar.get("window_long_directional_consistency", 0.0) or 0.0
        )
        window_short_move_pct = float(intrabar.get("window_short_move_pct", 0.0) or 0.0)
        window_short_push_ratio = float(intrabar.get("window_short_push_ratio", 0.0) or 0.0)
        window_short_directional_consistency = float(
            intrabar.get("window_short_directional_consistency", 0.0) or 0.0
        )

        if participation_ratio < self.min_participation_ratio:
            return None
        if flow_score_trend < self.min_flow_score_trend_3bar:
            return None
        if self.max_abs_price_extension_pct > 0.0 and abs(price_change_pct) > self.max_abs_price_extension_pct:
            return None

        effective_min_flow_score = self.min_flow_score
        if has_intrabar and self.spread_flow_score_penalty_per_bps > 0.0:
            spread_excess_bps = max(0.0, intrabar_spread_bps - self.spread_penalty_floor_bps)
            effective_min_flow_score += spread_excess_bps * self.spread_flow_score_penalty_per_bps

        long_flow, short_flow, direction_margin = self._flow_signal_flags(
            flow,
            effective_min_flow_score=effective_min_flow_score,
        )
        if not long_flow and not short_flow:
            return None

        if has_intrabar:
            window_ready = (
                window_eval_seconds > 0
                and window_eval_seconds == self.intrabar_eval_window_seconds
            )
            window_directional_consistency_ok = (
                window_ready
                and (
                    window_long_directional_consistency
                    >= self.min_intrabar_window_directional_consistency
                    or window_short_directional_consistency
                    >= self.min_intrabar_window_directional_consistency
                )
            )
            if intrabar_points < self.min_intrabar_coverage_points:
                return None
            if (
                intrabar_dir_consistency < self.min_intrabar_directional_consistency
                and not window_directional_consistency_ok
            ):
                return None
            if (
                self.max_intrabar_micro_volatility_bps > 0.0
                and intrabar_micro_volatility_bps > self.max_intrabar_micro_volatility_bps
            ):
                return None
            intrabar_spread_ok = (
                intrabar_spread_bps <= self.max_intrabar_spread_bps
                or self.max_intrabar_spread_bps <= 0.0
            )
            long_intrabar_minute = (
                intrabar_spread_ok
                and intrabar_move_pct >= self.min_intrabar_move_pct
                and intrabar_push_ratio >= self.min_intrabar_push_ratio
            )
            short_intrabar_minute = (
                intrabar_spread_ok
                and intrabar_move_pct <= -self.min_intrabar_move_pct
                and intrabar_push_ratio <= -self.min_intrabar_push_ratio
            )
            long_intrabar_window = (
                window_ready
                and intrabar_spread_ok
                and window_long_move_pct >= self.min_intrabar_window_move_pct
                and window_long_push_ratio >= self.min_intrabar_window_push_ratio
                and window_long_directional_consistency
                >= self.min_intrabar_window_directional_consistency
            )
            short_intrabar_window = (
                window_ready
                and intrabar_spread_ok
                and window_short_move_pct <= -self.min_intrabar_window_move_pct
                and window_short_push_ratio <= -self.min_intrabar_window_push_ratio
                and window_short_directional_consistency
                >= self.min_intrabar_window_directional_consistency
            )
            long_intrabar = long_intrabar_minute or long_intrabar_window
            short_intrabar = short_intrabar_minute or short_intrabar_window
        else:
            if self.require_intrabar_confirmation:
                return None
            fallback_threshold = effective_min_flow_score + self.no_intrabar_flow_buffer
            long_intrabar = flow_score >= fallback_threshold
            short_intrabar = flow_score >= fallback_threshold
            long_intrabar_minute = False
            short_intrabar_minute = False
            long_intrabar_window = False
            short_intrabar_window = False

        long_trigger = long_flow and long_intrabar
        short_trigger = short_flow and short_intrabar
        if not long_trigger and not short_trigger:
            return None

        if long_trigger:
            directional_intrabar_move_pct = max(intrabar_move_pct, window_long_move_pct)
            directional_intrabar_push_ratio = max(intrabar_push_ratio, window_long_push_ratio)
            directional_intrabar_consistency = max(
                intrabar_dir_consistency,
                window_long_directional_consistency,
            )
            intrabar_trigger_source = (
                "window_5s" if long_intrabar_window and not long_intrabar_minute else "minute_1m"
            )
        else:
            directional_intrabar_move_pct = min(intrabar_move_pct, window_short_move_pct)
            directional_intrabar_push_ratio = min(intrabar_push_ratio, window_short_push_ratio)
            directional_intrabar_consistency = max(
                intrabar_dir_consistency,
                window_short_directional_consistency,
            )
            intrabar_trigger_source = (
                "window_5s" if short_intrabar_window and not short_intrabar_minute else "minute_1m"
            )

        atr_series = indicators.get("atr") or []
        atr_val = float(atr_series[-1]) if atr_series else max(current_price * 0.0025, 0.01)
        effective_atr_stop_multiplier = (
            self.get_effective_atr_stop_multiplier() or self.atr_stop_multiplier
        )
        effective_min_stop_loss_pct = (
            self.get_effective_min_stop_loss_pct() or self.min_stop_loss_pct
        )
        effective_rr_ratio = self.get_effective_rr_ratio() or self.rr_ratio
        stop_distance = max(
            atr_val * effective_atr_stop_multiplier,
            current_price * (effective_min_stop_loss_pct / 100.0),
        )

        aggression_score = self._clamp01(abs(signed_aggr) / max(self.min_signed_aggression * 2.5, 1e-6))
        consistency_score = self._clamp01(consistency)
        imbalance_score = self._clamp01(abs(imbalance) / max(self.min_imbalance * 3.0, 1e-6))
        flow_score_norm = self._clamp01(flow_score / 100.0)
        book_score = self._clamp01(abs(book_pressure) / max(self.min_book_pressure * 4.0, 1e-6))
        participation_score = self._clamp01(
            participation_ratio / max(self.min_participation_ratio * 2.0, 1e-6)
        )
        trend_score = self._clamp01((flow_score_trend - self.min_flow_score_trend_3bar + 4.0) / 8.0)
        flow_conf = (
            0.20 * aggression_score
            + 0.18 * consistency_score
            + 0.17 * imbalance_score
            + 0.18 * flow_score_norm
            + 0.13 * book_score
            + 0.08 * participation_score
            + 0.06 * trend_score
        )

        intrabar_conf = 0.0
        if has_intrabar:
            move_score = self._clamp01(
                abs(directional_intrabar_move_pct) / max(self.min_intrabar_move_pct * 4.0, 1e-6)
            )
            push_score = self._clamp01(
                abs(directional_intrabar_push_ratio) / max(self.min_intrabar_push_ratio * 3.0, 1e-6)
            )
            spread_score = self._clamp01(
                1.0 - (intrabar_spread_bps / max(self.max_intrabar_spread_bps * 2.0, 1e-6))
            )
            coverage_score = self._clamp01(
                intrabar_points / max(float(self.min_intrabar_coverage_points * 2), 1.0)
            )
            volatility_score = (
                self._clamp01(
                    1.0
                    - (
                        intrabar_micro_volatility_bps
                        / max(self.max_intrabar_micro_volatility_bps * 1.5, 1e-6)
                    )
                )
                if self.max_intrabar_micro_volatility_bps > 0.0
                else 1.0
            )
            intrabar_conf = (
                0.28 * move_score
                + 0.24 * push_score
                + 0.16 * self._clamp01(directional_intrabar_consistency)
                + 0.14 * spread_score
                + 0.10 * volatility_score
                + 0.08 * coverage_score
            )

        combined_conf = (
            (0.65 * flow_conf + 0.35 * intrabar_conf)
            if has_intrabar
            else (0.90 * flow_conf)
        )
        confidence = min(100.0, 100.0 * combined_conf)
        if confidence < self.min_confidence:
            return None

        if long_trigger:
            signal_type = SignalType.BUY
            stop_loss = current_price - stop_distance
            take_profit = self.calculate_take_profit(
                current_price, stop_loss, effective_rr_ratio, side="long"
            )
            direction_label = "long_scalp"
        else:
            signal_type = SignalType.SELL
            stop_loss = current_price + stop_distance
            take_profit = self.calculate_take_profit(
                current_price, stop_loss, effective_rr_ratio, side="short"
            )
            direction_label = "short_scalp"

        expected_reward_bps = abs(take_profit - current_price) / max(current_price, 1e-6) * 10000.0
        expected_risk_bps = abs(current_price - stop_loss) / max(current_price, 1e-6) * 10000.0
        spread_cost_bps = abs(intrabar_spread_bps) * max(0.0, self.spread_cost_multiplier)
        estimated_round_trip_cost_bps = max(self.min_round_trip_cost_bps, spread_cost_bps)
        min_required_reward_bps = estimated_round_trip_cost_bps * max(1.0, self.min_reward_to_cost_ratio)
        if expected_reward_bps < min_required_reward_bps:
            return None

        signal = Signal(
            strategy_name=self.name,
            signal_type=signal_type,
            price=current_price,
            timestamp=timestamp,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=True,
            trailing_stop_pct=self.get_effective_trailing_stop_pct(),
            reasoning=(
                f"{direction_label}: flow {flow_score:.1f}/{effective_min_flow_score:.1f}, "
                f"signed {signed_aggr:+.3f}, margin {direction_margin:+.3f}, "
                f"intrabar_move {directional_intrabar_move_pct:+.3f}% ({intrabar_trigger_source}), "
                f"reward {expected_reward_bps:.1f}bps vs cost {estimated_round_trip_cost_bps:.1f}bps"
            ),
            metadata={
                "order_flow": {
                    "flow_score": flow_score,
                    "effective_min_flow_score": effective_min_flow_score,
                    "signed_aggression": signed_aggr,
                    "directional_consistency": consistency,
                    "imbalance_avg": imbalance,
                    "book_pressure_avg": book_pressure,
                    "participation_ratio": participation_ratio,
                    "flow_score_trend_3bar": flow_score_trend,
                    "direction_margin": direction_margin,
                    "sweep_intensity": sweep_intensity,
                    "price_change_pct": price_change_pct,
                },
                "intrabar_1s": {
                    "has_intrabar_coverage": has_intrabar,
                    "coverage_points": intrabar_points,
                    "mid_move_pct": intrabar_move_pct,
                    "push_ratio": intrabar_push_ratio,
                    "directional_consistency": intrabar_dir_consistency,
                    "micro_volatility_bps": intrabar_micro_volatility_bps,
                    "spread_bps_avg": intrabar_spread_bps,
                    "window_eval_seconds": window_eval_seconds,
                    "window_long_move_pct": window_long_move_pct,
                    "window_long_push_ratio": window_long_push_ratio,
                    "window_long_directional_consistency": window_long_directional_consistency,
                    "window_short_move_pct": window_short_move_pct,
                    "window_short_push_ratio": window_short_push_ratio,
                    "window_short_directional_consistency": window_short_directional_consistency,
                    "minute_long_ok": long_intrabar_minute,
                    "minute_short_ok": short_intrabar_minute,
                    "window_long_ok": long_intrabar_window,
                    "window_short_ok": short_intrabar_window,
                    "trigger_source": intrabar_trigger_source,
                },
                "cost_guard": {
                    "expected_reward_bps": expected_reward_bps,
                    "expected_risk_bps": expected_risk_bps,
                    "estimated_round_trip_cost_bps": estimated_round_trip_cost_bps,
                    "min_required_reward_bps": min_required_reward_bps,
                    "min_round_trip_cost_bps": self.min_round_trip_cost_bps,
                    "spread_cost_multiplier": self.spread_cost_multiplier,
                    "min_reward_to_cost_ratio": self.min_reward_to_cost_ratio,
                },
            },
        )
        self.add_signal(signal)
        return signal

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "min_flow_score": self.min_flow_score,
                "min_signed_aggression": self.min_signed_aggression,
                "min_directional_consistency": self.min_directional_consistency,
                "min_imbalance": self.min_imbalance,
                "min_book_pressure": self.min_book_pressure,
                "min_participation_ratio": self.min_participation_ratio,
                "min_flow_score_trend_3bar": self.min_flow_score_trend_3bar,
                "min_intrabar_move_pct": self.min_intrabar_move_pct,
                "min_intrabar_push_ratio": self.min_intrabar_push_ratio,
                "min_intrabar_coverage_points": self.min_intrabar_coverage_points,
                "min_intrabar_directional_consistency": self.min_intrabar_directional_consistency,
                "intrabar_eval_window_seconds": self.intrabar_eval_window_seconds,
                "min_intrabar_window_move_pct": self.min_intrabar_window_move_pct,
                "min_intrabar_window_push_ratio": self.min_intrabar_window_push_ratio,
                "min_intrabar_window_directional_consistency": self.min_intrabar_window_directional_consistency,
                "max_intrabar_micro_volatility_bps": self.max_intrabar_micro_volatility_bps,
                "max_intrabar_spread_bps": self.max_intrabar_spread_bps,
                "spread_penalty_floor_bps": self.spread_penalty_floor_bps,
                "spread_flow_score_penalty_per_bps": self.spread_flow_score_penalty_per_bps,
                "min_round_trip_cost_bps": self.min_round_trip_cost_bps,
                "spread_cost_multiplier": self.spread_cost_multiplier,
                "min_reward_to_cost_ratio": self.min_reward_to_cost_ratio,
                "min_flow_signal_margin": self.min_flow_signal_margin,
                "max_abs_price_extension_pct": self.max_abs_price_extension_pct,
                "require_intrabar_confirmation": self.require_intrabar_confirmation,
                "no_intrabar_flow_buffer": self.no_intrabar_flow_buffer,
                "min_confidence": self.min_confidence,
                "atr_stop_multiplier": self.atr_stop_multiplier,
                "min_stop_loss_pct": self.min_stop_loss_pct,
                "rr_ratio": self.rr_ratio,
                "trailing_stop_pct": self.trailing_stop_pct,
            }
        )
        return base
