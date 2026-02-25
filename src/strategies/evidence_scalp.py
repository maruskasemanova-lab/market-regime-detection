"""
Evidence Micro-Scalp Strategy.

Designed exclusively for 5-second checkpoints, leveraging live order flow (L2) 
and options flow (TCBBO) alongside high-frequency price action (push ratio)
to execute rapid, high-quantity scalps with tight stops.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .base_strategy import BaseStrategy, Regime, Signal, SignalType


class EvidenceScalpStrategy(BaseStrategy):
    """
    High-frequency micro-scalping strategy.
    Fires on strong 5s momentum supported by L2 flow OR options sweeps, 
    or fade setups where L2 limit orders aggressively absorb 5s momentum.
    """

    def __init__(
        self,
        # ── 5s Price Action ──
        min_5s_move_pct: float = 0.04,
        min_5s_push_ratio: float = 0.75,
        # ── Order Flow (L2) ──
        min_l2_aggression: float = 0.08,
        min_book_pressure: float = 0.08,
        # ── Options Flow (TCBBO) ──
        min_tcbbo_premium: float = 25000.0,
        # ── Risk / Exit Rules ──
        base_stop_loss_pct: float = 0.15,
        target_rr_ratio: float = 1.5,
        trailing_stop_activation_pct: float = 0.08,  # very tight trailing start
        # ── Cost / Frequency ──
        min_round_trip_cost_bps: float = 3.0,
        min_signal_interval_seconds: int = 5,  # can fire every next bar
    ):
        super().__init__(
            name="EvidenceScalp",
            # Works across all regimes; flow context handles the rest
            regimes=[Regime.TRENDING, Regime.CHOPPY, Regime.MIXED],
        )
        self._uses_l2_internally = True

        self.min_5s_move_pct = max(0.005, float(min_5s_move_pct))
        self.min_5s_push_ratio = max(0.2, float(min_5s_push_ratio))

        self.min_l2_aggression = max(0.01, float(min_l2_aggression))
        self.min_book_pressure = max(0.01, float(min_book_pressure))
        self.min_tcbbo_premium = float(min_tcbbo_premium)

        self.base_stop_loss_pct = float(base_stop_loss_pct)
        self.target_rr_ratio = float(target_rr_ratio)
        self.trailing_stop_activation_pct = float(trailing_stop_activation_pct)

        self.min_round_trip_cost_bps = float(min_round_trip_cost_bps)
        self.min_signal_interval_seconds = max(0, int(min_signal_interval_seconds))

        self._last_signal_ts: Optional[datetime] = None

    def _evaluate_scalp_setup(
        self, current_price: float, indicators: Dict[str, Any]
    ) -> Tuple[Optional[str], float, str]:
        """
        Returns (direction, confidence, reasoning) or (None, 0.0, "").
        """
        # ── Extract 5s Snapshot ──
        intra_1s = indicators.get("intrabar_1s") or {}
        has_intrabar = bool(intra_1s.get("has_intrabar_coverage", False))
        
        move_pct = float(intra_1s.get("mid_move_pct", 0.0) or 0.0)
        push_ratio = float(intra_1s.get("push_ratio", 0.0) or 0.0)
        
        # ── Extract Order Flow (L2) ──
        flow = indicators.get("order_flow") or {}
        has_l2 = bool(flow.get("has_l2_coverage", False))
        signed_aggr = float(flow.get("signed_aggression", 0.0) or 0.0)
        book_press = float(flow.get("book_pressure_avg", 0.0) or 0.0)
        delta_div = float(flow.get("delta_price_divergence", 0.0) or 0.0)
        
        # ── Extract Options Flow (TCBBO) ──
        tcbbo = indicators.get("tcbbo") or {}
        has_options_flow = bool(tcbbo and tcbbo.get("is_valid", False))
        net_premium = float(tcbbo.get("net_premium", 0.0) or 0.0)
        
        if not has_intrabar:
            return None, 0.0, "missing_5s_intrabar_data"

        strong_push_up = move_pct >= self.min_5s_move_pct and push_ratio >= self.min_5s_push_ratio
        strong_push_down = move_pct <= -self.min_5s_move_pct and push_ratio <= -self.min_5s_push_ratio
        
        l2_buy_pressure = signed_aggr >= self.min_l2_aggression or book_press >= self.min_book_pressure
        l2_sell_pressure = signed_aggr <= -self.min_l2_aggression or book_press <= -self.min_book_pressure
        
        options_buy = net_premium >= self.min_tcbbo_premium
        options_sell = net_premium <= -self.min_tcbbo_premium

        # SETUP 1: Trend alignment
        # 5s price pushing up + (L2 buying OR Options sweeping calls)
        if strong_push_up and (l2_buy_pressure or options_buy):
            score = 60.0 + (abs(push_ratio)*10.0) + (10.0 if l2_buy_pressure else 0.0) + (10.0 if options_buy else 0.0)
            reason = f"momentum_long: mv={move_pct:.3f}%, pr={push_ratio:.2f}, L2_aggr={signed_aggr:.2f}, Opt={net_premium/1000:.0f}k"
            return "bullish", min(95.0, score), reason

        if strong_push_down and (l2_sell_pressure or options_sell):
            score = 60.0 + (abs(push_ratio)*10.0) + (10.0 if l2_sell_pressure else 0.0) + (10.0 if options_sell else 0.0)
            reason = f"momentum_short: mv={move_pct:.3f}%, pr={push_ratio:.2f}, L2_aggr={signed_aggr:.2f}, Opt={net_premium/1000:.0f}k"
            return "bearish", min(95.0, score), reason

        # SETUP 2: Fade Absorption
        # Price pushing hard ONE way, but L2 order flow is aggressively absorbing it (opposing limit orders winning)
        # Delta divergence helps us see this (price up, delta deeply negative)
        if strong_push_up and signed_aggr <= -self.min_l2_aggression and delta_div < -self.min_l2_aggression * 2:
            reason = f"fade_absorption_short: pushed_up={move_pct:.3f}%, but heavy limit supply aggr={signed_aggr:.2f}, div={delta_div:.2f}"
            return "bearish", 75.0, reason
            
        if strong_push_down and signed_aggr >= self.min_l2_aggression and delta_div > self.min_l2_aggression * 2:
            reason = f"fade_absorption_long: pushed_down={move_pct:.3f}%, but heavy limit bids aggr={signed_aggr:.2f}, div={delta_div:.2f}"
            return "bullish", 75.0, reason

        return None, 0.0, "no_alignment"

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

        # Cooldown prevents immediate rapid-fire in the same direction
        if self._last_signal_ts is not None and self.min_signal_interval_seconds > 0:
            elapsed = (timestamp - self._last_signal_ts).total_seconds()
            if elapsed < self.min_signal_interval_seconds:
                return None

        direction, confidence, reason = self._evaluate_scalp_setup(current_price, indicators)
        if not direction:
            return None

        # ── stop / target ──
        # Since this is a strict micro-scalper, we use the tightest possible 
        # stops while dodging noise.
        eff_stop = self.get_effective_min_stop_loss_pct() or self.base_stop_loss_pct
        eff_rr = self.get_effective_rr_ratio() or self.target_rr_ratio
        
        stop_distance = current_price * (eff_stop / 100.0)

        if direction == "bullish":
            signal_type = SignalType.BUY
            stop_loss = current_price - stop_distance
            take_profit = self.calculate_take_profit(current_price, stop_loss, eff_rr, side="long")
        else:
            signal_type = SignalType.SELL
            stop_loss = current_price + stop_distance
            take_profit = self.calculate_take_profit(current_price, stop_loss, eff_rr, side="short")

        # ── Scalping Cost Guard ──
        reward_bps = abs(take_profit - current_price) / max(current_price, 1e-6) * 10000.0
        intra = indicators.get("intrabar_1s") or {}
        spread_bps = float(intra.get("spread_bps_avg", 0.0) or 0.0)
        
        # We need reward to simply be larger than the round-trip cost explicitly
        est_cost_bps = max(self.min_round_trip_cost_bps, spread_bps * 1.2)
        if reward_bps < est_cost_bps:
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
            trailing_stop_pct=self.get_effective_trailing_stop_pct() or self.trailing_stop_activation_pct,
            reasoning=reason,
            metadata={
                "cost_guard": {
                    "reward_bps": round(reward_bps, 1),
                    "estimated_cost_bps": round(est_cost_bps, 1),
                },
                "scalp_trigger": reason
            },
        )

        self.add_signal(signal)
        self._last_signal_ts = timestamp
        return signal

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "min_5s_move_pct": self.min_5s_move_pct,
            "min_5s_push_ratio": self.min_5s_push_ratio,
            "min_l2_aggression": self.min_l2_aggression,
            "min_book_pressure": self.min_book_pressure,
            "min_tcbbo_premium": self.min_tcbbo_premium,
            "base_stop_loss_pct": self.base_stop_loss_pct,
            "target_rr_ratio": self.target_rr_ratio,
            "trailing_stop_activation_pct": self.trailing_stop_activation_pct,
            "min_round_trip_cost_bps": self.min_round_trip_cost_bps,
            "min_signal_interval_seconds": self.min_signal_interval_seconds,
        })
        return base
