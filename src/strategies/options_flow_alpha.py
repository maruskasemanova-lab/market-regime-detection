"""
Options Flow Alpha Strategy (TCBBO).

Uses options flow as a direct alpha factor — not a filter.
Three independent setups:
  1. Price-Flow Divergence  – price moves opposite to net options premium
  2. Momentum Spike         – extreme Z-score in net premium → momentum follow
  3. Sweep Alert            – whale sweep activity as directional confirmation
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .base_strategy import BaseStrategy, Regime, Signal, SignalType


class OptionsFlowAlphaStrategy(BaseStrategy):
    """Generates signals directly from TCBBO options flow data."""

    def __init__(
        self,
        # ── Z-Score Lookback ──
        zscore_lookback_bars: int = 20,
        # ── Setup 1: Price-Flow Divergence ──
        divergence_min_price_move_pct: float = 0.12,
        divergence_min_premium_z: float = 1.5,
        divergence_base_confidence: float = 72.0,
        # ── Setup 2: Momentum Spike ──
        momentum_min_premium_z: float = 1.5,
        momentum_max_opposing_price_pct: float = 0.08,
        momentum_base_confidence: float = 68.0,
        # ── Setup 3: Sweep Alert ──
        sweep_multiplier: float = 1.8,
        sweep_min_count: int = 2,
        sweep_base_confidence: float = 74.0,
        # ── Risk ──
        stop_loss_pct: float = 0.30,
        take_profit_rr: float = 3.0,
        trailing_stop_pct: float = 0.20,
    ):
        super().__init__(
            name="options_flow_alpha",
            regimes=[Regime.TRENDING, Regime.CHOPPY, Regime.MIXED],
        )
        self.zscore_lookback_bars = max(10, zscore_lookback_bars)

        # Setup 1
        self.divergence_min_price_move_pct = divergence_min_price_move_pct
        self.divergence_min_premium_z = divergence_min_premium_z
        self.divergence_base_confidence = divergence_base_confidence

        # Setup 2
        self.momentum_min_premium_z = momentum_min_premium_z
        self.momentum_max_opposing_price_pct = momentum_max_opposing_price_pct
        self.momentum_base_confidence = momentum_base_confidence

        # Setup 3
        self.sweep_multiplier = sweep_multiplier
        self.sweep_min_count = sweep_min_count
        self.sweep_base_confidence = sweep_base_confidence

        # Risk
        self.stop_loss_pct = 0.35
        self.take_profit_rr = take_profit_rr
        self.trailing_stop_pct = trailing_stop_pct

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(val: Any, default: float = 0.0) -> float:
        try:
            return float(val) if val is not None else default
        except (TypeError, ValueError):
            return default

    def _compute_zscore(
        self, values: List[float], current: float
    ) -> Optional[float]:
        """Compute Z-score of current vs rolling window."""
        if len(values) < self.zscore_lookback_bars:
            return None
        window = values[-self.zscore_lookback_bars :]
        mean = sum(window) / len(window)
        variance = sum((v - mean) ** 2 for v in window) / len(window)
        stdev = variance ** 0.5
        if stdev < 1e-9:
            return None
        return (current - mean) / stdev

    def _extract_tcbbo_history(
        self, indicators: Dict[str, Any]
    ) -> Tuple[List[float], List[float], List[float]]:
        """Extract net_premium, sweep_premium, and sweep_count history.

        Uses ``session_bars`` (list of recent bar dicts) that the evidence
        engine attaches to ``indicators``.
        """
        net_premiums: List[float] = []
        sweep_premiums: List[float] = []
        sweep_counts: List[float] = []

        session_bars = indicators.get("session_bars") or []
        for bar in session_bars:
            np_val = self._safe_float(
                bar.get("tcbbo_net_premium")
                if isinstance(bar, dict) else getattr(bar, "tcbbo_net_premium", None)
            )
            sp_val = self._safe_float(
                bar.get("tcbbo_sweep_premium")
                if isinstance(bar, dict) else getattr(bar, "tcbbo_sweep_premium", None)
            )
            sc_val = self._safe_float(
                bar.get("tcbbo_sweep_count")
                if isinstance(bar, dict) else getattr(bar, "tcbbo_sweep_count", None)
            )
            net_premiums.append(np_val)
            sweep_premiums.append(sp_val)
            sweep_counts.append(sc_val)

        return net_premiums, sweep_premiums, sweep_counts

    def _price_return_5bar(self, ohlcv: Dict[str, List[float]]) -> float:
        """Compute most recent 5-bar close return (%)."""
        closes = ohlcv.get("close") or []
        if len(closes) < 6:
            return 0.0
        baseline = closes[-6]
        if baseline == 0:
            return 0.0
        return ((closes[-1] - baseline) / baseline) * 100.0

    # ------------------------------------------------------------------
    # Setup Evaluators
    # ------------------------------------------------------------------

    def _evaluate_divergence(
        self,
        price_return_5: float,
        current_net_premium: float,
        net_premium_z: Optional[float],
    ) -> Tuple[Optional[str], float, str]:
        """Setup 1: Price-Flow Divergence."""
        if net_premium_z is None:
            return None, 0.0, ""

        # Price down + premium bullish spike → long
        if (
            price_return_5 < -self.divergence_min_price_move_pct
            and net_premium_z > self.divergence_min_premium_z
        ):
            conf = min(
                95.0,
                self.divergence_base_confidence + abs(net_premium_z) * 5.0,
            )
            reason = (
                f"divergence_long: price_5bar={price_return_5:.2f}%, "
                f"premium_z={net_premium_z:.2f}, "
                f"net_prem=${current_net_premium / 1000:.0f}k"
            )
            return "bullish", conf, reason

        # Price up + premium bearish spike → short
        if (
            price_return_5 > self.divergence_min_price_move_pct
            and net_premium_z < -self.divergence_min_premium_z
        ):
            conf = min(
                95.0,
                self.divergence_base_confidence + abs(net_premium_z) * 5.0,
            )
            reason = (
                f"divergence_short: price_5bar={price_return_5:+.2f}%, "
                f"premium_z={net_premium_z:.2f}, "
                f"net_prem=${current_net_premium / 1000:.0f}k"
            )
            return "bearish", conf, reason

        return None, 0.0, ""

    def _evaluate_momentum(
        self,
        price_return_5: float,
        current_net_premium: float,
        net_premium_z: Optional[float],
    ) -> Tuple[Optional[str], float, str]:
        """Setup 2: Momentum Spike."""
        if net_premium_z is None:
            return None, 0.0, ""

        # Bullish momentum: premium Z > threshold, price not strongly bearish
        if (
            net_premium_z > self.momentum_min_premium_z
            and price_return_5 > -self.momentum_max_opposing_price_pct
        ):
            conf = min(
                95.0,
                self.momentum_base_confidence + abs(net_premium_z) * 5.0,
            )
            reason = (
                f"momentum_long: premium_z={net_premium_z:.2f}, "
                f"price_5bar={price_return_5:+.2f}%, "
                f"net_prem=${current_net_premium / 1000:.0f}k"
            )
            return "bullish", conf, reason

        # Bearish momentum
        if (
            net_premium_z < -self.momentum_min_premium_z
            and price_return_5 < self.momentum_max_opposing_price_pct
        ):
            conf = min(
                95.0,
                self.momentum_base_confidence + abs(net_premium_z) * 5.0,
            )
            reason = (
                f"momentum_short: premium_z={net_premium_z:.2f}, "
                f"price_5bar={price_return_5:+.2f}%, "
                f"net_prem=${current_net_premium / 1000:.0f}k"
            )
            return "bearish", conf, reason

        return None, 0.0, ""

    def _evaluate_sweep(
        self,
        current_net_premium: float,
        current_sweep_premium: float,
        current_sweep_count: float,
        sweep_premiums: List[float],
    ) -> Tuple[Optional[str], float, str]:
        """Setup 3: Sweep Alert."""
        if current_sweep_count < self.sweep_min_count:
            return None, 0.0, ""

        # Median of non-zero sweep premiums
        nonzero_sweeps = [s for s in sweep_premiums if s > 0]
        if len(nonzero_sweeps) < 5:
            return None, 0.0, ""
        nonzero_sweeps.sort()
        median_sweep = nonzero_sweeps[len(nonzero_sweeps) // 2]

        if current_sweep_premium < median_sweep * self.sweep_multiplier:
            return None, 0.0, ""

        # Direction from net premium
        if current_net_premium > 0:
            conf = min(
                95.0,
                self.sweep_base_confidence + min(current_sweep_count, 10) * 2.0,
            )
            reason = (
                f"sweep_long: sweeps={int(current_sweep_count)}, "
                f"sweep_prem=${current_sweep_premium / 1000:.0f}k "
                f"({current_sweep_premium / median_sweep:.1f}x median), "
                f"net_prem=${current_net_premium / 1000:.0f}k"
            )
            return "bullish", conf, reason

        if current_net_premium < 0:
            conf = min(
                95.0,
                self.sweep_base_confidence + min(current_sweep_count, 10) * 2.0,
            )
            reason = (
                f"sweep_short: sweeps={int(current_sweep_count)}, "
                f"sweep_prem=${current_sweep_premium / 1000:.0f}k "
                f"({current_sweep_premium / median_sweep:.1f}x median), "
                f"net_prem=${current_net_premium / 1000:.0f}k"
            )
            return "bearish", conf, reason

        return None, 0.0, ""

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        current_price: float,
        ohlcv: Dict[str, List[float]],
        indicators: Dict[str, Any],
        regime: Regime,
        timestamp: datetime,
    ) -> Optional[Signal]:
        # ── Check TCBBO availability ──
        tcbbo = indicators.get("tcbbo") or {}
        has_data = bool(tcbbo.get("is_valid", False)) or bool(
            tcbbo.get("has_data", False)
        )
        print(f"DEBUG: OptionsFlowAlphaStrategy.generate_signal called, has_data={has_data}")
        if not has_data:
            return None

        current_net_premium = self._safe_float(tcbbo.get("net_premium"))
        current_sweep_premium = self._safe_float(tcbbo.get("sweep_premium"))
        current_sweep_count = self._safe_float(tcbbo.get("sweep_count"))

        # ── Build history for Z-score ──
        net_premiums, sweep_premiums, _ = self._extract_tcbbo_history(indicators)
        net_premium_z = self._compute_zscore(net_premiums, current_net_premium)

        price_return_5 = self._price_return_5bar(ohlcv)

        # ── Evaluate setups (priority order) ──
        direction: Optional[str] = None
        confidence = 0.0
        reasoning = ""
        setups_fired: List[str] = []

        # 1. Divergence (highest conviction — smart money vs. retail)
        d, c, r = self._evaluate_divergence(
            price_return_5, current_net_premium, net_premium_z
        )
        if d and c > confidence:
            direction, confidence, reasoning = d, c, r
            setups_fired.append("divergence")

        # 2. Momentum spike
        d2, c2, r2 = self._evaluate_momentum(
            price_return_5, current_net_premium, net_premium_z
        )
        if d2 and c2 > confidence:
            direction, confidence, reasoning = d2, c2, r2
        if d2:
            setups_fired.append("momentum")

        # 3. Sweep alert
        d3, c3, r3 = self._evaluate_sweep(
            current_net_premium,
            current_sweep_premium,
            current_sweep_count,
            sweep_premiums,
        )
        if d3 and c3 > confidence:
            direction, confidence, reasoning = d3, c3, r3
        if d3:
            setups_fired.append("sweep")

        # 4. Cumulative premium trend confirmation
        cum_premium = self._safe_float(tcbbo.get("cumulative_net_premium"))
        cum_trend_bullish = cum_premium > 0 and current_net_premium > 0
        cum_trend_bearish = cum_premium < 0 and current_net_premium < 0

        if direction is None:
            return None

        print(f"DEBUG: OptionsFlowAlphaStrategy GENERATING SIGNAL: side={direction}, conf={confidence}, reasoning={reasoning}")

        # ── Confluence bonus ──
        # Multiple setups agree → stronger conviction
        if len(setups_fired) >= 2:
            confidence = min(95.0, confidence + 8.0)
            reasoning += f" [CONFLUENCE: {'+'.join(setups_fired)}]"

        # Cumulative trend aligns with direction → extra boost
        if (direction == "bullish" and cum_trend_bullish) or (
            direction == "bearish" and cum_trend_bearish
        ):
            confidence = min(95.0, confidence + 5.0)
            reasoning += " [CUM_TREND_ALIGNED]"

        # ── Build signal ──
        side = "long" if direction == "bullish" else "short"
        signal_type = SignalType.BUY if side == "long" else SignalType.SELL

        stop_loss = self.calculate_percent_stop(
            current_price, self.stop_loss_pct, side=side
        )
        take_profit = self.calculate_take_profit(
            current_price, stop_loss, rr_ratio=self.take_profit_rr, side=side
        )

        return Signal(
            strategy_name=self.name,
            signal_type=signal_type,
            price=current_price,
            timestamp=timestamp,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=True,
            trailing_stop_pct=self.trailing_stop_pct,
            reasoning=reasoning,
            metadata={
                "setup": reasoning.split(":")[0] if ":" in reasoning else "unknown",
                "premium_z": round(net_premium_z, 2) if net_premium_z else None,
                "net_premium": current_net_premium,
                "sweep_count": int(current_sweep_count),
                "price_return_5bar": round(price_return_5, 3),
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "zscore_lookback_bars": self.zscore_lookback_bars,
            "divergence_min_price_move_pct": self.divergence_min_price_move_pct,
            "divergence_min_premium_z": self.divergence_min_premium_z,
            "momentum_min_premium_z": self.momentum_min_premium_z,
            "sweep_multiplier": self.sweep_multiplier,
            "sweep_min_count": self.sweep_min_count,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_rr": self.take_profit_rr,
            "trailing_stop_pct": self.trailing_stop_pct,
        }
