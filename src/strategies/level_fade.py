"""
Level Fade (Mean-Reversion) Strategy.

Generates fade signals when price touches a known intraday S/R level and L2
order flow shows absorption of opposing pressure.

Key design principles:
- No lookahead: uses only immediate L2 conditions (imbalance, spread, book
  pressure) rather than historical "bounce" labels.
- ATR-adaptive entry tolerance with hard tick floor.
- Spread gate to ensure edge is not consumed by transaction costs.
- Hybrid stop (computed in **price space**, not distance):
    long stop  = max(atr_stop_price, level_stop_price)  → tighter stop
    short stop = min(atr_stop_price, level_stop_price)  → tighter stop
- POC take-profit is capped to R:R target to prevent "hold & hope".
- Flow metrics grouped into a single bucket (max 40% of confidence) to
  avoid double-counting correlated L2 signals.
- Level cooldown after stop-out prevents re-entry at "damaged" level.
- Open/close time gate skips noisy microstructure windows.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .base_strategy import BaseStrategy, Regime, Signal, SignalType


class LevelFadeStrategy(BaseStrategy):
    """Level-based mean-reversion / fade strategy."""

    def __init__(
        self,
        # -- entry tolerance --
        entry_tolerance_pct: float = 1.0,
        entry_tolerance_atr_frac: float = 1.0,
        min_tolerance_ticks: int = 2,
        # -- level quality --
        min_tests: int = 1,
        min_confluence: int = 1,
        # -- L2 gates --
        min_absorption_rate: float = 0.01,
        min_signed_aggression: float = 0.01,
        min_book_pressure: float = 0.01,
        max_spread_ticks: float = 50.0,
        # -- confidence --
        min_confidence: float = 10.0,
        # -- risk/reward --
        atr_stop_multiplier: float = 1.0,
        level_break_buffer_ticks: float = 5.0,
        rr_ratio: float = 2.0,
        trailing_stop_pct: float = 0.5,
        poc_target_enabled: bool = True,
        # -- cooldown / time gates --
        level_cooldown_seconds: float = 120.0,
        skip_open_minutes: int = 3,
        skip_close_minutes: int = 5,
    ):
        super().__init__(
            name="LevelFade",
            regimes=[Regime.CHOPPY, Regime.MIXED],
        )
        self._uses_l2_internally = True
        self.entry_tolerance_pct = entry_tolerance_pct
        self.entry_tolerance_atr_frac = entry_tolerance_atr_frac
        self.min_tolerance_ticks = min_tolerance_ticks
        self.min_tests = min_tests
        self.min_confluence = min_confluence
        self.min_absorption_rate = min_absorption_rate
        self.min_signed_aggression = min_signed_aggression
        self.min_book_pressure = min_book_pressure
        self.max_spread_ticks = max_spread_ticks
        self.min_confidence = min_confidence
        self.atr_stop_multiplier = atr_stop_multiplier
        self.level_break_buffer_ticks = level_break_buffer_ticks
        self.rr_ratio = rr_ratio
        self.trailing_stop_pct = trailing_stop_pct
        self.poc_target_enabled = poc_target_enabled
        self.level_cooldown_seconds = level_cooldown_seconds
        self.skip_open_minutes = skip_open_minutes
        self.skip_close_minutes = skip_close_minutes

        # State: track recently stopped-out levels → (level_price, stop_out_ts)
        self._cooled_levels: List[Tuple[float, datetime]] = []

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _compute_entry_tolerance(
        self, price: float, atr: float, tick_size: float = 0.01,
    ) -> float:
        """Use the tighter of percent-based and ATR-based tolerance,
        with a hard floor in ticks so low-vol doesn't kill the strategy."""
        pct_tol = price * self.entry_tolerance_pct / 100.0
        atr_tol = atr * self.entry_tolerance_atr_frac if atr > 0.0 else pct_tol
        dynamic = min(pct_tol, atr_tol)
        floor = self.min_tolerance_ticks * tick_size
        return max(dynamic, floor)

    def _is_level_cooled(self, lvl_price: float, now: datetime, tick_size: float = 0.01) -> bool:
        """Check if a level was recently stopped-out and is on cooldown.
        Also garbage-collects entries older than 1 day."""
        if self.level_cooldown_seconds <= 0:
            return False
        # GC: remove entries older than 1 day
        max_age = 86400.0  # 1 day in seconds
        self._cooled_levels = [
            (p, t) for p, t in self._cooled_levels
            if (now - t).total_seconds() < max_age
        ]
        tol = 3 * tick_size  # within 3 ticks = same level
        for cooled_price, cooled_ts in self._cooled_levels:
            if abs(cooled_price - lvl_price) <= tol:
                elapsed = (now - cooled_ts).total_seconds()
                if elapsed < self.level_cooldown_seconds:
                    return True
        return False

    def record_level_stopout(self, level_price: float, ts: datetime) -> None:
        """Called externally (by exit engine or runtime) when this strategy
        gets stopped out. Records the level for cooldown purposes."""
        self._cooled_levels.append((level_price, ts))
        # Keep list bounded
        if len(self._cooled_levels) > 50:
            self._cooled_levels = self._cooled_levels[-20:]

    def _is_in_skip_window(self, ts: datetime) -> bool:
        """Return True if timestamp is in the open/close skip window.
        Market hours: 09:30–16:00 ET.  Uses timedelta for robust arithmetic."""
        try:
            from zoneinfo import ZoneInfo
            et = ts.astimezone(ZoneInfo("US/Eastern")) if ts.tzinfo else ts
        except Exception:
            et = ts
        t = et.time()
        market_open = time(9, 30)
        market_close = time(16, 0)
        # Skip first N minutes after open
        if self.skip_open_minutes > 0:
            open_cutoff = (datetime.combine(et.date(), market_open)
                           + timedelta(minutes=self.skip_open_minutes)).time()
            if market_open <= t < open_cutoff:
                return True
        # Skip last N minutes before close
        if self.skip_close_minutes > 0:
            close_cutoff = (datetime.combine(et.date(), market_close)
                            - timedelta(minutes=self.skip_close_minutes)).time()
            if close_cutoff <= t <= market_close:
                return True
        return False

    def _find_nearest_active_level(
        self,
        price: float,
        levels: List[Dict[str, Any]],
        tolerance: float,
        now: datetime,
        tick_size: float = 0.01,
    ) -> Optional[Dict[str, Any]]:
        """Return the nearest unbroken, sufficiently-tested level within
        tolerance that is not on cooldown."""
        best: Optional[Dict[str, Any]] = None
        best_dist = float("inf")

        for lvl in levels:
            if lvl.get("broken", False):
                continue
            lvl_price = float(lvl.get("price", 0.0))
            if lvl_price <= 0.0:
                continue
            tests = int(lvl.get("tests", 0))
            if tests < self.min_tests:
                continue
            confluence = int(lvl.get("confluence_points", 0))
            if confluence < self.min_confluence:
                continue
            if self._is_level_cooled(lvl_price, now, tick_size):
                continue
            dist = abs(price - lvl_price)
            if dist <= tolerance and dist < best_dist:
                best = lvl
                best_dist = dist
        return best

    # ------------------------------------------------------------------
    # generate_signal
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        current_price: float,
        ohlcv: Dict[str, List[float]],
        indicators: Dict[str, Any],
        regime: Regime,
        timestamp: datetime,
    ) -> Optional[Signal]:
        # TEMP: gate counter
        if not hasattr(self, '_gc'):
            self._gc = {}; self._gn = 0
        self._gn += 1
        def _g(k):
            self._gc[k] = self._gc.get(k, 0) + 1
            if self._gn % 10 == 0:
                import logging; logging.getLogger("LF").warning(f"[LF_GATES] n={self._gn} {self._gc}")

        if not self.is_allowed_in_regime(regime):
            _g('regime'); return None
        if len(self.get_open_positions()) > 0:
            _g('pos'); return None
        if self._is_in_skip_window(timestamp):
            _g('time'); return None
        flow = indicators.get("order_flow") or {}
        if not flow.get("has_l2_coverage", False):
            _g('l2'); return None
        absorption = float(flow.get("absorption_rate", 0.0) or 0.0)
        if absorption < self.min_absorption_rate:
            _g('absorp'); return None

        signed_aggr = float(flow.get("signed_aggression", 0.0) or 0.0)
        book_pressure = float(flow.get("book_pressure_avg", 0.0) or 0.0)
        imbalance = float(flow.get("imbalance_avg", 0.0) or 0.0)
        consistency = float(flow.get("directional_consistency", 0.0) or 0.0)
        divergence = float(flow.get("delta_price_divergence", 0.0) or 0.0)

        # ── Spread gate (convert bps → ticks for consistent gating) ──
        # spread_bps is in basis points; convert to ticks:
        #   spread_price = spread_bps * current_price / 10000
        #   spread_ticks  = spread_price / tick_size
        tick_size = float(indicators.get("tick_size", 0.01) or 0.01)
        spread_bps = float(flow.get("spread_bps", 0.0) or 0.0)
        spread_ticks = (spread_bps * current_price / 10000.0) / tick_size if tick_size > 0 else 0.0
        if self.max_spread_ticks > 0 and spread_ticks > self.max_spread_ticks:
            _g('spread'); return None

        # ── ATR ───────────────────────────────────────────────────────
        atr_series = indicators.get("atr") or []
        atr_val = float(atr_series[-1]) if atr_series else max(current_price * 0.004, 0.01)

        # ── Intraday levels ───────────────────────────────────────────
        il = indicators.get("intraday_levels") or {}
        levels = il.get("levels") or []
        if not levels:
            _g('levels'); return None

        # ATR-adaptive tolerance with hard tick floor
        tolerance = self._compute_entry_tolerance(current_price, atr_val, tick_size)

        level = self._find_nearest_active_level(
            current_price, levels, tolerance, timestamp, tick_size,
        )
        if level is None:
            _g('tolerance'); return None

        lvl_price = float(level.get("price", 0.0))
        lvl_kind = str(level.get("kind", "")).lower()
        lvl_tests = int(level.get("tests", 0))
        lvl_confluence = int(level.get("confluence_points", 0))

        # ── Direction (mutually exclusive) ─────────────────────────────
        # If the level has an explicit kind, trust it unconditionally.
        # Otherwise infer from price position relative to the level.
        if lvl_kind == "support":
            is_support, is_resistance = True, False
        elif lvl_kind == "resistance":
            is_support, is_resistance = False, True
        else:
            is_support = current_price >= lvl_price
            is_resistance = not is_support

        long_trigger = False
        short_trigger = False

        # Level integrity: price must still be ON the level's side.
        # Prevents entry when price has already broken through.
        level_buffer = self.level_break_buffer_ticks * tick_size

        # Immediate L2 conditions only — no historical "bounce" label.
        # At support: sellers active but level holding (bids replenishing).
        # Price must be at or above (level - buffer) to confirm level intact.
        if is_support:
            if current_price >= lvl_price - level_buffer:
                if (signed_aggr <= -self.min_signed_aggression
                        or book_pressure >= self.min_book_pressure
                        or absorption >= self.min_absorption_rate):
                    long_trigger = True
        # At resistance: buyers active but level holding (asks replenishing).
        # Price must be at or below (level + buffer) to confirm level intact.
        if is_resistance:
            if current_price <= lvl_price + level_buffer:
                if (signed_aggr >= self.min_signed_aggression
                        or book_pressure <= -self.min_book_pressure
                        or absorption >= self.min_absorption_rate):
                    short_trigger = True

        if not long_trigger and not short_trigger:
            _g('direction'); return None

        # ── Confidence (flow bucket ≤40%, structure bucket ≤60%) ──────
        flow_absorption = self._clamp01(absorption / max(self.min_absorption_rate * 2.0, 1e-6))
        flow_aggression = self._clamp01(abs(signed_aggr) / max(self.min_signed_aggression * 3.0, 1e-6))
        flow_book = self._clamp01(abs(book_pressure) / max(self.min_book_pressure * 3.0, 1e-6))
        flow_bucket = self._clamp01(0.50 * flow_absorption + 0.30 * flow_aggression + 0.20 * flow_book)

        confluence_score = self._clamp01((lvl_confluence - 1.0) / 3.0)
        test_score = self._clamp01((lvl_tests - 1.0) / 4.0)
        divergence_score = self._clamp01(abs(divergence) / 0.30)
        structure_bucket = 0.40 * confluence_score + 0.30 * test_score + 0.30 * divergence_score

        confidence = 100.0 * (0.40 * flow_bucket + 0.60 * structure_bucket)

        if confidence < self.min_confidence:
            _g('conf'); return None

        # ── Hybrid stop (price space) ─────────────────────────────────
        # For longs:  stop = max(atr_stop, level_stop) → picks the HIGHER
        #             price (closer to entry), giving a tighter stop.
        # For shorts: stop = min(atr_stop, level_stop) → picks the LOWER
        #             price (closer to entry), giving a tighter stop.
        effective_atr_mult = self.get_effective_atr_stop_multiplier() or self.atr_stop_multiplier
        effective_rr = self.get_effective_rr_ratio() or self.rr_ratio
        # level_buffer already computed above for integrity check

        if long_trigger:
            signal_type = SignalType.BUY
            atr_stop = current_price - (atr_val * effective_atr_mult)
            level_stop = lvl_price - level_buffer
            stop_loss = max(atr_stop, level_stop)  # tighter (higher price)
            
            take_profit = self.calculate_take_profit(
                current_price, stop_loss, effective_rr, side="long"
            )
            # POC target: can only tighten, never widen
            vp = il.get("volume_profile") or {}
            poc_price = float(vp.get("poc_price") or 0.0)
            if self.poc_target_enabled and poc_price > current_price:
                take_profit = min(take_profit, poc_price)
            direction_label = "support_fade"
            
        else:
            signal_type = SignalType.SELL
            atr_stop = current_price + (atr_val * effective_atr_mult)
            level_stop = lvl_price + level_buffer
            stop_loss = min(atr_stop, level_stop)  # tighter (lower price)
            
            take_profit = self.calculate_take_profit(
                current_price, stop_loss, effective_rr, side="short"
            )
            vp = il.get("volume_profile") or {}
            poc_price = float(vp.get("poc_price") or 0.0)
            if self.poc_target_enabled and poc_price > 0.0 and poc_price < current_price:
                take_profit = max(take_profit, poc_price)
            direction_label = "resistance_fade"

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
                f"Level fade at {lvl_price:.2f} ({lvl_kind}, tests={lvl_tests}, "
                f"confluence={lvl_confluence}). "
                f"Absorption {absorption:.2f}, signed_aggr {signed_aggr:+.2f}, "
                f"book {book_pressure:+.2f}, spread {spread_ticks:.1f}ticks ({spread_bps:.1f}bps)"
            ),
            metadata={
                "flow_direction": direction_label,
                "level_price": lvl_price,
                "level_kind": lvl_kind,
                "level_tests": lvl_tests,
                "level_confluence": lvl_confluence,
                "stop_type": "hybrid_price_space",
                "atr_stop": round(atr_stop, 4),
                "level_stop": round(level_stop, 4),
                "poc_price": poc_price if poc_price > 0.0 else None,
                "entry_tolerance": round(tolerance, 4),
                "spread_bps": spread_bps,
                "spread_ticks": round(spread_ticks, 2),
                "tick_size": tick_size,
                "confidence_breakdown": {
                    "flow_bucket": round(flow_bucket, 3),
                    "structure_bucket": round(structure_bucket, 3),
                },
                "order_flow": {
                    "absorption_rate": absorption,
                    "signed_aggression": signed_aggr,
                    "book_pressure_avg": book_pressure,
                    "imbalance_avg": imbalance,
                    "directional_consistency": consistency,
                    "delta_price_divergence": divergence,
                },
            },
        )
        self.add_signal(signal)
        import logging; logging.getLogger("LF").warning(f"[LF_SUCCESS] Generated signal for {current_price} at {timestamp}")
        return signal

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "entry_tolerance_pct": self.entry_tolerance_pct,
                "entry_tolerance_atr_frac": self.entry_tolerance_atr_frac,
                "min_tolerance_ticks": self.min_tolerance_ticks,
                "min_tests": self.min_tests,
                "min_absorption_rate": self.min_absorption_rate,
                "min_signed_aggression": self.min_signed_aggression,
                "min_book_pressure": self.min_book_pressure,
                "max_spread_ticks": self.max_spread_ticks,
                "min_confluence": self.min_confluence,
                "min_confidence": self.min_confidence,
                "atr_stop_multiplier": self.atr_stop_multiplier,
                "level_break_buffer_ticks": self.level_break_buffer_ticks,
                "rr_ratio": self.rr_ratio,
                "trailing_stop_pct": self.trailing_stop_pct,
                "poc_target_enabled": self.poc_target_enabled,
                "level_cooldown_seconds": self.level_cooldown_seconds,
                "skip_open_minutes": self.skip_open_minutes,
                "skip_close_minutes": self.skip_close_minutes,
            }
        )
        return base
