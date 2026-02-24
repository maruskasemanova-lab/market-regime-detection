"""
Golden Setup Detector.

Identifies 5 high-probability L2 + TCBBO setup patterns that warrant
lowered entry thresholds and boosted confidence.  Acts as a pre-gate
threshold modifier — does NOT create standalone signals.

Setups (all have bearish mirrors):
  1. Absorption Reversal  – selling absorbed at support, bid holds
  2. Gamma Squeeze         – ask thinning + buying aggression near resistance
  3. Liquidity Trap        – delta flip after false breakout below support
  4. Iceberg Defense       – iceberg reloading + absorption at support
  5. Fuel Injection        – resistance-side thinning in uptrend consolidation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums / Result types
# ---------------------------------------------------------------------------

class GoldenSetupType(Enum):
    ABSORPTION_REVERSAL = "absorption_reversal"
    GAMMA_SQUEEZE = "gamma_squeeze"
    LIQUIDITY_TRAP = "liquidity_trap"
    ICEBERG_DEFENSE = "iceberg_defense"
    FUEL_INJECTION = "fuel_injection"


@dataclass
class GoldenSetupMatch:
    """A single matched golden setup for the current bar."""
    setup_type: GoldenSetupType
    direction: str                   # "bullish" / "bearish"
    l2_score: float                  # 0-100
    tcbbo_score: float               # 0-100 (0 when no TCBBO data)
    has_tcbbo: bool
    bypass_choppy: bool
    threshold_reduction: float       # points to subtract from gate threshold
    confidence_boost: float          # points to add to signal confidence
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GoldenSetupResult:
    """Aggregate output from running all 5 detectors on one bar."""
    active: bool = False
    matches: List[GoldenSetupMatch] = field(default_factory=list)
    best_match: Optional[GoldenSetupMatch] = None
    bypass_choppy: bool = False
    threshold_reduction: float = 0.0
    confidence_boost: float = 0.0
    bar_index: int = -1
    entries_today: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active": self.active,
            "setup_count": len(self.matches),
            "best_setup": self.best_match.setup_type.value if self.best_match else None,
            "best_direction": self.best_match.direction if self.best_match else None,
            "bypass_choppy": self.bypass_choppy,
            "threshold_reduction": round(self.threshold_reduction, 2),
            "confidence_boost": round(self.confidence_boost, 2),
            "entries_today": self.entries_today,
            "matches": [
                {
                    "type": m.setup_type.value,
                    "direction": m.direction,
                    "l2_score": round(m.l2_score, 1),
                    "tcbbo_score": round(m.tcbbo_score, 1),
                    "has_tcbbo": m.has_tcbbo,
                    "threshold_reduction": round(m.threshold_reduction, 2),
                    "confidence_boost": round(m.confidence_boost, 2),
                    "diagnostics": m.diagnostics,
                }
                for m in self.matches
            ],
        }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GoldenSetupConfig:
    """Per-setup thresholds — built from TradingConfig fields at call time."""

    enabled: bool = False
    max_entries_per_day: int = 3
    cooldown_bars: int = 5
    min_l2_score: float = 60.0

    # -- Absorption Reversal --
    ar_enabled: bool = True
    ar_min_absorption_rate: float = 0.50
    ar_min_book_pressure: float = 0.05
    ar_max_signed_aggression: float = -0.02
    ar_level_tolerance_pct: float = 0.15
    ar_threshold_reduction: float = 12.0
    ar_confidence_boost_l2: float = 15.0
    ar_confidence_boost_tcbbo: float = 10.0
    ar_bypass_choppy: bool = True

    # -- Gamma Squeeze --
    gsb_enabled: bool = True
    gsb_min_signed_aggression: float = 0.02
    gsb_min_book_pressure_increase: float = 0.02
    gsb_ask_depth_decline_pct: float = 10.0
    gsb_level_tolerance_pct: float = 0.15
    gsb_min_tcbbo_sweep_count: int = 2
    gsb_threshold_reduction: float = 10.0
    gsb_confidence_boost_l2: float = 12.0
    gsb_confidence_boost_tcbbo: float = 8.0
    gsb_bypass_choppy: bool = False

    # -- Liquidity Trap --
    lt_enabled: bool = True
    lt_min_delta_acceleration: float = 0.01
    lt_false_breakout_min_pct: float = 0.10
    lt_false_breakout_max_pct: float = 0.25
    lt_level_tolerance_pct: float = 0.25
    lt_threshold_reduction: float = 10.0
    lt_confidence_boost_l2: float = 12.0
    lt_confidence_boost_tcbbo: float = 8.0
    lt_bypass_choppy: bool = True

    # -- Iceberg Defense --
    id_enabled: bool = True
    id_min_iceberg_count: int = 1
    id_min_absorption_rate: float = 0.40
    id_min_book_pressure: float = 0.03
    id_level_tolerance_pct: float = 0.15
    id_threshold_reduction: float = 8.0
    id_confidence_boost_l2: float = 10.0
    id_confidence_boost_tcbbo: float = 8.0
    id_bypass_choppy: bool = False

    # -- Fuel Injection --
    fi_enabled: bool = True
    fi_min_signed_aggression: float = 0.01
    fi_ask_depth_decline_pct: float = 8.0
    fi_max_bar_range_pct: float = 0.15
    fi_level_tolerance_pct: float = 0.20
    fi_threshold_reduction: float = 8.0
    fi_confidence_boost_l2: float = 10.0
    fi_confidence_boost_tcbbo: float = 8.0
    fi_bypass_choppy: bool = False


def build_golden_config_from_trading_config(tc: Any) -> GoldenSetupConfig:
    """Extract GoldenSetupConfig from a TradingConfig instance."""

    def _f(name: str, default: float) -> float:
        return float(getattr(tc, name, default) or default)

    def _i(name: str, default: int) -> int:
        return int(getattr(tc, name, default) or default)

    def _b(name: str, default: bool) -> bool:
        return bool(getattr(tc, name, default))

    return GoldenSetupConfig(
        enabled=_b("golden_setups_enabled", False),
        max_entries_per_day=_i("golden_setups_max_entries_per_day", 3),
        cooldown_bars=_i("golden_setups_cooldown_bars", 5),
        min_l2_score=_f("golden_setups_min_l2_score", 60.0),
        ar_enabled=_b("golden_setup_absorption_reversal_enabled", True),
        ar_threshold_reduction=_f("golden_setup_ar_threshold_reduction", 12.0),
        ar_confidence_boost_l2=_f("golden_setup_ar_confidence_boost", 15.0),
        ar_confidence_boost_tcbbo=_f("golden_setup_ar_tcbbo_boost", 10.0),
        gsb_enabled=_b("golden_setup_gamma_squeeze_enabled", True),
        gsb_threshold_reduction=_f("golden_setup_gsb_threshold_reduction", 10.0),
        gsb_confidence_boost_l2=_f("golden_setup_gsb_confidence_boost", 12.0),
        gsb_confidence_boost_tcbbo=_f("golden_setup_gsb_tcbbo_boost", 8.0),
        lt_enabled=_b("golden_setup_liquidity_trap_enabled", True),
        lt_threshold_reduction=_f("golden_setup_lt_threshold_reduction", 10.0),
        lt_confidence_boost_l2=_f("golden_setup_lt_confidence_boost", 12.0),
        lt_confidence_boost_tcbbo=_f("golden_setup_lt_tcbbo_boost", 8.0),
        id_enabled=_b("golden_setup_iceberg_defense_enabled", True),
        id_threshold_reduction=_f("golden_setup_id_threshold_reduction", 8.0),
        id_confidence_boost_l2=_f("golden_setup_id_confidence_boost", 10.0),
        id_confidence_boost_tcbbo=_f("golden_setup_id_tcbbo_boost", 8.0),
        fi_enabled=_b("golden_setup_fuel_injection_enabled", True),
        fi_threshold_reduction=_f("golden_setup_fi_threshold_reduction", 8.0),
        fi_confidence_boost_l2=_f("golden_setup_fi_confidence_boost", 10.0),
        fi_confidence_boost_tcbbo=_f("golden_setup_fi_tcbbo_boost", 8.0),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _level_distance_pct(price: float, level_price: float) -> float:
    if price <= 0.0 or level_price <= 0.0:
        return 999.0
    return abs((level_price - price) / price) * 100.0


def _find_nearest_level(
    levels: List[Dict[str, Any]],
    current_price: float,
    kind_filter: str,
    tolerance_pct: float,
) -> Optional[Dict[str, Any]]:
    """Find the nearest unbroken level of the given kind within tolerance."""
    best: Optional[Dict[str, Any]] = None
    best_dist = tolerance_pct + 1.0
    for row in levels:
        if not isinstance(row, dict):
            continue
        level_price = _safe_float(row.get("price"))
        if level_price <= 0.0:
            continue
        if bool(row.get("broken", False)):
            continue
        kind = str(row.get("kind", "")).strip().lower()
        if kind not in {"support", "resistance"}:
            kind = "support" if level_price <= current_price else "resistance"
        if kind != kind_filter:
            continue
        dist = _level_distance_pct(current_price, level_price)
        if dist <= tolerance_pct and dist < best_dist:
            best_dist = dist
            best = {
                "price": level_price,
                "kind": kind,
                "source": str(row.get("source", "")),
                "distance_pct": round(dist, 4),
                "tests": int(row.get("tests", 0) or 0),
            }
    return best


def _avg_depth(bars: list, field_name: str, lookback: int = 5) -> float:
    """Average of a bar-level field over the last `lookback` bars."""
    recent = bars[-lookback:] if len(bars) >= lookback else bars
    vals = [_safe_float(getattr(b, field_name, None)) for b in recent]
    vals = [v for v in vals if v > 0.0]
    return sum(vals) / len(vals) if vals else 0.0


def _tcbbo_fields(bar: Any) -> Dict[str, float]:
    """Extract TCBBO fields from bar as floats."""
    return {
        "has_data": bool(getattr(bar, "tcbbo_has_data", False)),
        "net_premium": _safe_float(getattr(bar, "tcbbo_net_premium", None)),
        "call_buy_premium": _safe_float(getattr(bar, "tcbbo_call_buy_premium", None)),
        "put_buy_premium": _safe_float(getattr(bar, "tcbbo_put_buy_premium", None)),
        "call_sell_premium": _safe_float(getattr(bar, "tcbbo_call_sell_premium", None)),
        "put_sell_premium": _safe_float(getattr(bar, "tcbbo_put_sell_premium", None)),
        "sweep_count": _safe_float(getattr(bar, "tcbbo_sweep_count", None)),
        "sweep_premium": _safe_float(getattr(bar, "tcbbo_sweep_premium", None)),
    }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class GoldenSetupDetector:
    """Stateless per-bar detector.  Instantiated once per DayTradingManager."""

    def __init__(self) -> None:
        pass

    def evaluate(
        self,
        *,
        bar: Any,
        bars: list,
        flow_metrics: Dict[str, Any],
        intraday_levels_state: Dict[str, Any],
        current_price: float,
        vwap: Optional[float],
        regime: str,
        golden_entries_today: int,
        last_golden_bar_index: int,
        current_bar_index: int,
        config: Optional[GoldenSetupConfig] = None,
    ) -> GoldenSetupResult:
        cfg = config or GoldenSetupConfig()
        result = GoldenSetupResult(
            bar_index=current_bar_index,
            entries_today=golden_entries_today,
        )

        # Global guards
        if not cfg.enabled:
            return result
        if golden_entries_today >= cfg.max_entries_per_day:
            return result
        if (current_bar_index - last_golden_bar_index) < cfg.cooldown_bars:
            return result
        if not flow_metrics.get("has_l2_coverage", False):
            return result
        if len(bars) < 5:
            return result

        levels = intraday_levels_state.get("levels", [])
        if not isinstance(levels, list):
            levels = []

        tcbbo = _tcbbo_fields(bar)
        matches: List[GoldenSetupMatch] = []

        # Run all detectors (both directions)
        for direction in ("bullish", "bearish"):
            if cfg.ar_enabled:
                m = self._detect_absorption_reversal(
                    cfg, bar, bars, flow_metrics, levels, current_price, tcbbo, direction,
                )
                if m:
                    matches.append(m)

            if cfg.gsb_enabled:
                m = self._detect_gamma_squeeze(
                    cfg, bar, bars, flow_metrics, levels, current_price, tcbbo, direction,
                )
                if m:
                    matches.append(m)

            if cfg.lt_enabled:
                m = self._detect_liquidity_trap(
                    cfg, bar, bars, flow_metrics, levels, current_price, tcbbo, direction,
                )
                if m:
                    matches.append(m)

            if cfg.id_enabled:
                m = self._detect_iceberg_defense(
                    cfg, bar, bars, flow_metrics, levels, current_price, tcbbo, direction,
                )
                if m:
                    matches.append(m)

            if cfg.fi_enabled:
                m = self._detect_fuel_injection(
                    cfg, bar, bars, flow_metrics, levels, current_price, vwap, tcbbo, direction,
                )
                if m:
                    matches.append(m)

        # Filter by min L2 score
        matches = [m for m in matches if m.l2_score >= cfg.min_l2_score]

        if not matches:
            return result

        best = max(matches, key=lambda m: m.l2_score + m.tcbbo_score)
        result.active = True
        result.matches = matches
        result.best_match = best
        result.bypass_choppy = best.bypass_choppy
        result.threshold_reduction = best.threshold_reduction
        result.confidence_boost = best.confidence_boost

        logger.info(
            "Golden setup detected: %s %s l2=%.1f tcbbo=%.1f thresh_red=%.1f conf_boost=%.1f",
            best.setup_type.value,
            best.direction,
            best.l2_score,
            best.tcbbo_score,
            best.threshold_reduction,
            best.confidence_boost,
        )
        return result

    # -------------------------------------------------------------------
    # Setup 1: Absorption Reversal
    # -------------------------------------------------------------------
    def _detect_absorption_reversal(
        self,
        cfg: GoldenSetupConfig,
        bar: Any,
        bars: list,
        flow: Dict[str, Any],
        levels: list,
        price: float,
        tcbbo: Dict[str, float],
        direction: str,
    ) -> Optional[GoldenSetupMatch]:
        """
        Bullish: selling absorbed at support — aggressive selling but price stops falling.
        Bearish: buying absorbed at resistance — aggressive buying but price stops rising.
        """
        absorption = _safe_float(flow.get("absorption_rate"))
        signed_aggr = _safe_float(flow.get("signed_aggression"))
        book_pressure = _safe_float(flow.get("book_pressure_avg"))

        if direction == "bullish":
            level_kind = "support"
            aggr_ok = signed_aggr <= cfg.ar_max_signed_aggression
            bp_ok = book_pressure >= cfg.ar_min_book_pressure
        else:
            level_kind = "resistance"
            aggr_ok = signed_aggr >= -cfg.ar_max_signed_aggression
            bp_ok = book_pressure <= -cfg.ar_min_book_pressure

        # Context: price near level
        nearest = _find_nearest_level(levels, price, level_kind, cfg.ar_level_tolerance_pct)
        if not nearest:
            return None

        # L2 core conditions
        if absorption < cfg.ar_min_absorption_rate:
            return None
        if not aggr_ok:
            return None
        if not bp_ok:
            return None

        # Score (partial credit)
        absorption_score = _clamp01(absorption / max(cfg.ar_min_absorption_rate * 2.0, 0.01))
        aggression_score = _clamp01(abs(signed_aggr) / 0.20)
        bp_score = _clamp01(abs(book_pressure) / 0.20)
        proximity_score = _clamp01(1.0 - nearest["distance_pct"] / max(cfg.ar_level_tolerance_pct, 0.01))
        l2_score = 100.0 * (
            0.35 * absorption_score
            + 0.25 * aggression_score
            + 0.20 * bp_score
            + 0.20 * proximity_score
        )

        # TCBBO boost
        tcbbo_score = 0.0
        if tcbbo["has_data"]:
            net_ok = (tcbbo["net_premium"] > 0) if direction == "bullish" else (tcbbo["net_premium"] < 0)
            sweep_ok = tcbbo["sweep_count"] > 0
            if net_ok:
                tcbbo_score += 50.0
            if sweep_ok:
                tcbbo_score += 50.0

        conf_boost = cfg.ar_confidence_boost_l2
        if tcbbo_score > 0:
            conf_boost += cfg.ar_confidence_boost_tcbbo * (tcbbo_score / 100.0)

        return GoldenSetupMatch(
            setup_type=GoldenSetupType.ABSORPTION_REVERSAL,
            direction=direction,
            l2_score=l2_score,
            tcbbo_score=tcbbo_score,
            has_tcbbo=tcbbo["has_data"],
            bypass_choppy=cfg.ar_bypass_choppy,
            threshold_reduction=cfg.ar_threshold_reduction,
            confidence_boost=conf_boost,
            diagnostics={
                "absorption_rate": round(absorption, 4),
                "signed_aggression": round(signed_aggr, 4),
                "book_pressure": round(book_pressure, 4),
                "level": nearest,
            },
        )

    # -------------------------------------------------------------------
    # Setup 2: Gamma Squeeze Breakout
    # -------------------------------------------------------------------
    def _detect_gamma_squeeze(
        self,
        cfg: GoldenSetupConfig,
        bar: Any,
        bars: list,
        flow: Dict[str, Any],
        levels: list,
        price: float,
        tcbbo: Dict[str, float],
        direction: str,
    ) -> Optional[GoldenSetupMatch]:
        """
        Bullish: ask-side thinning + buying aggression near resistance.
        Bearish: bid-side thinning + selling aggression near support.
        """
        signed_aggr = _safe_float(flow.get("signed_aggression"))
        book_pressure = _safe_float(flow.get("book_pressure_avg"))
        bp_change = _safe_float(flow.get("book_pressure_trend"))

        if direction == "bullish":
            level_kind = "resistance"
            aggr_ok = signed_aggr >= cfg.gsb_min_signed_aggression
            bp_increasing = bp_change >= cfg.gsb_min_book_pressure_increase
            # Ask-side thinning
            cur_depth = _safe_float(getattr(bar, "l2_ask_depth_total", None))
            avg_depth = _avg_depth(bars, "l2_ask_depth_total", 5)
        else:
            level_kind = "support"
            aggr_ok = signed_aggr <= -cfg.gsb_min_signed_aggression
            bp_increasing = bp_change <= -cfg.gsb_min_book_pressure_increase
            cur_depth = _safe_float(getattr(bar, "l2_bid_depth_total", None))
            avg_depth = _avg_depth(bars, "l2_bid_depth_total", 5)

        nearest = _find_nearest_level(levels, price, level_kind, cfg.gsb_level_tolerance_pct)
        if not nearest:
            return None

        if not aggr_ok:
            return None

        # Depth decline check
        depth_decline_pct = 0.0
        if avg_depth > 0 and cur_depth >= 0:
            depth_decline_pct = ((avg_depth - cur_depth) / avg_depth) * 100.0
        if depth_decline_pct < cfg.gsb_ask_depth_decline_pct:
            return None

        # Score
        aggr_score = _clamp01(abs(signed_aggr) / 0.15)
        decline_score = _clamp01(depth_decline_pct / 30.0)
        bp_score = _clamp01(abs(bp_change) / 0.10) if bp_increasing else 0.0
        proximity_score = _clamp01(1.0 - nearest["distance_pct"] / max(cfg.gsb_level_tolerance_pct, 0.01))
        l2_score = 100.0 * (
            0.30 * aggr_score
            + 0.30 * decline_score
            + 0.20 * bp_score
            + 0.20 * proximity_score
        )

        # TCBBO
        tcbbo_score = 0.0
        if tcbbo["has_data"]:
            if direction == "bullish":
                sweep_ok = tcbbo["sweep_count"] >= cfg.gsb_min_tcbbo_sweep_count
                call_dominant = tcbbo["call_buy_premium"] > tcbbo["put_buy_premium"] * 2.0
            else:
                sweep_ok = tcbbo["sweep_count"] >= cfg.gsb_min_tcbbo_sweep_count
                call_dominant = tcbbo["put_buy_premium"] > tcbbo["call_buy_premium"] * 2.0
            if sweep_ok:
                tcbbo_score += 60.0
            if call_dominant:
                tcbbo_score += 40.0

        conf_boost = cfg.gsb_confidence_boost_l2
        if tcbbo_score > 0:
            conf_boost += cfg.gsb_confidence_boost_tcbbo * (tcbbo_score / 100.0)

        return GoldenSetupMatch(
            setup_type=GoldenSetupType.GAMMA_SQUEEZE,
            direction=direction,
            l2_score=l2_score,
            tcbbo_score=tcbbo_score,
            has_tcbbo=tcbbo["has_data"],
            bypass_choppy=cfg.gsb_bypass_choppy,
            threshold_reduction=cfg.gsb_threshold_reduction,
            confidence_boost=conf_boost,
            diagnostics={
                "signed_aggression": round(signed_aggr, 4),
                "depth_decline_pct": round(depth_decline_pct, 2),
                "book_pressure_change": round(bp_change, 4),
                "level": nearest,
            },
        )

    # -------------------------------------------------------------------
    # Setup 3: Liquidity Trap
    # -------------------------------------------------------------------
    def _detect_liquidity_trap(
        self,
        cfg: GoldenSetupConfig,
        bar: Any,
        bars: list,
        flow: Dict[str, Any],
        levels: list,
        price: float,
        tcbbo: Dict[str, float],
        direction: str,
    ) -> Optional[GoldenSetupMatch]:
        """
        Bullish: false breakout below support + delta flip from selling to buying.
        Bearish: false breakout above resistance + delta flip from buying to selling.
        """
        if len(bars) < 3:
            return None

        signed_aggr = _safe_float(flow.get("signed_aggression"))
        delta_accel = _safe_float(flow.get("delta_acceleration"))

        # Need previous bar's signed aggression to detect the flip
        prev_bar = bars[-2]
        prev_aggr = _safe_float(getattr(prev_bar, "l2_imbalance", None))

        if direction == "bullish":
            level_kind = "support"
            # Delta flip: was negative, now positive or accelerating
            flip_ok = prev_aggr < -0.02 and (signed_aggr > prev_aggr or delta_accel > cfg.lt_min_delta_acceleration)
        else:
            level_kind = "resistance"
            flip_ok = prev_aggr > 0.02 and (signed_aggr < prev_aggr or delta_accel < -cfg.lt_min_delta_acceleration)

        if not flip_ok:
            return None

        # False breakout: price beyond level by a small amount
        # Find levels including slightly breached ones
        best_level = None
        best_breach = None
        for row in levels:
            if not isinstance(row, dict):
                continue
            lp = _safe_float(row.get("price"))
            if lp <= 0.0:
                continue
            kind = str(row.get("kind", "")).strip().lower()
            if kind not in {"support", "resistance"}:
                kind = "support" if lp <= price else "resistance"
            if kind != level_kind:
                continue
            breach_pct = 0.0
            if direction == "bullish" and price < lp:
                breach_pct = ((lp - price) / lp) * 100.0
            elif direction == "bearish" and price > lp:
                breach_pct = ((price - lp) / lp) * 100.0
            else:
                # Also count being within tolerance (just returned from breach)
                dist = _level_distance_pct(price, lp)
                if dist <= cfg.lt_level_tolerance_pct * 0.5:
                    breach_pct = 0.05  # nominal small breach
                else:
                    continue

            if cfg.lt_false_breakout_min_pct <= breach_pct <= cfg.lt_false_breakout_max_pct:
                if best_breach is None or breach_pct < best_breach:
                    best_breach = breach_pct
                    best_level = {
                        "price": lp,
                        "kind": kind,
                        "source": str(row.get("source", "")),
                        "breach_pct": round(breach_pct, 4),
                    }

        if not best_level:
            return None

        # Score
        flip_magnitude = abs(signed_aggr - prev_aggr)
        flip_score = _clamp01(flip_magnitude / 0.15)
        accel_score = _clamp01(abs(delta_accel) / 0.05)
        breach_score = _clamp01(1.0 - (best_breach - cfg.lt_false_breakout_min_pct) / max(
            cfg.lt_false_breakout_max_pct - cfg.lt_false_breakout_min_pct, 0.01
        ))
        l2_score = 100.0 * (
            0.40 * flip_score
            + 0.30 * accel_score
            + 0.30 * breach_score
        )

        # TCBBO
        tcbbo_score = 0.0
        if tcbbo["has_data"]:
            if direction == "bullish":
                net_ok = tcbbo["net_premium"] > 0
                put_selling = tcbbo["put_sell_premium"] > tcbbo["put_buy_premium"]
            else:
                net_ok = tcbbo["net_premium"] < 0
                put_selling = tcbbo["call_sell_premium"] > tcbbo["call_buy_premium"]
            if net_ok:
                tcbbo_score += 50.0
            if put_selling:
                tcbbo_score += 50.0

        conf_boost = cfg.lt_confidence_boost_l2
        if tcbbo_score > 0:
            conf_boost += cfg.lt_confidence_boost_tcbbo * (tcbbo_score / 100.0)

        return GoldenSetupMatch(
            setup_type=GoldenSetupType.LIQUIDITY_TRAP,
            direction=direction,
            l2_score=l2_score,
            tcbbo_score=tcbbo_score,
            has_tcbbo=tcbbo["has_data"],
            bypass_choppy=cfg.lt_bypass_choppy,
            threshold_reduction=cfg.lt_threshold_reduction,
            confidence_boost=conf_boost,
            diagnostics={
                "signed_aggression": round(signed_aggr, 4),
                "prev_aggression": round(prev_aggr, 4),
                "delta_acceleration": round(delta_accel, 4),
                "flip_magnitude": round(flip_magnitude, 4),
                "level": best_level,
            },
        )

    # -------------------------------------------------------------------
    # Setup 4: Iceberg Defense
    # -------------------------------------------------------------------
    def _detect_iceberg_defense(
        self,
        cfg: GoldenSetupConfig,
        bar: Any,
        bars: list,
        flow: Dict[str, Any],
        levels: list,
        price: float,
        tcbbo: Dict[str, float],
        direction: str,
    ) -> Optional[GoldenSetupMatch]:
        """
        Bullish: iceberg buy orders reloading at support, high volume with no price movement.
        Bearish: iceberg sell orders reloading at resistance, high volume with no movement.
        """
        absorption = _safe_float(flow.get("absorption_rate"))
        book_pressure = _safe_float(flow.get("book_pressure_avg"))
        price_change = _safe_float(flow.get("price_change_pct"))

        if direction == "bullish":
            level_kind = "support"
            iceberg_count = int(_safe_float(getattr(bar, "l2_iceberg_buy_count", 0)))
            bp_ok = book_pressure >= cfg.id_min_book_pressure
        else:
            level_kind = "resistance"
            iceberg_count = int(_safe_float(getattr(bar, "l2_iceberg_sell_count", 0)))
            bp_ok = book_pressure <= -cfg.id_min_book_pressure

        nearest = _find_nearest_level(levels, price, level_kind, cfg.id_level_tolerance_pct)
        if not nearest:
            return None

        # Core conditions
        if iceberg_count < cfg.id_min_iceberg_count:
            return None
        if absorption < cfg.id_min_absorption_rate:
            return None
        if not bp_ok:
            return None
        # High volume, minimal movement
        if abs(price_change) > 0.10:
            return None

        # Score
        iceberg_score = _clamp01(iceberg_count / 3.0)
        absorption_score = _clamp01(absorption / max(cfg.id_min_absorption_rate * 2.0, 0.01))
        bp_score = _clamp01(abs(book_pressure) / 0.15)
        stillness_score = _clamp01(1.0 - abs(price_change) / 0.10)
        proximity_score = _clamp01(1.0 - nearest["distance_pct"] / max(cfg.id_level_tolerance_pct, 0.01))
        l2_score = 100.0 * (
            0.25 * iceberg_score
            + 0.25 * absorption_score
            + 0.15 * bp_score
            + 0.15 * stillness_score
            + 0.20 * proximity_score
        )

        # TCBBO: absence of bearish flow
        tcbbo_score = 0.0
        if tcbbo["has_data"]:
            if direction == "bullish":
                put_absent = tcbbo["put_buy_premium"] < max(tcbbo["call_buy_premium"] * 0.3, 1000.0)
                net_ok = tcbbo["net_premium"] >= 0
            else:
                put_absent = tcbbo["call_buy_premium"] < max(tcbbo["put_buy_premium"] * 0.3, 1000.0)
                net_ok = tcbbo["net_premium"] <= 0
            if put_absent:
                tcbbo_score += 60.0
            if net_ok:
                tcbbo_score += 40.0

        conf_boost = cfg.id_confidence_boost_l2
        if tcbbo_score > 0:
            conf_boost += cfg.id_confidence_boost_tcbbo * (tcbbo_score / 100.0)

        return GoldenSetupMatch(
            setup_type=GoldenSetupType.ICEBERG_DEFENSE,
            direction=direction,
            l2_score=l2_score,
            tcbbo_score=tcbbo_score,
            has_tcbbo=tcbbo["has_data"],
            bypass_choppy=cfg.id_bypass_choppy,
            threshold_reduction=cfg.id_threshold_reduction,
            confidence_boost=conf_boost,
            diagnostics={
                "iceberg_count": iceberg_count,
                "absorption_rate": round(absorption, 4),
                "book_pressure": round(book_pressure, 4),
                "price_change_pct": round(price_change, 4),
                "level": nearest,
            },
        )

    # -------------------------------------------------------------------
    # Setup 5: Fuel Injection
    # -------------------------------------------------------------------
    def _detect_fuel_injection(
        self,
        cfg: GoldenSetupConfig,
        bar: Any,
        bars: list,
        flow: Dict[str, Any],
        levels: list,
        price: float,
        vwap: Optional[float],
        tcbbo: Dict[str, float],
        direction: str,
    ) -> Optional[GoldenSetupMatch]:
        """
        Bullish: uptrend (above VWAP), not near resistance, ask-side thinning, consolidation.
        Bearish: downtrend (below VWAP), not near support, bid-side thinning, consolidation.
        """
        signed_aggr = _safe_float(flow.get("signed_aggression"))

        # Trend context via VWAP
        if vwap is None or vwap <= 0.0:
            return None
        if direction == "bullish":
            trend_ok = price > vwap
            aggr_ok = signed_aggr >= cfg.fi_min_signed_aggression
            opposite_kind = "resistance"
            cur_depth = _safe_float(getattr(bar, "l2_ask_depth_total", None))
            avg_depth = _avg_depth(bars, "l2_ask_depth_total", 5)
        else:
            trend_ok = price < vwap
            aggr_ok = signed_aggr <= -cfg.fi_min_signed_aggression
            opposite_kind = "support"
            cur_depth = _safe_float(getattr(bar, "l2_bid_depth_total", None))
            avg_depth = _avg_depth(bars, "l2_bid_depth_total", 5)

        if not trend_ok:
            return None
        if not aggr_ok:
            return None

        # NOT near resistance/support (should be "in the air")
        near_level = _find_nearest_level(levels, price, opposite_kind, cfg.fi_level_tolerance_pct)
        if near_level is not None:
            return None  # too close to level — not a fuel injection setup

        # Depth decline
        depth_decline_pct = 0.0
        if avg_depth > 0 and cur_depth >= 0:
            depth_decline_pct = ((avg_depth - cur_depth) / avg_depth) * 100.0
        if depth_decline_pct < cfg.fi_ask_depth_decline_pct:
            return None

        # Consolidation: low bar range
        bar_range = abs(float(getattr(bar, "high", 0) or 0) - float(getattr(bar, "low", 0) or 0))
        bar_range_pct = (bar_range / price) * 100.0 if price > 0 else 999.0
        if bar_range_pct > cfg.fi_max_bar_range_pct:
            return None

        # Score
        decline_score = _clamp01(depth_decline_pct / 25.0)
        aggr_score = _clamp01(abs(signed_aggr) / 0.10)
        consolidation_score = _clamp01(1.0 - bar_range_pct / max(cfg.fi_max_bar_range_pct, 0.01))
        vwap_dist = abs(price - vwap) / price * 100.0
        trend_score = _clamp01(vwap_dist / 0.30)
        l2_score = 100.0 * (
            0.35 * decline_score
            + 0.25 * aggr_score
            + 0.20 * consolidation_score
            + 0.20 * trend_score
        )

        # TCBBO
        tcbbo_score = 0.0
        if tcbbo["has_data"]:
            if direction == "bullish":
                sweep_ok = tcbbo["sweep_count"] > 0
                premium_ok = tcbbo["call_buy_premium"] > tcbbo["put_buy_premium"]
            else:
                sweep_ok = tcbbo["sweep_count"] > 0
                premium_ok = tcbbo["put_buy_premium"] > tcbbo["call_buy_premium"]
            if sweep_ok:
                tcbbo_score += 60.0
            if premium_ok:
                tcbbo_score += 40.0

        conf_boost = cfg.fi_confidence_boost_l2
        if tcbbo_score > 0:
            conf_boost += cfg.fi_confidence_boost_tcbbo * (tcbbo_score / 100.0)

        return GoldenSetupMatch(
            setup_type=GoldenSetupType.FUEL_INJECTION,
            direction=direction,
            l2_score=l2_score,
            tcbbo_score=tcbbo_score,
            has_tcbbo=tcbbo["has_data"],
            bypass_choppy=cfg.fi_bypass_choppy,
            threshold_reduction=cfg.fi_threshold_reduction,
            confidence_boost=conf_boost,
            diagnostics={
                "signed_aggression": round(signed_aggr, 4),
                "depth_decline_pct": round(depth_decline_pct, 2),
                "bar_range_pct": round(bar_range_pct, 4),
                "vwap_distance_pct": round(vwap_dist, 4),
            },
        )
