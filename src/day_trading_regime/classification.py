"""Micro-regime classification helpers."""

from __future__ import annotations

from typing import Dict, Optional

from ..strategies.base_strategy import Regime

def regime_safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator

def regime_map_micro_to_regime(self, micro_regime: str) -> Regime:
    if micro_regime in {"TRENDING_UP", "TRENDING_DOWN", "BREAKOUT"}:
        return Regime.TRENDING
    if micro_regime in {"CHOPPY", "ABSORPTION"}:
        return Regime.CHOPPY
    # TRANSITION maps to MIXED at macro but retains micro detail
    return Regime.MIXED

def regime_map_adaptive_regime(primary: str) -> Regime:
    """Map AdaptiveRegimeDetector primary label to legacy Regime enum."""
    return {'TRENDING': Regime.TRENDING, 'CHOPPY': Regime.CHOPPY}.get(primary, Regime.MIXED)

def regime_classify_micro_regime(
    self,
    trend_efficiency: float,
    adx: Optional[float],
    volatility: float,
    price_change_pct: float,
    flow: Dict[str, float],
) -> str:
    """Map price + order-flow behavior to a richer intraday micro-regime."""
    signed_aggr = float(flow.get("signed_aggression", 0.0))
    consistency = float(flow.get("directional_consistency", 0.0))
    sweep_intensity = float(flow.get("sweep_intensity", 0.0))
    absorption_rate = float(flow.get("absorption_rate", 0.0))
    book_pressure_avg = float(flow.get("book_pressure_avg", 0.0))
    large_trader_activity = float(flow.get("large_trader_activity", 0.0))
    flow_price_change_pct = float(flow.get("price_change_pct", price_change_pct) or price_change_pct)
    realized_volatility_pct = abs(float(flow.get("realized_volatility_pct", 0.0) or 0.0))
    flow_has_coverage = bool(flow.get("has_l2_coverage", False))

    # If ADX is not fully warmed up, default to MIXED so we can still initialize
    # with partial bars and allow signal generation instead of pausing entirely.
    if adx is None:
        return "MIXED"
    adx_val = float(adx)

    # Normalize directional signal: tiny absolute price drift should not decide direction.
    directional_noise_floor_pct = max(realized_volatility_pct * 0.25, 0.01)
    directional_price_change_pct = (
        flow_price_change_pct if abs(flow_price_change_pct) >= directional_noise_floor_pct else 0.0
    )

    if flow_has_coverage:
        trend_eff_threshold = 0.45 if adx_val > 35.0 else 0.58

        if (
            sweep_intensity >= 0.30
            and consistency >= 0.55
            and abs(signed_aggr) >= 0.10
            and large_trader_activity >= 0.15
        ):
            return "BREAKOUT"
        if (
            absorption_rate >= 0.50
            and abs(book_pressure_avg) >= 0.10
            and abs(flow_price_change_pct) <= max(0.12, volatility * 100.0 * 0.8)
        ):
            return "ABSORPTION"

        if (
            trend_efficiency >= trend_eff_threshold
            and adx_val >= 25.0
            and signed_aggr >= 0.08
            and directional_price_change_pct > 0.0
        ):
            return "TRENDING_UP"
        if (
            trend_efficiency >= trend_eff_threshold
            and adx_val >= 25.0
            and signed_aggr <= -0.08
            and directional_price_change_pct < 0.0
        ):
            return "TRENDING_DOWN"

        if adx_val < 20.0 and trend_efficiency < 0.40:
            return "CHOPPY"

        # TRANSITION: trend indicators suggest direction but efficiency is very low
        # This indicates noisy/unreliable trend conditions
        if trend_efficiency < 0.15 and adx_val >= 15.0:
            return "TRANSITION"

        return "MIXED"

    # Price-only branch without L2 coverage.
    if adx_val >= 40.0:
        no_l2_trend_eff_threshold = 0.30
    elif adx_val >= 30.0:
        no_l2_trend_eff_threshold = 0.35
    else:
        no_l2_trend_eff_threshold = 0.45
    if (
        trend_efficiency >= no_l2_trend_eff_threshold
        and adx_val > 30.0
        and directional_price_change_pct != 0.0
    ):
        return "TRENDING_UP" if directional_price_change_pct > 0.0 else "TRENDING_DOWN"

    # TRANSITION: ADX suggests trend but efficiency is very low (no L2 coverage)
    if trend_efficiency < 0.15 and adx_val >= 20.0:
        return "TRANSITION"

    if adx_val < 20.0:
        return "CHOPPY"
    return "MIXED"
