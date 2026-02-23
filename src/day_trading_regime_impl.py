"""Regime and strategy-selection method implementations extracted from DayTradingManager."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .day_trading_models import BarData, TradingSession
from .strategies.base_strategy import Regime

from .day_trading_regime.classification import (
    regime_classify_micro_regime as _regime_classify_micro_regime_impl,
    regime_map_adaptive_regime as _regime_map_adaptive_regime_impl,
    regime_map_micro_to_regime as _regime_map_micro_to_regime_impl,
    regime_safe_div as _regime_safe_div_impl,
)
from .day_trading_regime.detection import (
    regime_detect_regime as _regime_detect_regime_impl,
)
from .day_trading_regime.flow_metrics import (
    _calculate_window_flow_components as _calculate_window_flow_components_impl,
    _flow_score_from_components as _flow_score_from_components_impl,
    regime_calculate_order_flow_metrics as _regime_calculate_order_flow_metrics_impl,
)
from .day_trading_regime.refresh import (
    regime_maybe_refresh_regime as _regime_maybe_refresh_regime_impl,
)
from .day_trading_regime.selection import (
    regime_build_momentum_route_candidates as _regime_build_momentum_route_candidates_impl,
    regime_resolve_momentum_diversification as _regime_resolve_momentum_diversification_impl,
    regime_select_momentum_sleeve as _regime_select_momentum_sleeve_impl,
    regime_select_strategies as _regime_select_strategies_impl,
)
from .day_trading_regime.trend_indicators import (
    regime_calc_adx as _regime_calc_adx_impl,
    regime_calc_adx_series as _regime_calc_adx_series_impl,
    regime_calc_trend_efficiency as _regime_calc_trend_efficiency_impl,
    regime_calc_volatility as _regime_calc_volatility_impl,
)

def regime_detect_regime(self, session: TradingSession) -> Regime:
    return _regime_detect_regime_impl(
        self=self,
        session=session,
    )


def regime_safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return _regime_safe_div_impl(
        numerator=numerator,
        denominator=denominator,
        default=default,
    )


def regime_map_micro_to_regime(self, micro_regime: str) -> Regime:
    return _regime_map_micro_to_regime_impl(
        self=self,
        micro_regime=micro_regime,
    )


def regime_map_adaptive_regime(primary: str) -> Regime:
    return _regime_map_adaptive_regime_impl(primary=primary)


def regime_classify_micro_regime(
    self,
    trend_efficiency: float,
    adx: Optional[float],
    volatility: float,
    price_change_pct: float,
    flow: Dict[str, float],
) -> str:
    return _regime_classify_micro_regime_impl(
        self=self,
        trend_efficiency=trend_efficiency,
        adx=adx,
        volatility=volatility,
        price_change_pct=price_change_pct,
        flow=flow,
    )


def _flow_score_from_components(
    self,
    *,
    cumulative_delta: float,
    deltas: List[float],
    directional_consistency: float,
    avg_imbalance: float,
    sweep_intensity: float,
    participation_ratio: float,
    large_trader_activity: float,
    vwap_execution_flow: float,
    book_pressure_avg: float,
) -> float:
    return _flow_score_from_components_impl(
        self=self,
        cumulative_delta=cumulative_delta,
        deltas=deltas,
        directional_consistency=directional_consistency,
        avg_imbalance=avg_imbalance,
        sweep_intensity=sweep_intensity,
        participation_ratio=participation_ratio,
        large_trader_activity=large_trader_activity,
        vwap_execution_flow=vwap_execution_flow,
        book_pressure_avg=book_pressure_avg,
    )


def _calculate_window_flow_components(self, window: List[BarData]) -> Dict[str, float]:
    return _calculate_window_flow_components_impl(
        self=self,
        window=window,
    )


def regime_calculate_order_flow_metrics(
    self,
    bars: List[BarData],
    lookback: int = 20,
) -> Dict[str, float]:
    return _regime_calculate_order_flow_metrics_impl(
        self=self,
        bars=bars,
        lookback=lookback,
    )


def regime_resolve_momentum_diversification(
    self,
    session: TradingSession,
    adaptive_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    return _regime_resolve_momentum_diversification_impl(
        self=self,
        session=session,
        adaptive_cfg=adaptive_cfg,
    )


def regime_select_momentum_sleeve(
    self,
    momentum_cfg: Dict[str, Any],
    *,
    strategy_key: str = "",
    micro_regime: str = "MIXED",
    has_l2_coverage: Optional[bool] = None,
    preferred_sleeve_id: str = "",
) -> Tuple[Dict[str, Any], Optional[str], str]:
    return _regime_select_momentum_sleeve_impl(
        self=self,
        momentum_cfg=momentum_cfg,
        strategy_key=strategy_key,
        micro_regime=micro_regime,
        has_l2_coverage=has_l2_coverage,
        preferred_sleeve_id=preferred_sleeve_id,
    )


def regime_build_momentum_route_candidates(
    self,
    session: TradingSession,
    flow_metrics: Dict[str, Any],
    momentum_cfg: Dict[str, Any],
) -> List[str]:
    return _regime_build_momentum_route_candidates_impl(
        self=self,
        session=session,
        flow_metrics=flow_metrics,
        momentum_cfg=momentum_cfg,
    )


def regime_select_strategies(self, session: TradingSession) -> List[str]:
    return _regime_select_strategies_impl(
        self=self,
        session=session,
    )


def regime_calc_trend_efficiency(self, bars: List[BarData]) -> float:
    return _regime_calc_trend_efficiency_impl(
        self=self,
        bars=bars,
    )


def regime_calc_volatility(self, bars: List[BarData]) -> float:
    return _regime_calc_volatility_impl(
        self=self,
        bars=bars,
    )


def regime_calc_adx(self, bars: List[BarData], period: int = 14) -> Optional[float]:
    return _regime_calc_adx_impl(
        self=self,
        bars=bars,
        period=period,
    )


def regime_calc_adx_series(self, bars: List[BarData], period: int = 14) -> List[float]:
    return _regime_calc_adx_series_impl(
        self=self,
        bars=bars,
        period=period,
    )

def regime_maybe_refresh_regime(
    self,
    session: TradingSession,
    current_bar_index: int,
    timestamp: datetime,
) -> Optional[Dict[str, Any]]:
    return _regime_maybe_refresh_regime_impl(
        self=self,
        session=session,
        current_bar_index=current_bar_index,
        timestamp=timestamp,
    )
