"""
Pydantic API models shared by strategy API endpoints.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel


class StrategyToggle(BaseModel):
    strategy_name: str
    enabled: bool


class StrategyUpdate(BaseModel):
    strategy_name: str
    params: Dict[str, Any]


class TrailingStopConfig(BaseModel):
    stop_type: str = "PERCENT"
    initial_stop_pct: float = 2.0
    trailing_pct: float = 0.8
    atr_multiplier: float = 2.0


class BarInput(BaseModel):
    """Input model for processing a single bar."""

    run_id: str
    ticker: str
    timestamp: str  # ISO format datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    l2_delta: Optional[float] = None
    l2_buy_volume: Optional[float] = None
    l2_sell_volume: Optional[float] = None
    l2_volume: Optional[float] = None
    l2_imbalance: Optional[float] = None
    l2_bid_depth_total: Optional[float] = None
    l2_ask_depth_total: Optional[float] = None
    l2_book_pressure: Optional[float] = None
    l2_book_pressure_change: Optional[float] = None
    l2_iceberg_buy_count: Optional[float] = None
    l2_iceberg_sell_count: Optional[float] = None
    l2_iceberg_bias: Optional[float] = None
    # Cross-asset reference bar (optional, e.g. QQQ)
    ref_ticker: Optional[str] = None
    ref_open: Optional[float] = None
    ref_high: Optional[float] = None
    ref_low: Optional[float] = None
    ref_close: Optional[float] = None
    ref_volume: Optional[float] = None


class SessionQuery(BaseModel):
    """Query model for session operations."""

    run_id: str
    ticker: str
    date: str  # YYYY-MM-DD format


class TradingConfig(BaseModel):
    """Global trading configuration."""

    regime_detection_minutes: int = 60
    regime_refresh_bars: int = 12
    max_daily_loss: float = 300.0
    max_trades_per_day: int = 3
    trade_cooldown_bars: int = 15
    account_size_usd: float = 10000.0
    risk_per_trade_pct: float = 1.0
    max_position_notional_pct: float = 100.0
    max_fill_participation_rate: float = 0.20
    min_fill_ratio: float = 0.35
    enable_partial_take_profit: bool = True
    partial_take_profit_rr: float = 1.0
    partial_take_profit_fraction: float = 0.5
    time_exit_bars: int = 40
    adverse_flow_exit_enabled: bool = True
    adverse_flow_threshold: float = 0.12
    adverse_flow_min_hold_bars: int = 3


class MultiLayerConfig(BaseModel):
    """Configuration for the multi-layer decision engine."""

    pattern_weight: Optional[float] = None
    strategy_weight: Optional[float] = None
    threshold: Optional[float] = None
    strategy_only_threshold: Optional[float] = None
    require_pattern: Optional[bool] = None
    # Candlestick pattern detector settings
    body_doji_pct: Optional[float] = None
    wick_ratio_hammer: Optional[float] = None
    engulfing_min_body_pct: Optional[float] = None
    volume_confirm_ratio: Optional[float] = None
    vwap_proximity_pct: Optional[float] = None
