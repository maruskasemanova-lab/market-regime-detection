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
    l2_iceberg_buy_count: Optional[float] = None
    l2_iceberg_sell_count: Optional[float] = None
    l2_iceberg_bias: Optional[float] = None


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


class MultiLayerConfig(BaseModel):
    """Configuration for the multi-layer decision engine."""

    pattern_weight: Optional[float] = None
    strategy_weight: Optional[float] = None
    threshold: Optional[float] = None
    require_pattern: Optional[bool] = None
    # Candlestick pattern detector settings
    body_doji_pct: Optional[float] = None
    wick_ratio_hammer: Optional[float] = None
    engulfing_min_body_pct: Optional[float] = None
    volume_confirm_ratio: Optional[float] = None
    vwap_proximity_pct: Optional[float] = None

