"""
Session-based Day Trading Manager.
Manages trading sessions for individual days with regime detection and strategy execution.
"""
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import json
import logging
import math
import re

logger = logging.getLogger(__name__)

from .strategies.base_strategy import BaseStrategy, Signal, SignalType, Position, Regime
from .multi_layer_decision import MultiLayerDecision
from .strategy_factory import build_strategy_registry


class SessionPhase(Enum):
    """Trading session phases."""
    PRE_MARKET = "PRE_MARKET"           # Before 9:30 ET
    REGIME_DETECTION = "REGIME_DETECTION"  # First X minutes
    TRADING = "TRADING"                  # Active trading
    END_OF_DAY = "END_OF_DAY"           # Closing positions
    CLOSED = "CLOSED"                    # Session ended


@dataclass
class BarData:
    """Single bar data."""
    timestamp: datetime
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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'vwap': self.vwap,
            'l2_delta': self.l2_delta,
            'l2_buy_volume': self.l2_buy_volume,
            'l2_sell_volume': self.l2_sell_volume,
            'l2_volume': self.l2_volume,
            'l2_imbalance': self.l2_imbalance,
            'l2_bid_depth_total': self.l2_bid_depth_total,
            'l2_ask_depth_total': self.l2_ask_depth_total,
            'l2_book_pressure': self.l2_book_pressure,
            'l2_book_pressure_change': self.l2_book_pressure_change,
            'l2_iceberg_buy_count': self.l2_iceberg_buy_count,
            'l2_iceberg_sell_count': self.l2_iceberg_sell_count,
            'l2_iceberg_bias': self.l2_iceberg_bias,
        }


@dataclass
class DayTrade:
    """Completed trade for the day."""
    id: int
    strategy: str
    side: str
    entry_price: float
    entry_time: datetime
    exit_price: float
    exit_time: datetime
    size: float
    pnl_pct: float
    pnl_dollars: float
    exit_reason: str
    # Trading costs breakdown
    slippage: float = 0.0
    commission: float = 0.0
    reg_fee: float = 0.0
    sec_fee: float = 0.0
    finra_fee: float = 0.0
    market_impact: float = 0.0
    total_costs: float = 0.0
    gross_pnl_pct: float = 0.0  # PnL before costs
    signal_bar_index: Optional[int] = None
    entry_bar_index: Optional[int] = None
    signal_timestamp: Optional[str] = None
    signal_price: Optional[float] = None
    signal_metadata: Dict[str, Any] = field(default_factory=dict)
    flow_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'strategy': self.strategy,
            'side': self.side,
            'entry_price': round(self.entry_price, 2),
            'entry_time': self.entry_time.isoformat(),
            'exit_price': round(self.exit_price, 2),
            'exit_time': self.exit_time.isoformat(),
            'size': self.size,
            'pnl_pct': round(self.pnl_pct, 2),
            'pnl_dollars': round(self.pnl_dollars, 2),
            'exit_reason': self.exit_reason,
            'costs': {
                'slippage': round(self.slippage, 4),
                'commission': round(self.commission, 4),
                'reg_fee': round(self.reg_fee, 4),
                'sec_fee': round(self.sec_fee, 6),
                'finra_fee': round(self.finra_fee, 6),
                'market_impact': round(self.market_impact, 4),
                'total': round(self.total_costs, 4)
            },
            'gross_pnl_pct': round(self.gross_pnl_pct, 2),
            'signal_bar_index': self.signal_bar_index,
            'entry_bar_index': self.entry_bar_index,
            'signal_timestamp': self.signal_timestamp,
            'signal_price': round(self.signal_price, 4) if self.signal_price is not None else None,
            'signal_metadata': self.signal_metadata,
            'flow_snapshot': self.flow_snapshot,
        }


@dataclass
class TradingCosts:
    """Trading costs (IBKR Tiered, liquid US large-caps, Feb 2026).
    Model:
      - Commission: $0.0035/share with $0.35 minimum per order (per side).
      - Slippage: $0.01 per share round-trip (0.5¢ per side through the spread).
      - FINRA TAF: SELL side only, min $0.01, cap $9.79.
      - SEC fee: optional SELL-side notional fee (default 0).
      - reg_fee: optional pass-through per share (default 0).
    """
    commission_per_share: float = 0.0035
    commission_min: float = 0.35      # per order (per side)
    # 0.5¢ per side so round‑trip slippage totals 1¢ per share.
    slippage_per_share: float = 0.005
    # Slippage and impact realism controls.
    low_volume_threshold_shares: float = 25_000.0
    low_volume_slippage_multiplier: float = 1.75
    impact_coeff_bps: float = 6.0
    impact_participation_cap: float = 0.20
    finra_fee_per_share: float = 0.000195
    finra_fee_cap: float = 9.79
    finra_fee_min: float = 0.01
    sec_fee_per_dollar: float = 0.0   # SELL-side notional, default 0
    reg_fee_per_share: float = 0.0    # optional pass-through, SELL side only

    def calculate_costs(
        self,
        entry_price: float,
        exit_price: float,
        shares: float,
        side: str,
        avg_bar_volume: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Calculate all trading costs.
        Returns dict with commission, reg_fee, slippage, finra_fee, total.
        """
        # Estimate participation to model liquidity-dependent slippage/impact.
        avg_volume = float(avg_bar_volume or 0.0)
        participation_ratio = shares / avg_volume if avg_volume > 0 else 0.0

        volume_multiplier = 1.0
        if avg_volume > 0 and avg_volume < self.low_volume_threshold_shares:
            shortage = (self.low_volume_threshold_shares - avg_volume) / self.low_volume_threshold_shares
            volume_multiplier += shortage * max(0.0, self.low_volume_slippage_multiplier - 1.0)

        participation_multiplier = 1.0 + max(0.0, participation_ratio - 0.02) * 6.0
        dynamic_slippage_per_share = self.slippage_per_share * volume_multiplier * participation_multiplier
        slippage_cost = dynamic_slippage_per_share * shares * 2

        # Commission per side with minimum
        entry_commission = max(self.commission_per_share * shares, self.commission_min)
        exit_commission = max(self.commission_per_share * shares, self.commission_min)
        commission = entry_commission + exit_commission

        # SELL-side only fees (long -> sell on exit, short -> sell on entry)
        sell_price = exit_price if side == "long" else entry_price
        sell_notional = sell_price * shares

        # FINRA TAF: per share, apply cap and minimum (SELL side only)
        raw_finra = shares * self.finra_fee_per_share
        finra_fee = max(self.finra_fee_min, min(raw_finra, self.finra_fee_cap))

        # SEC fee (SELL side only; default 0)
        sec_fee = sell_notional * self.sec_fee_per_dollar

        # Optional pass-through reg fee (SELL side only; default 0)
        reg_fee = shares * self.reg_fee_per_share

        impact_ratio = min(
            1.0,
            participation_ratio / max(1e-6, float(self.impact_participation_cap)),
        )
        avg_notional = ((entry_price + exit_price) / 2.0) * shares
        market_impact = avg_notional * ((self.impact_coeff_bps * impact_ratio) / 10_000.0)

        total = slippage_cost + commission + reg_fee + finra_fee + sec_fee + market_impact
        
        return {
            'slippage': slippage_cost,
            'commission': commission,
            'reg_fee': reg_fee,
            'sec_fee': sec_fee,
            'finra_fee': finra_fee,
            'market_impact': market_impact,
            'dynamic_slippage_per_share': dynamic_slippage_per_share,
            'participation_ratio': participation_ratio,
            'total': total
        }


@dataclass
class TradingSession:
    """Single day trading session for a ticker."""
    run_id: str
    ticker: str
    date: str
    regime_detection_minutes: int = 15
    regime_refresh_bars: int = 12
    
    # Session state
    phase: SessionPhase = SessionPhase.PRE_MARKET
    detected_regime: Optional[Regime] = None
    micro_regime: str = "MIXED"
    active_strategies: List[str] = field(default_factory=list)
    selected_strategy: Optional[str] = None  # Helper for backwards compat logic
    
    # Data storage
    bars: List[BarData] = field(default_factory=list)
    pre_market_bars: List[BarData] = field(default_factory=list)
    
    # Trading state
    active_position: Optional[Position] = None
    trades: List[DayTrade] = field(default_factory=list)
    signals: List[Signal] = field(default_factory=list)
    trade_counter: int = 0
    pending_signal: Optional[Signal] = None
    pending_signal_bar_index: int = -1
    last_exit_bar_index: int = -1
    last_regime_refresh_bar_index: int = -1
    regime_history: List[Dict[str, Any]] = field(default_factory=list)
    multi_layer: Optional[MultiLayerDecision] = None
    
    # Market hours (ET)
    market_open: time = field(default_factory=lambda: time(9, 30))
    market_close: time = field(default_factory=lambda: time(16, 0))
    pre_market_start: time = field(default_factory=lambda: time(4, 0))
    
    # Trailing stop config
    trailing_stop_pct: float = 0.8
    
    # Session results
    start_price: Optional[float] = None
    end_price: Optional[float] = None
    total_pnl: float = 0.0
    regime_start_ts: Optional[datetime] = None
    account_size_usd: float = 10_000.0
    risk_per_trade_pct: float = 1.0
    max_position_notional_pct: float = 100.0
    min_position_size: float = 1.0
    max_fill_participation_rate: float = 0.20
    min_fill_ratio: float = 0.35
    enable_partial_take_profit: bool = True
    partial_take_profit_rr: float = 1.0
    partial_take_profit_fraction: float = 0.5
    time_exit_bars: int = 40
    adverse_flow_exit_enabled: bool = True
    adverse_flow_threshold: float = 0.12
    adverse_flow_min_hold_bars: int = 3
    l2_confirm_enabled: bool = False
    l2_min_delta: float = 0.0
    l2_min_imbalance: float = 0.0
    l2_min_iceberg_bias: float = 0.0
    l2_lookback_bars: int = 3
    l2_min_participation_ratio: float = 0.0
    l2_min_directional_consistency: float = 0.0
    l2_min_signed_aggression: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'run_id': self.run_id,
            'ticker': self.ticker,
            'date': self.date,
            'phase': self.phase.value,
            'regime': self.detected_regime.value if self.detected_regime else None,
            'micro_regime': self.micro_regime,
            'selected_strategy': self.selected_strategy,
            'active_strategies': list(self.active_strategies),
            'bars_count': len(self.bars),
            'pre_market_bars_count': len(self.pre_market_bars),
            'active_position': self.active_position.to_dict() if self.active_position else None,
            'has_pending_signal': self.pending_signal is not None,
            'trades_count': len(self.trades),
            'signals_count': len(self.signals),
            'total_pnl': round(self.total_pnl, 2),
            'start_price': self.start_price,
            'end_price': self.end_price,
            'l2_confirm_enabled': self.l2_confirm_enabled,
            'l2_min_delta': self.l2_min_delta,
            'l2_min_imbalance': self.l2_min_imbalance,
            'l2_min_iceberg_bias': self.l2_min_iceberg_bias,
            'l2_lookback_bars': self.l2_lookback_bars,
            'l2_min_participation_ratio': self.l2_min_participation_ratio,
            'l2_min_directional_consistency': self.l2_min_directional_consistency,
            'l2_min_signed_aggression': self.l2_min_signed_aggression,
            'regime_refresh_bars': self.regime_refresh_bars,
            'regime_history': list(self.regime_history),
            'risk_per_trade_pct': self.risk_per_trade_pct,
            'max_position_notional_pct': self.max_position_notional_pct,
            'min_position_size': self.min_position_size,
            'max_fill_participation_rate': self.max_fill_participation_rate,
            'min_fill_ratio': self.min_fill_ratio,
            'enable_partial_take_profit': self.enable_partial_take_profit,
            'partial_take_profit_rr': self.partial_take_profit_rr,
            'partial_take_profit_fraction': self.partial_take_profit_fraction,
            'time_exit_bars': self.time_exit_bars,
            'adverse_flow_exit_enabled': self.adverse_flow_exit_enabled,
            'adverse_flow_threshold': self.adverse_flow_threshold,
            'adverse_flow_min_hold_bars': self.adverse_flow_min_hold_bars,
        }


class DayTradingManager:
    """
    Manages multiple trading sessions.
    Each session is identified by run_id + ticker + date.
    """
    
    def __init__(
        self,
        regime_detection_minutes: int = 30,
        trading_costs: TradingCosts = None,
        max_daily_loss: float = 300.0,
        max_trades_per_day: int = 3,
        trade_cooldown_bars: int = 15,  # Cooldown between trades in bars
        regime_refresh_bars: int = 12,
        risk_per_trade_pct: float = 1.0,
        max_position_notional_pct: float = 100.0,
        time_exit_bars: int = 40,
        enable_partial_take_profit: bool = True,
        partial_take_profit_rr: float = 1.0,
        partial_take_profit_fraction: float = 0.5,
        adverse_flow_exit_enabled: bool = True,
        adverse_flow_exit_threshold: float = 0.12,
        adverse_flow_min_hold_bars: int = 3,
    ):
        self.sessions: Dict[str, TradingSession] = {}  # session_key -> TradingSession
        self.regime_detection_minutes = regime_detection_minutes
        self.trading_costs = trading_costs or TradingCosts()
        self.max_daily_loss = max_daily_loss  # Stop trading if loss exceeds this amount
        self.max_trades_per_day = max_trades_per_day  # Limit trades to reduce fees
        self.trade_cooldown_bars = trade_cooldown_bars  # Cooldown between trades
        self.regime_refresh_bars = max(3, int(regime_refresh_bars))
        self.risk_per_trade_pct = max(0.1, float(risk_per_trade_pct))
        self.max_position_notional_pct = max(1.0, float(max_position_notional_pct))
        self.max_fill_participation_rate = 0.20
        self.min_fill_ratio = 0.35
        self.time_exit_bars = max(1, int(time_exit_bars))
        self.enable_partial_take_profit = bool(enable_partial_take_profit)
        self.partial_take_profit_rr = max(0.25, float(partial_take_profit_rr))
        self.partial_take_profit_fraction = min(0.95, max(0.05, float(partial_take_profit_fraction)))
        self.adverse_flow_exit_enabled = bool(adverse_flow_exit_enabled)
        self.adverse_flow_exit_threshold = max(0.02, float(adverse_flow_exit_threshold))
        self.adverse_flow_min_hold_bars = max(1, int(adverse_flow_min_hold_bars))
        self.market_tz = ZoneInfo("America/New_York")
        self.run_defaults: Dict[tuple, Dict[str, Any]] = {}
        
        self.last_trade_bar_index = {}  # Track last trade bar per session

        # Multi-layer decision engine
        self.multi_layer = MultiLayerDecision(
            pattern_weight=0.05,
            strategy_weight=0.95,
            threshold=65.0,
            require_pattern=False,
        )

        # Pre-configured strategies
        self.strategies = build_strategy_registry()
        
        # Strategy selection by regime
        # Default Preferences - including new volume-focused strategies
        self.default_preference = {
            Regime.TRENDING: ['momentum_flow', 'momentum', 'pullback', 'gap_liquidity', 'volume_profile', 'vwap_magnet'],
            Regime.CHOPPY: ['absorption_reversal', 'exhaustion_fade', 'mean_reversion', 'vwap_magnet', 'volume_profile'],
            Regime.MIXED: ['exhaustion_fade', 'absorption_reversal', 'volume_profile', 'gap_liquidity', 'mean_reversion', 'vwap_magnet', 'rotation']
        }

        self.micro_regime_preference = {
            "TRENDING_UP": ['momentum_flow', 'momentum', 'pullback', 'gap_liquidity'],
            "TRENDING_DOWN": ['momentum_flow', 'momentum', 'gap_liquidity', 'pullback'],
            "CHOPPY": ['absorption_reversal', 'exhaustion_fade', 'mean_reversion', 'vwap_magnet'],
            "ABSORPTION": ['absorption_reversal', 'exhaustion_fade', 'vwap_magnet'],
            "BREAKOUT": ['momentum_flow', 'momentum', 'gap_liquidity'],
            "MIXED": ['exhaustion_fade', 'volume_profile', 'rotation'],
        }
        
        # Ticker-Specific Preferences (AOS Optimized)
        self.ticker_preferences = {
            "NVDA": {
                Regime.TRENDING: ['pullback', 'momentum', 'volume_profile'],
                Regime.CHOPPY: [],  # Skip choppy - losses historically
                Regime.MIXED: ['volume_profile', 'vwap_magnet']
            },
            "TSLA": {
                Regime.TRENDING: ['momentum', 'gap_liquidity'],
                Regime.CHOPPY: [],  # Skip choppy - high volatility losses
                Regime.MIXED: []  # Only trade clear trends
            },
            "AAPL": {
                Regime.TRENDING: ['mean_reversion', 'vwap_magnet', 'pullback'],
                Regime.CHOPPY: ['mean_reversion', 'vwap_magnet'],
                Regime.MIXED: ['mean_reversion', 'vwap_magnet']
            },
            "AMD": {
                Regime.TRENDING: ['momentum', 'volume_profile', 'pullback'],
                Regime.CHOPPY: [],  # Skip choppy
                Regime.MIXED: ['volume_profile', 'vwap_magnet']
            },
            "GOOGL": {
                Regime.TRENDING: ['mean_reversion', 'vwap_magnet'],
                Regime.CHOPPY: ['mean_reversion', 'vwap_magnet'],
                Regime.MIXED: ['mean_reversion', 'vwap_magnet']
            },
            "META": {
                Regime.TRENDING: ['pullback', 'vwap_magnet', 'volume_profile'],
                Regime.CHOPPY: ['mean_reversion'],
                Regime.MIXED: ['pullback', 'vwap_magnet']
            },
            "MSFT": {
                Regime.TRENDING: ['mean_reversion', 'vwap_magnet'],
                Regime.CHOPPY: ['mean_reversion', 'vwap_magnet'],
                Regime.MIXED: ['mean_reversion', 'vwap_magnet']
            },
            "MU": {
                Regime.TRENDING: ['momentum', 'gap_liquidity', 'volume_profile'],
                Regime.CHOPPY: [],  # Skip choppy
                Regime.MIXED: ['volume_profile']
            },
            "AMZN": {
                Regime.TRENDING: ['pullback', 'mean_reversion', 'vwap_magnet'],
                Regime.CHOPPY: ['mean_reversion'],
                Regime.MIXED: ['pullback', 'mean_reversion']
            }
        }
        
        # AOS ticker-specific parameters (will be loaded from config)
        self.ticker_params = {}
        
        # Try to load AOS config
        self._load_aos_config()
    
    def _load_aos_config(self, config_path: str = None):
        """Load AOS configuration from file if available."""
        import os
        
        # Try default paths
        if config_path is None:
            possible_paths = [
                '/Users/hotovo/.gemini/antigravity/scratch/backtest-runner/aos_optimization/aos_config.json',
                'aos_optimization/aos_config.json',
                '../backtest-runner/aos_optimization/aos_config.json'
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)

                loaded_ticker_params: Dict[str, Dict[str, Any]] = {}

                # Load ticker-specific parameters (do not mutate global strategy
                # instances per ticker; apply overrides at signal generation time).
                for ticker, ticker_config in config.get('tickers', {}).items():
                    ticker_key = str(ticker).upper()
                    primary_strategy = self._canonical_strategy_key(ticker_config.get('strategy', ''))
                    backup_strategy = self._canonical_strategy_key(ticker_config.get('backup_strategy', ''))
                    loaded_ticker_params[ticker_key] = {
                        'strategy': primary_strategy,
                        'backup_strategy': backup_strategy,
                        'params': ticker_config.get('params', {}),
                        'regime_filter': ticker_config.get('regime_filter', []),
                        'avoid_days': ticker_config.get('avoid_days', []),
                        'trading_hours': ticker_config.get('trading_hours', None),  # Time-based filter
                        'long_only': ticker_config.get('long_only', False),
                        'time_filter_enabled': ticker_config.get('time_filter_enabled', True),
                        'min_confidence': ticker_config.get('min_confidence', 65.0),
                        'max_daily_trades': ticker_config.get('max_daily_trades', 2),
                        'multilayer': self._extract_ticker_multilayer_config(ticker_config),
                        'l2': ticker_config.get('l2', {}) if isinstance(ticker_config.get('l2'), dict) else {},
                    }

                self.ticker_params = loaded_ticker_params
                logger.info(f"Loaded AOS config from {config_path} ({len(self.ticker_params)} tickers)")
            except Exception as e:
                logger.warning(f"Could not load AOS config: {e}")
        
    def _to_market_time(self, ts: datetime) -> datetime:
        """Convert timestamp to US/Eastern for market-hour comparisons."""
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=ZoneInfo("UTC"))
        return ts.astimezone(self.market_tz)

    def _canonical_strategy_key(self, strategy_name: str) -> str:
        """Normalize strategy name to one of self.strategies keys."""
        if not strategy_name:
            return ""

        normalized = strategy_name.strip().replace("-", "_").replace(" ", "_")
        lowered = normalized.lower()
        if lowered in self.strategies:
            return lowered

        snake = re.sub(r'(?<!^)(?=[A-Z])', '_', normalized).lower()
        snake = re.sub(r'__+', '_', snake)
        if snake in self.strategies:
            return snake

        compact = re.sub(r'[^a-z0-9]', '', lowered)
        for key in self.strategies.keys():
            if key.replace('_', '') == compact:
                return key

        return snake

    @staticmethod
    def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off"}:
                return False
        return default

    def _extract_ticker_multilayer_config(self, ticker_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract ticker-specific multi-layer/candlestick settings."""
        allowed_core = {"pattern_weight", "strategy_weight", "threshold", "require_pattern"}
        allowed_detector = {
            "body_doji_pct",
            "wick_ratio_hammer",
            "engulfing_min_body_pct",
            "volume_confirm_ratio",
            "vwap_proximity_pct",
        }

        raw: Dict[str, Any] = {}
        for key in ("multilayer", "multi_layer"):
            block = ticker_config.get(key)
            if isinstance(block, dict):
                raw.update(block)

        candlestick = ticker_config.get("candlestick")
        if isinstance(candlestick, dict):
            nested_ml = candlestick.get("multilayer")
            if isinstance(nested_ml, dict):
                raw.update(nested_ml)
            detector = candlestick.get("detector")
            if isinstance(detector, dict):
                raw.update(detector)
            for key in allowed_core.union(allowed_detector):
                if key in candlestick:
                    raw[key] = candlestick[key]

        for key in allowed_core.union(allowed_detector):
            if key in ticker_config and key not in raw:
                raw[key] = ticker_config[key]

        nested_detector = raw.get("detector")
        if isinstance(nested_detector, dict):
            for key in allowed_detector:
                if key in nested_detector and key not in raw:
                    raw[key] = nested_detector[key]

        out: Dict[str, Any] = {}
        for key in ("pattern_weight", "strategy_weight", "threshold"):
            if key not in raw:
                continue
            value = self._safe_float(raw.get(key))
            if value is not None:
                out[key] = value

        if "require_pattern" in raw:
            out["require_pattern"] = self._safe_bool(raw.get("require_pattern"), default=False)

        for key in allowed_detector:
            if key not in raw:
                continue
            value = self._safe_float(raw.get(key))
            if value is not None:
                out[key] = value

        return out

    def _build_multilayer_for_ticker(self, ticker: str) -> MultiLayerDecision:
        """Build a per-session multi-layer engine using ticker overrides."""
        ticker_cfg = self.ticker_params.get(str(ticker).upper(), {})
        ml_overrides = ticker_cfg.get("multilayer", {}) if isinstance(ticker_cfg, dict) else {}
        l2_cfg = ticker_cfg.get("l2", {}) if isinstance(ticker_cfg, dict) else {}
        l2_confirm_enabled = bool(l2_cfg.get("confirm_enabled", False)) if isinstance(l2_cfg, dict) else False

        base_ml = self.multi_layer
        strategy_weight = float(ml_overrides.get("strategy_weight", base_ml.strategy_weight))
        if l2_confirm_enabled:
            # In flow mode, keep strategy confidence unattenuated by weight scaling.
            strategy_weight = 1.0
        ml = MultiLayerDecision(
            pattern_weight=float(ml_overrides.get("pattern_weight", base_ml.pattern_weight)),
            strategy_weight=strategy_weight,
            threshold=float(ml_overrides.get("threshold", base_ml.threshold)),
            require_pattern=self._safe_bool(
                ml_overrides.get("require_pattern", base_ml.require_pattern),
                default=base_ml.require_pattern,
            ),
        )

        detector = ml.pattern_detector
        base_detector = base_ml.pattern_detector
        detector.body_doji_pct = float(ml_overrides.get("body_doji_pct", base_detector.body_doji_pct))
        detector.wick_ratio_hammer = float(ml_overrides.get("wick_ratio_hammer", base_detector.wick_ratio_hammer))
        detector.engulfing_min_body_pct = float(
            ml_overrides.get("engulfing_min_body_pct", base_detector.engulfing_min_body_pct)
        )
        detector.volume_confirm_ratio = float(
            ml_overrides.get("volume_confirm_ratio", base_detector.volume_confirm_ratio)
        )
        detector.vwap_proximity_pct = float(
            ml_overrides.get("vwap_proximity_pct", base_detector.vwap_proximity_pct)
        )
        return ml

    def _is_day_allowed(self, date_str: str, ticker: str) -> bool:
        """Check if trading day is allowed by ticker-specific avoid_days."""
        ticker_cfg = self.ticker_params.get(str(ticker).upper(), {})
        avoid_days = {str(day).strip().lower() for day in ticker_cfg.get("avoid_days", [])}
        if not avoid_days:
            return True

        try:
            weekday = datetime.strptime(date_str, "%Y-%m-%d").strftime("%A").lower()
            return weekday not in avoid_days
        except Exception:
            return True

    def _get_ticker_strategy_overrides(self, ticker: str, strategy_key: str) -> Dict[str, Any]:
        """Get per-ticker parameter overrides for a strategy."""
        ticker_cfg = self.ticker_params.get(str(ticker).upper(), {})
        if ticker_cfg.get("strategy") == strategy_key:
            return ticker_cfg.get("params", {}) or {}
        return {}

    def _generate_signal_with_overrides(
        self,
        strategy: BaseStrategy,
        overrides: Dict[str, Any],
        current_price: float,
        ohlcv: Dict[str, List[float]],
        indicators: Dict[str, Any],
        regime: Regime,
        timestamp: datetime
    ) -> Optional[Signal]:
        """Generate a signal with temporary attribute overrides."""
        if not overrides:
            return strategy.generate_signal(
                current_price=current_price,
                ohlcv=ohlcv,
                indicators=indicators,
                regime=regime,
                timestamp=timestamp
            )

        original_values = {}
        for key, value in overrides.items():
            if hasattr(strategy, key):
                original_values[key] = getattr(strategy, key)
                setattr(strategy, key, value)

        try:
            return strategy.generate_signal(
                current_price=current_price,
                ohlcv=ohlcv,
                indicators=indicators,
                regime=regime,
                timestamp=timestamp
            )
        finally:
            for key, value in original_values.items():
                setattr(strategy, key, value)

    def _get_session_key(self, run_id: str, ticker: str, date: str) -> str:
        """Generate unique session key."""
        return f"{run_id}:{ticker}:{date}"
    
    def get_or_create_session(
        self, 
        run_id: str, 
        ticker: str, 
        date: str,
        regime_detection_minutes: int = None
    ) -> TradingSession:
        """Get existing session or create new one."""
        key = self._get_session_key(run_id, ticker, date)
        
        if key not in self.sessions:
            # Refresh AOS config for newly created sessions so dashboard updates
            # (including candlestick/multi-layer thresholds) are picked up.
            self._load_aos_config()

            self.sessions[key] = TradingSession(
                run_id=run_id,
                ticker=ticker,
                date=date,
                regime_detection_minutes=regime_detection_minutes or self.regime_detection_minutes
            )
            self.sessions[key].regime_refresh_bars = self.regime_refresh_bars
            self.sessions[key].risk_per_trade_pct = self.risk_per_trade_pct
            self.sessions[key].max_position_notional_pct = self.max_position_notional_pct
            self.sessions[key].time_exit_bars = self.time_exit_bars
            self.sessions[key].max_fill_participation_rate = self.max_fill_participation_rate
            self.sessions[key].min_fill_ratio = self.min_fill_ratio
            self.sessions[key].enable_partial_take_profit = self.enable_partial_take_profit
            self.sessions[key].partial_take_profit_rr = self.partial_take_profit_rr
            self.sessions[key].partial_take_profit_fraction = self.partial_take_profit_fraction
            self.sessions[key].adverse_flow_exit_enabled = self.adverse_flow_exit_enabled
            self.sessions[key].adverse_flow_threshold = self.adverse_flow_exit_threshold
            self.sessions[key].adverse_flow_min_hold_bars = self.adverse_flow_min_hold_bars
            self.sessions[key].multi_layer = self._build_multilayer_for_ticker(ticker)
            ticker_cfg = self.ticker_params.get(str(ticker).upper(), {})
            l2_cfg = ticker_cfg.get("l2", {}) if isinstance(ticker_cfg, dict) else {}
            if isinstance(l2_cfg, dict):
                self.sessions[key].l2_confirm_enabled = bool(l2_cfg.get("confirm_enabled", False))
                self.sessions[key].l2_min_delta = self._safe_float(
                    l2_cfg.get("min_delta"), 0.0
                ) or 0.0
                self.sessions[key].l2_min_imbalance = self._safe_float(
                    l2_cfg.get("min_imbalance"), 0.0
                ) or 0.0
                self.sessions[key].l2_min_iceberg_bias = self._safe_float(
                    l2_cfg.get("min_iceberg_bias"), 0.0
                ) or 0.0
                self.sessions[key].l2_lookback_bars = max(
                    1, int(self._safe_float(l2_cfg.get("lookback_bars"), 3) or 3)
                )
                self.sessions[key].l2_min_participation_ratio = self._safe_float(
                    l2_cfg.get("min_participation_ratio"), 0.0
                ) or 0.0
                self.sessions[key].l2_min_directional_consistency = self._safe_float(
                    l2_cfg.get("min_directional_consistency"), 0.0
                ) or 0.0
                self.sessions[key].l2_min_signed_aggression = self._safe_float(
                    l2_cfg.get("min_signed_aggression"), 0.0
                ) or 0.0

            # Optional config-driven day filter.
            if not self._is_day_allowed(date, ticker):
                self.sessions[key].phase = SessionPhase.CLOSED
        
        return self.sessions[key]

    def set_run_defaults(
        self,
        run_id: str,
        ticker: str,
        regime_detection_minutes: Optional[int] = None,
        regime_refresh_bars: Optional[int] = None,
        account_size_usd: Optional[float] = None,
        risk_per_trade_pct: Optional[float] = None,
        max_position_notional_pct: Optional[float] = None,
        max_fill_participation_rate: Optional[float] = None,
        min_fill_ratio: Optional[float] = None,
        enable_partial_take_profit: Optional[bool] = None,
        partial_take_profit_rr: Optional[float] = None,
        partial_take_profit_fraction: Optional[float] = None,
        time_exit_bars: Optional[int] = None,
        adverse_flow_exit_enabled: Optional[bool] = None,
        adverse_flow_threshold: Optional[float] = None,
        adverse_flow_min_hold_bars: Optional[int] = None,
        l2_confirm_enabled: Optional[bool] = None,
        l2_min_delta: Optional[float] = None,
        l2_min_imbalance: Optional[float] = None,
        l2_min_iceberg_bias: Optional[float] = None,
        l2_lookback_bars: Optional[int] = None,
        l2_min_participation_ratio: Optional[float] = None,
        l2_min_directional_consistency: Optional[float] = None,
        l2_min_signed_aggression: Optional[float] = None,
    ) -> None:
        """Set default parameters for all sessions in a run."""
        key = (run_id, ticker)
        defaults = self.run_defaults.get(key, {})
        if regime_detection_minutes is not None:
            defaults["regime_detection_minutes"] = regime_detection_minutes
        if regime_refresh_bars is not None:
            defaults["regime_refresh_bars"] = max(3, int(regime_refresh_bars))
        if account_size_usd is not None:
            defaults["account_size_usd"] = account_size_usd
        if risk_per_trade_pct is not None:
            defaults["risk_per_trade_pct"] = max(0.1, float(risk_per_trade_pct))
        if max_position_notional_pct is not None:
            defaults["max_position_notional_pct"] = max(1.0, float(max_position_notional_pct))
        if max_fill_participation_rate is not None:
            defaults["max_fill_participation_rate"] = min(1.0, max(0.01, float(max_fill_participation_rate)))
        if min_fill_ratio is not None:
            defaults["min_fill_ratio"] = min(1.0, max(0.01, float(min_fill_ratio)))
        if enable_partial_take_profit is not None:
            defaults["enable_partial_take_profit"] = bool(enable_partial_take_profit)
        if partial_take_profit_rr is not None:
            defaults["partial_take_profit_rr"] = max(0.25, float(partial_take_profit_rr))
        if partial_take_profit_fraction is not None:
            defaults["partial_take_profit_fraction"] = min(0.95, max(0.05, float(partial_take_profit_fraction)))
        if time_exit_bars is not None:
            defaults["time_exit_bars"] = max(1, int(time_exit_bars))
        if adverse_flow_exit_enabled is not None:
            defaults["adverse_flow_exit_enabled"] = bool(adverse_flow_exit_enabled)
        if adverse_flow_threshold is not None:
            defaults["adverse_flow_threshold"] = max(0.02, float(adverse_flow_threshold))
        if adverse_flow_min_hold_bars is not None:
            defaults["adverse_flow_min_hold_bars"] = max(1, int(adverse_flow_min_hold_bars))
        if l2_confirm_enabled is not None:
            defaults["l2_confirm_enabled"] = bool(l2_confirm_enabled)
        if l2_min_delta is not None:
            defaults["l2_min_delta"] = float(l2_min_delta)
        if l2_min_imbalance is not None:
            defaults["l2_min_imbalance"] = float(l2_min_imbalance)
        if l2_min_iceberg_bias is not None:
            defaults["l2_min_iceberg_bias"] = float(l2_min_iceberg_bias)
        if l2_lookback_bars is not None:
            defaults["l2_lookback_bars"] = max(1, int(l2_lookback_bars))
        if l2_min_participation_ratio is not None:
            defaults["l2_min_participation_ratio"] = float(l2_min_participation_ratio)
        if l2_min_directional_consistency is not None:
            defaults["l2_min_directional_consistency"] = float(l2_min_directional_consistency)
        if l2_min_signed_aggression is not None:
            defaults["l2_min_signed_aggression"] = float(l2_min_signed_aggression)
        self.run_defaults[key] = defaults
    
    def get_session(self, run_id: str, ticker: str, date: str) -> Optional[TradingSession]:
        """Get existing session."""
        key = self._get_session_key(run_id, ticker, date)
        return self.sessions.get(key)
    
    def process_bar(
        self,
        run_id: str,
        ticker: str,
        timestamp: datetime,
        bar_data: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Process a new bar for the session.
        
        Args:
            run_id: Unique run identifier
            ticker: Stock ticker
            timestamp: Bar timestamp
            bar_data: Dict with open, high, low, close, volume, vwap (optional)
        
        Returns:
            Processing result with signals, trades, etc.
        """
        market_ts = self._to_market_time(timestamp)
        date = market_ts.strftime('%Y-%m-%d')
        defaults = self.run_defaults.get((run_id, ticker), {})
        session = self.get_or_create_session(
            run_id,
            ticker,
            date,
            regime_detection_minutes=defaults.get("regime_detection_minutes")
        )
        if "account_size_usd" in defaults:
            session.account_size_usd = defaults["account_size_usd"]
        if "risk_per_trade_pct" in defaults:
            session.risk_per_trade_pct = float(defaults["risk_per_trade_pct"])
        if "max_position_notional_pct" in defaults:
            session.max_position_notional_pct = float(defaults["max_position_notional_pct"])
        if "max_fill_participation_rate" in defaults:
            session.max_fill_participation_rate = min(1.0, max(0.01, float(defaults["max_fill_participation_rate"])))
        if "min_fill_ratio" in defaults:
            session.min_fill_ratio = min(1.0, max(0.01, float(defaults["min_fill_ratio"])))
        if "enable_partial_take_profit" in defaults:
            session.enable_partial_take_profit = bool(defaults["enable_partial_take_profit"])
        if "partial_take_profit_rr" in defaults:
            session.partial_take_profit_rr = max(0.25, float(defaults["partial_take_profit_rr"]))
        if "partial_take_profit_fraction" in defaults:
            session.partial_take_profit_fraction = min(0.95, max(0.05, float(defaults["partial_take_profit_fraction"])))
        if "time_exit_bars" in defaults:
            session.time_exit_bars = max(1, int(defaults["time_exit_bars"]))
        if "adverse_flow_exit_enabled" in defaults:
            session.adverse_flow_exit_enabled = bool(defaults["adverse_flow_exit_enabled"])
        if "adverse_flow_threshold" in defaults:
            session.adverse_flow_threshold = max(0.02, float(defaults["adverse_flow_threshold"]))
        if "adverse_flow_min_hold_bars" in defaults:
            session.adverse_flow_min_hold_bars = max(1, int(defaults["adverse_flow_min_hold_bars"]))
        if "regime_refresh_bars" in defaults:
            session.regime_refresh_bars = max(3, int(defaults["regime_refresh_bars"]))
        if "l2_confirm_enabled" in defaults:
            session.l2_confirm_enabled = bool(defaults["l2_confirm_enabled"])
        if "l2_min_delta" in defaults:
            session.l2_min_delta = float(defaults["l2_min_delta"])
        if "l2_min_imbalance" in defaults:
            session.l2_min_imbalance = float(defaults["l2_min_imbalance"])
        if "l2_min_iceberg_bias" in defaults:
            session.l2_min_iceberg_bias = float(defaults["l2_min_iceberg_bias"])
        if "l2_lookback_bars" in defaults:
            session.l2_lookback_bars = max(1, int(defaults["l2_lookback_bars"]))
        if "l2_min_participation_ratio" in defaults:
            session.l2_min_participation_ratio = float(defaults["l2_min_participation_ratio"])
        if "l2_min_directional_consistency" in defaults:
            session.l2_min_directional_consistency = float(defaults["l2_min_directional_consistency"])
        if "l2_min_signed_aggression" in defaults:
            session.l2_min_signed_aggression = float(defaults["l2_min_signed_aggression"])
        if session.phase == SessionPhase.CLOSED:
             return {'action': 'skipped_finished_session'}
        
        # Debug
        # print(f"Processing bar {timestamp}, phase: {session.phase}")
        
        # Debug logging
        # print(f"DEBUG: Processing bar for {run_id}, key: {self._get_session_key(run_id, ticker, date)}")
        
        # Create bar
        bar = BarData(
            timestamp=timestamp,
            open=bar_data.get('open', 0),
            high=bar_data.get('high', 0),
            low=bar_data.get('low', 0),
            close=bar_data.get('close', 0),
            volume=bar_data.get('volume', 0),
            vwap=bar_data.get('vwap'),
            l2_delta=bar_data.get('l2_delta'),
            l2_buy_volume=bar_data.get('l2_buy_volume'),
            l2_sell_volume=bar_data.get('l2_sell_volume'),
            l2_volume=bar_data.get('l2_volume'),
            l2_imbalance=bar_data.get('l2_imbalance'),
            l2_bid_depth_total=bar_data.get('l2_bid_depth_total'),
            l2_ask_depth_total=bar_data.get('l2_ask_depth_total'),
            l2_book_pressure=bar_data.get('l2_book_pressure'),
            l2_book_pressure_change=bar_data.get('l2_book_pressure_change'),
            l2_iceberg_buy_count=bar_data.get('l2_iceberg_buy_count'),
            l2_iceberg_sell_count=bar_data.get('l2_iceberg_sell_count'),
            l2_iceberg_bias=bar_data.get('l2_iceberg_bias'),
        )
        
        bar_time = market_ts.time()
        result = {
            'session_key': self._get_session_key(run_id, ticker, date),
            'bar_timestamp': timestamp.isoformat(),
            'bar_price': bar.close,
            'phase': session.phase.value,
            'action': None,
            'signal': None,
            'trade_closed': None,
            'regime': session.detected_regime.value if session.detected_regime else None,
            'micro_regime': session.micro_regime,
            'strategy': session.selected_strategy
        }
        
        # Determine phase based on time
        if bar_time < session.market_open:
            # Pre-market
            session.pre_market_bars.append(bar)
            session.phase = SessionPhase.PRE_MARKET
            result['action'] = 'stored_pre_market_bar'
            
        elif session.phase == SessionPhase.PRE_MARKET:
            # First regular market bar - transition to regime detection
            session.phase = SessionPhase.REGIME_DETECTION
            session.bars.append(bar)
            session.start_price = bar.close
            session.regime_start_ts = datetime.combine(
                market_ts.date(),
                session.market_open,
                tzinfo=self.market_tz
            )
            result['action'] = 'started_regime_detection'
            
        elif session.phase == SessionPhase.REGIME_DETECTION:
            session.bars.append(bar)
            
            # Check if we have enough time for regime detection
            if session.regime_start_ts is None:
                session.regime_start_ts = datetime.combine(
                    market_ts.date(),
                    session.market_open,
                    tzinfo=self.market_tz
                )
            elapsed_minutes = (market_ts - session.regime_start_ts).total_seconds() / 60.0
            effective_regime_minutes = session.regime_detection_minutes
            if self._bar_has_l2_data(bar):
                effective_regime_minutes = min(effective_regime_minutes, 10)
            elif len(session.bars) >= 3:
                flow_probe = self._calculate_order_flow_metrics(
                    session.bars,
                    lookback=min(10, len(session.bars)),
                )
                if flow_probe.get("has_l2_coverage", False):
                    effective_regime_minutes = min(effective_regime_minutes, 10)
            
            if elapsed_minutes >= effective_regime_minutes:
                # Detect regime
                regime = self._detect_regime(session)
                session.detected_regime = regime
                
                # Select BEST strategies for this regime
                active_strategies = self._select_strategies(session)
                session.active_strategies = active_strategies
                session.selected_strategy = (
                    "adaptive" if len(active_strategies) > 1 else (active_strategies[0] if active_strategies else None)
                )
                session.last_regime_refresh_bar_index = max(0, len(session.bars) - 1)
                session.regime_history.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "bar_index": session.last_regime_refresh_bar_index,
                        "regime": regime.value,
                        "micro_regime": session.micro_regime,
                        "strategies": list(active_strategies),
                    }
                )
                
                # Transition to trading
                session.phase = SessionPhase.TRADING
                result['action'] = 'regime_detected'
                result['regime'] = regime.value
                result['micro_regime'] = session.micro_regime
                result['strategies'] = active_strategies
                result['strategy'] = session.selected_strategy
                # print(f"DEBUG: Regime DETECTED {regime} at {timestamp}. Strategy: {active_strategies[0]}", flush=True)
                
                # Include indicator values for frontend decision markers
                indicators = self._calculate_indicators(session.bars)
                flow = indicators.get('order_flow') or {}
                result['indicators'] = {
                    'trend_efficiency': self._calc_trend_efficiency(session.bars),
                    'volatility': self._calc_volatility(session.bars),
                    'atr': indicators.get('atr', [0])[-1] if indicators.get('atr') else 0,
                    'adx': indicators.get('adx', [0])[-1] if indicators.get('adx') else 0,
                    'flow_score': float(flow.get('flow_score', 0.0) or 0.0),
                    'signed_aggression': float(flow.get('signed_aggression', 0.0) or 0.0),
                    'absorption_rate': float(flow.get('absorption_rate', 0.0) or 0.0),
                    'book_pressure_avg': float(flow.get('book_pressure_avg', 0.0) or 0.0),
                    'book_pressure_trend': float(flow.get('book_pressure_trend', 0.0) or 0.0),
                    'large_trader_activity': float(flow.get('large_trader_activity', 0.0) or 0.0),
                    'vwap_execution_flow': float(flow.get('vwap_execution_flow', 0.0) or 0.0),
                }
            else:
                result['action'] = 'collecting_regime_data'
                minutes_elapsed = int(elapsed_minutes)
                result['minutes_remaining'] = max(
                    0,
                    effective_regime_minutes - minutes_elapsed
                )
                
        elif session.phase == SessionPhase.TRADING:
            # print(f"DEBUG: TRADING Phase. Bar: {timestamp}", flush=True)
            session.bars.append(bar)
            
            # Check for end of day
            if bar_time >= session.market_close or bar_time >= time(15, 55):
                # Close any open position
                if session.active_position:
                    trade = self._close_position(
                        session,
                        bar.close,
                        timestamp,
                        'end_of_day',
                        bar_volume=bar.volume,
                    )
                    result['trade_closed'] = trade.to_dict()
                
                session.phase = SessionPhase.END_OF_DAY
                session.end_price = bar.close
                result['action'] = 'session_ended'
                result['session_summary'] = self._get_session_summary(session)
            else:
                # Normal trading - manage position and look for signals
                trade_result = self._process_trading_bar(session, bar, timestamp)
                result.update(trade_result)
                
        elif session.phase == SessionPhase.END_OF_DAY:
            session.end_price = bar.close
            result['action'] = 'session_already_closed'
            
        result['phase'] = session.phase.value
        result['micro_regime'] = session.micro_regime
        return result
    
    def _detect_regime(self, session: TradingSession) -> Regime:
        """Detect macro regime and update session micro-regime."""
        all_bars = session.bars
        if len(all_bars) < 20:
            session.micro_regime = "MIXED"
            return Regime.MIXED

        closes = [b.close for b in all_bars[-min(len(all_bars), 60):]]
        if len(closes) < 10:
            session.micro_regime = "MIXED"
            return Regime.MIXED

        net_move = abs(closes[-1] - closes[0])
        total_move = sum(abs(closes[i] - closes[i - 1]) for i in range(1, len(closes)))
        if total_move == 0:
            session.micro_regime = "MIXED"
            return Regime.MIXED
        trend_efficiency = net_move / total_move

        adx = self._calc_adx(all_bars)
        returns = [
            (closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(1, len(closes))
            if closes[i - 1] != 0
        ]
        avg_return = (sum(returns) / len(returns)) if returns else 0.0
        variance = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) if returns else 0.0
        volatility = variance ** 0.5
        price_bias = closes[-1] - closes[0]

        # Preserve prior gap context from pre-market.
        gap_factor = 0.0
        if session.pre_market_bars and session.bars:
            pre_close = session.pre_market_bars[-1].close
            open_price = session.bars[0].open
            if pre_close > 0:
                gap_pct = abs(open_price - pre_close) / pre_close * 100.0
                if gap_pct > 1.0:
                    gap_factor = 0.1

        flow = self._calculate_order_flow_metrics(all_bars, lookback=min(24, len(all_bars)))
        micro_regime = self._classify_micro_regime(
            trend_efficiency=trend_efficiency + gap_factor,
            adx=adx,
            volatility=volatility,
            price_bias=price_bias,
            flow=flow,
        )
        session.micro_regime = micro_regime
        return self._map_micro_to_regime(micro_regime)

    @staticmethod
    def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
        if denominator == 0:
            return default
        return numerator / denominator

    def _map_micro_to_regime(self, micro_regime: str) -> Regime:
        if micro_regime in {"TRENDING_UP", "TRENDING_DOWN", "BREAKOUT"}:
            return Regime.TRENDING
        if micro_regime in {"CHOPPY", "ABSORPTION"}:
            return Regime.CHOPPY
        return Regime.MIXED

    def _classify_micro_regime(
        self,
        trend_efficiency: float,
        adx: float,
        volatility: float,
        price_bias: float,
        flow: Dict[str, float],
    ) -> str:
        """Map price + order-flow behavior to a richer intraday micro-regime."""
        signed_aggr = float(flow.get("signed_aggression", 0.0))
        consistency = float(flow.get("directional_consistency", 0.0))
        sweep_intensity = float(flow.get("sweep_intensity", 0.0))
        absorption_rate = float(flow.get("absorption_rate", 0.0))
        book_pressure_avg = float(flow.get("book_pressure_avg", 0.0))
        large_trader_activity = float(flow.get("large_trader_activity", 0.0))
        vwap_execution_flow = float(flow.get("vwap_execution_flow", 0.0))
        price_change_pct = float(flow.get("price_change_pct", 0.0))
        flow_has_coverage = bool(flow.get("has_l2_coverage", False))

        if flow_has_coverage:
            trend_eff_threshold = 0.45 if adx > 35.0 else 0.58

            # Large one-way sessions should not be blocked by efficiency checks.
            if abs(price_change_pct) > 2.0:
                return "TRENDING_UP" if price_change_pct >= 0 else "TRENDING_DOWN"

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
                and abs(price_change_pct) <= max(0.12, volatility * 100.0 * 0.8)
            ):
                return "ABSORPTION"

            if trend_efficiency >= trend_eff_threshold and adx >= 25.0 and signed_aggr >= 0.08 and price_bias > 0:
                return "TRENDING_UP"
            if trend_efficiency >= trend_eff_threshold and adx >= 25.0 and signed_aggr <= -0.08 and price_bias < 0:
                return "TRENDING_DOWN"

            if adx < 20.0 and trend_efficiency < 0.40:
                return "CHOPPY"
            return "MIXED"

        # Fallback without L2 flow coverage.
        if trend_efficiency >= 0.62 or adx > 30.0:
            return "TRENDING_UP" if price_bias >= 0 else "TRENDING_DOWN"
        if adx < 20.0:
            return "CHOPPY"
        return "MIXED"

    def _calculate_order_flow_metrics(
        self,
        bars: List[BarData],
        lookback: int = 20,
    ) -> Dict[str, float]:
        """Calculate no-lookahead microstructure metrics from existing L2-enriched bars."""
        window_size = max(5, int(lookback))
        window = bars[-window_size:] if bars else []

        if not window:
            return {
                "has_l2_coverage": False,
                "bars_with_l2": 0.0,
                "lookback_bars": float(window_size),
            }

        bars_with_l2 = 0
        deltas: List[float] = []
        imbalances: List[float] = []
        iceberg_biases: List[float] = []
        l2_volumes: List[float] = []
        book_pressures: List[float] = []
        book_pressure_changes: List[float] = []
        bid_depth_totals: List[float] = []
        ask_depth_totals: List[float] = []
        bar_volumes: List[float] = []
        close_to_close_pct: List[float] = []
        vwap_cluster_l2_volume = 0.0
        for i, b in enumerate(window):
            has_l2 = (
                b.l2_delta is not None
                or b.l2_imbalance is not None
                or b.l2_volume is not None
                or b.l2_book_pressure is not None
                or b.l2_bid_depth_total is not None
                or b.l2_ask_depth_total is not None
                or b.l2_iceberg_bias is not None
            )
            if has_l2:
                bars_with_l2 += 1

            deltas.append(self._to_float(b.l2_delta, 0.0))
            imbalances.append(self._to_float(b.l2_imbalance, 0.0))
            iceberg_biases.append(self._to_float(b.l2_iceberg_bias, 0.0))
            l2_volumes.append(max(0.0, self._to_float(b.l2_volume, 0.0)))
            book_pressures.append(self._to_float(b.l2_book_pressure, 0.0))
            book_pressure_changes.append(self._to_float(b.l2_book_pressure_change, 0.0))
            bid_depth_totals.append(max(0.0, self._to_float(b.l2_bid_depth_total, 0.0)))
            ask_depth_totals.append(max(0.0, self._to_float(b.l2_ask_depth_total, 0.0)))
            bar_volumes.append(max(0.0, self._to_float(b.volume, 0.0)))
            if b.vwap is not None and b.close > 0:
                vwap_distance_pct = abs((b.close - b.vwap) / b.close) * 100.0
                if vwap_distance_pct <= 0.05:
                    vwap_cluster_l2_volume += l2_volumes[-1]

            if i > 0 and window[i - 1].close > 0:
                close_to_close_pct.append((b.close - window[i - 1].close) / window[i - 1].close * 100.0)

        has_l2_coverage = bars_with_l2 >= max(3, len(window) // 2)
        cumulative_delta = float(sum(deltas))
        avg_imbalance = float(sum(imbalances) / len(imbalances)) if imbalances else 0.0
        iceberg_bias_sum = float(sum(iceberg_biases))
        total_l2_volume = float(sum(l2_volumes))
        total_bar_volume = float(sum(bar_volumes))
        participation_ratio = self._safe_div(total_l2_volume, total_bar_volume, 0.0)
        signed_aggression = self._safe_div(cumulative_delta, total_l2_volume, 0.0)
        avg_bid_depth_total = float(sum(bid_depth_totals) / len(bid_depth_totals)) if bid_depth_totals else 0.0
        avg_ask_depth_total = float(sum(ask_depth_totals) / len(ask_depth_totals)) if ask_depth_totals else 0.0
        book_pressure_avg = float(sum(book_pressures) / len(book_pressures)) if book_pressures else 0.0
        book_pressure_change_avg = (
            float(sum(book_pressure_changes) / len(book_pressure_changes))
            if book_pressure_changes else 0.0
        )
        if len(book_pressures) >= 2:
            book_pressure_trend = float(book_pressures[-1] - book_pressures[0])
        else:
            book_pressure_trend = float(book_pressure_change_avg)

        # Delta acceleration compares current window against immediate previous window.
        prev_window = bars[-(window_size * 2): -window_size] if len(bars) > window_size else []
        prev_delta = float(sum(self._to_float(b.l2_delta, 0.0) for b in prev_window)) if prev_window else 0.0
        delta_acceleration = cumulative_delta - prev_delta

        first_close = window[0].close
        last_close = window[-1].close
        price_change_pct = self._safe_div((last_close - first_close) * 100.0, first_close, 0.0)

        # Directional consistency: delta sign agrees with bar-to-bar close change sign.
        directional_base = 0
        directional_hits = 0
        low_progress_l2_volume = 0.0
        for i in range(1, len(window)):
            delta_val = deltas[i]
            prev_close = window[i - 1].close
            if prev_close <= 0:
                continue
            price_change = (window[i].close - prev_close) / prev_close * 100.0
            if abs(delta_val) > 1e-9 and abs(price_change) > 1e-6:
                directional_base += 1
                if (delta_val * price_change) > 0:
                    directional_hits += 1

            # Absorption proxy: large flow while price barely moves.
            if abs(price_change) <= 0.02:
                low_progress_l2_volume += l2_volumes[i]

        directional_consistency = self._safe_div(float(directional_hits), float(directional_base), 0.0)
        absorption_rate = self._safe_div(low_progress_l2_volume, total_l2_volume, 0.0)

        # Sweep proxy: bursty delta and elevated L2 volume.
        abs_deltas = [abs(v) for v in deltas]
        mean_abs_delta = sum(abs_deltas) / len(abs_deltas) if abs_deltas else 0.0
        delta_variance = (
            sum((ad - mean_abs_delta) ** 2 for ad in abs_deltas) / len(abs_deltas)
            if abs_deltas else 0.0
        )
        delta_std = math.sqrt(delta_variance) if delta_variance > 0 else 0.0
        avg_l2_volume = sum(l2_volumes) / len(l2_volumes) if l2_volumes else 0.0
        sweep_hits = 0
        large_trade_hits = 0
        for d, lv in zip(abs_deltas, l2_volumes):
            if d >= (mean_abs_delta + delta_std) and lv >= (avg_l2_volume * 1.2):
                sweep_hits += 1
            if lv >= max(avg_l2_volume * 1.8, 5_000.0):
                large_trade_hits += 1
        sweep_intensity = self._safe_div(float(sweep_hits), float(len(window)), 0.0)
        large_trader_activity = self._safe_div(float(large_trade_hits), float(len(window)), 0.0)
        vwap_execution_flow = self._safe_div(vwap_cluster_l2_volume, total_l2_volume, 0.0)

        # Last-delta z-score for exhaustion checks.
        delta_mean = sum(deltas) / len(deltas) if deltas else 0.0
        delta_var = (
            sum((d - delta_mean) ** 2 for d in deltas) / len(deltas)
            if deltas else 0.0
        )
        delta_sigma = math.sqrt(delta_var) if delta_var > 0 else 0.0
        last_delta = deltas[-1] if deltas else 0.0
        delta_zscore = self._safe_div(last_delta - delta_mean, delta_sigma, 0.0)

        realized_volatility_pct = 0.0
        if close_to_close_pct:
            mean_ret = sum(close_to_close_pct) / len(close_to_close_pct)
            var_ret = sum((r - mean_ret) ** 2 for r in close_to_close_pct) / len(close_to_close_pct)
            realized_volatility_pct = math.sqrt(var_ret)
        vol_floor = max(realized_volatility_pct, 0.05)
        normalized_price = price_change_pct / vol_floor
        delta_price_divergence = signed_aggression - normalized_price

        flow_score = 100.0 * (
            0.22 * self._safe_div(abs(cumulative_delta), abs(cumulative_delta) + 5000.0, 0.0)
            + 0.19 * max(0.0, min(1.0, directional_consistency))
            + 0.15 * max(0.0, min(1.0, abs(avg_imbalance)))
            + 0.11 * max(0.0, min(1.0, sweep_intensity))
            + 0.10 * max(0.0, min(1.0, participation_ratio))
            + 0.08 * max(0.0, min(1.0, large_trader_activity))
            + 0.07 * max(0.0, min(1.0, vwap_execution_flow))
            + 0.08 * max(0.0, min(1.0, abs(book_pressure_avg)))
        )

        return {
            "has_l2_coverage": bool(has_l2_coverage),
            "bars_with_l2": float(bars_with_l2),
            "lookback_bars": float(len(window)),
            "cumulative_delta": cumulative_delta,
            "delta_acceleration": delta_acceleration,
            "delta_zscore": delta_zscore,
            "imbalance_avg": avg_imbalance,
            "iceberg_bias": iceberg_bias_sum,
            "participation_ratio": participation_ratio,
            "directional_consistency": directional_consistency,
            "signed_aggression": signed_aggression,
            "absorption_rate": absorption_rate,
            "book_pressure_avg": book_pressure_avg,
            "book_pressure_trend": book_pressure_trend,
            "book_pressure_change_avg": book_pressure_change_avg,
            "bid_depth_total_avg": avg_bid_depth_total,
            "ask_depth_total_avg": avg_ask_depth_total,
            "sweep_intensity": sweep_intensity,
            "large_trader_activity": large_trader_activity,
            "vwap_execution_flow": vwap_execution_flow,
            "price_change_pct": price_change_pct,
            "realized_volatility_pct": realized_volatility_pct,
            "delta_price_divergence": delta_price_divergence,
            "flow_score": flow_score,
        }
    
    def _select_strategies(self, session: TradingSession) -> List[str]:
        """Select active strategies for the detected regime and ticker."""
        regime = session.detected_regime or Regime.MIXED
        micro_regime = (session.micro_regime or "MIXED").upper()
        ticker = session.ticker.upper()
        ticker_cfg = self.ticker_params.get(ticker.upper(), {})

        # Optional AOS regime filter: if the regime is explicitly disallowed, skip trading.
        allowed_regimes_cfg = {
            str(r).strip().upper()
            for r in ticker_cfg.get("regime_filter", [])
            if str(r).strip()
        }
        if allowed_regimes_cfg and regime.value not in allowed_regimes_cfg:
            return []

        # Build ordered candidates:
        # 1) AOS primary/backup strategy
        # 2) Ticker defaults for this regime
        # 3) Global defaults for this regime
        candidates: List[str] = []
        primary = ticker_cfg.get("strategy")
        backup = ticker_cfg.get("backup_strategy")
        if primary:
            candidates.append(primary)
        if backup:
            candidates.append(backup)

        # Flow-first micro-regime preferences.
        candidates.extend(self.micro_regime_preference.get(micro_regime, []))

        ticker_prefs = self.ticker_preferences.get(ticker, {})
        candidates.extend(ticker_prefs.get(regime, []))
        candidates.extend(self.default_preference.get(regime, []))

        # If L2 coverage is present on current bars, bias toward flow strategies.
        flow_metrics = self._calculate_order_flow_metrics(session.bars, lookback=min(20, len(session.bars)))
        if flow_metrics.get("has_l2_coverage", False):
            candidates = ['momentum_flow', 'absorption_reversal', 'exhaustion_fade'] + candidates

        filtered: List[str] = []
        seen = set()
        for raw_name in candidates:
            name = self._canonical_strategy_key(raw_name)
            if not name or name in seen:
                continue
            strat = self.strategies.get(name)
            if not strat:
                continue
            if not getattr(strat, "enabled", True):
                continue
            if hasattr(strat, "allowed_regimes") and regime not in strat.allowed_regimes:
                continue
            seen.add(name)
            filtered.append(name)

        # Fallback: any enabled strategy that supports this regime.
        if not filtered:
            for name, strat in self.strategies.items():
                if not getattr(strat, "enabled", True):
                    continue
                if regime not in getattr(strat, "allowed_regimes", [regime]):
                    continue
                filtered.append(name)

        return filtered[:3]
    
    def _calc_trend_efficiency(self, bars: List[BarData]) -> float:
        """Calculate trend efficiency (net move / total move)."""
        if len(bars) < 5:
            return 0.0
        
        closes = [b.close for b in bars[-min(len(bars), 30):]]
        net_move = abs(closes[-1] - closes[0])
        total_move = sum(abs(closes[i] - closes[i-1]) for i in range(1, len(closes)))
        
        if total_move == 0:
            return 0.0
        
        return round(net_move / total_move, 3)
    
    def _calc_volatility(self, bars: List[BarData]) -> float:
        """Calculate volatility as standard deviation of returns in percent."""
        if len(bars) < 5:
            return 0.0
        
        closes = [b.close for b in bars[-min(len(bars), 30):]]
        returns = [(closes[i] - closes[i-1]) / closes[i-1] * 100 for i in range(1, len(closes))]
        
        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        
        return round(variance ** 0.5, 3)

    def _calc_adx(self, bars: List[BarData], period: int = 14) -> float:
        """Calculate ADX from bar data."""
        if len(bars) < period * 2:
            return 0.0
            
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        closes = [b.close for b in bars]
        
        tr_list = []
        dm_plus_list = []
        dm_minus_list = []
        
        for i in range(1, len(bars)):
            # True Range
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr = max(hl, hc, lc)
            tr_list.append(tr)
            
            # Directional Movement
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            
            if up_move > down_move and up_move > 0:
                dm_plus_list.append(up_move)
            else:
                dm_plus_list.append(0.0)
                
            if down_move > up_move and down_move > 0:
                dm_minus_list.append(down_move)
            else:
                dm_minus_list.append(0.0)
        
        # Need enough data for Wilder's smoothing
        if len(tr_list) < period:
            return 0.0
            
        # Initial smoothed averages (using simple average for first period)
        tr_smooth = sum(tr_list[:period])
        dm_plus_smooth = sum(dm_plus_list[:period])
        dm_minus_smooth = sum(dm_minus_list[:period])
        
        adx_values = []
        
        # Calculate smoothed values for the rest
        for i in range(period, len(tr_list)):
            tr_smooth = tr_smooth - (tr_smooth / period) + tr_list[i]
            dm_plus_smooth = dm_plus_smooth - (dm_plus_smooth / period) + dm_plus_list[i]
            dm_minus_smooth = dm_minus_smooth - (dm_minus_smooth / period) + dm_minus_list[i]
            
            if tr_smooth == 0:
                di_plus = 0
                di_minus = 0
            else:
                di_plus = 100 * (dm_plus_smooth / tr_smooth)
                di_minus = 100 * (dm_minus_smooth / tr_smooth)
            
            dx = 0
            if di_plus + di_minus > 0:
                dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
            
            # ADX Smoothing
            if len(adx_values) == 0:
                adx_values.append(dx) # Initial ADX is just DX
            else:
                # Wilder's smoothing for ADX
                prev_adx = adx_values[-1]
                adx = (prev_adx * (period - 1) + dx) / period
                adx_values.append(adx)
                
        return round(adx_values[-1], 2) if adx_values else 0.0
    
    def _calc_adx_series(self, bars: List[BarData], period: int = 14) -> List[float]:
        """Calculate ADX series incrementally (point-in-time, no look-ahead bias).
        
        Returns a list of ADX values, one for each bar, where each value
        only uses data up to that point in time.
        """
        if len(bars) < 2:
            return [0.0] * len(bars)
            
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        closes = [b.close for b in bars]
        
        # Pre-fill with zeros for bars that don't have enough data
        adx_series = [0.0] * len(bars)
        
        tr_list = []
        dm_plus_list = []
        dm_minus_list = []
        
        # First bar has no ADX
        for i in range(1, len(bars)):
            # True Range
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr = max(hl, hc, lc)
            tr_list.append(tr)
            
            # Directional Movement
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            
            if up_move > down_move and up_move > 0:
                dm_plus_list.append(up_move)
            else:
                dm_plus_list.append(0.0)
                
            if down_move > up_move and down_move > 0:
                dm_minus_list.append(down_move)
            else:
                dm_minus_list.append(0.0)
            
            # Need at least period bars of TR data
            if len(tr_list) < period:
                continue
            
            # Calculate ADX at this point using data up to bar i
            if len(tr_list) == period:
                # Initial smoothed averages
                tr_smooth = sum(tr_list[:period])
                dm_plus_smooth = sum(dm_plus_list[:period])
                dm_minus_smooth = sum(dm_minus_list[:period])
                
                if tr_smooth > 0:
                    di_plus = 100 * (dm_plus_smooth / tr_smooth)
                    di_minus = 100 * (dm_minus_smooth / tr_smooth)
                else:
                    di_plus = 0.0
                    di_minus = 0.0
                
                if di_plus + di_minus > 0:
                    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
                else:
                    dx = 0.0
                
                adx_series[i] = round(dx, 2)
            else:
                # Wilder's smoothing
                j = len(tr_list) - 1  # Current index in tr_list
                start_idx = j - period
                
                # Calculate smoothed values up to this point
                tr_smooth = sum(tr_list[start_idx:start_idx + period])
                dm_plus_smooth = sum(dm_plus_list[start_idx:start_idx + period])
                dm_minus_smooth = sum(dm_minus_list[start_idx:start_idx + period])
                
                # Apply Wilder's smoothing for remaining bars
                for k in range(start_idx + period, j + 1):
                    tr_smooth = tr_smooth - (tr_smooth / period) + tr_list[k]
                    dm_plus_smooth = dm_plus_smooth - (dm_plus_smooth / period) + dm_plus_list[k]
                    dm_minus_smooth = dm_minus_smooth - (dm_minus_smooth / period) + dm_minus_list[k]
                
                if tr_smooth > 0:
                    di_plus = 100 * (dm_plus_smooth / tr_smooth)
                    di_minus = 100 * (dm_minus_smooth / tr_smooth)
                else:
                    di_plus = 0.0
                    di_minus = 0.0
                
                if di_plus + di_minus > 0:
                    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
                else:
                    dx = 0.0
                
                # Use previous ADX for smoothing
                prev_adx = adx_series[i - 1]
                if prev_adx > 0:
                    adx = (prev_adx * (period - 1) + dx) / period
                else:
                    adx = dx
                
                adx_series[i] = round(adx, 2)
        
        return adx_series

    def _maybe_refresh_regime(
        self,
        session: TradingSession,
        current_bar_index: int,
        timestamp: datetime,
    ) -> Optional[Dict[str, Any]]:
        """
        Re-evaluate regime every N bars during trading.
        Returns update payload only when macro/micro regime or strategy set changed.
        """
        if current_bar_index < 0:
            return None
        if session.last_regime_refresh_bar_index >= 0:
            bars_since_refresh = current_bar_index - session.last_regime_refresh_bar_index
            if bars_since_refresh < max(3, int(session.regime_refresh_bars)):
                return None
        if len(session.bars) < 20:
            return None

        prev_macro = session.detected_regime or Regime.MIXED
        prev_micro = session.micro_regime
        prev_active = list(session.active_strategies)

        new_macro = self._detect_regime(session)
        session.detected_regime = new_macro
        new_active = self._select_strategies(session)
        session.active_strategies = new_active
        session.selected_strategy = (
            "adaptive" if len(new_active) > 1 else (new_active[0] if new_active else None)
        )
        session.last_regime_refresh_bar_index = current_bar_index

        changed = (
            new_macro != prev_macro
            or session.micro_regime != prev_micro
            or new_active != prev_active
        )
        if not changed:
            return None

        indicators = self._calculate_indicators(session.bars[-80:] if len(session.bars) >= 80 else session.bars)
        flow = indicators.get("order_flow") or {}
        payload = {
            "timestamp": timestamp.isoformat(),
            "bar_index": current_bar_index,
            "regime": new_macro.value,
            "micro_regime": session.micro_regime,
            "strategies": list(new_active),
            "strategy": session.selected_strategy,
            "previous_regime": prev_macro.value,
            "previous_micro_regime": prev_micro,
            "indicators": {
                "trend_efficiency": self._calc_trend_efficiency(session.bars),
                "volatility": self._calc_volatility(session.bars),
                "atr": indicators.get("atr", [0])[-1] if indicators.get("atr") else 0,
                "adx": indicators.get("adx", [0])[-1] if indicators.get("adx") else 0,
                "flow_score": float(flow.get("flow_score", 0.0) or 0.0),
                "signed_aggression": float(flow.get("signed_aggression", 0.0) or 0.0),
                "absorption_rate": float(flow.get("absorption_rate", 0.0) or 0.0),
                "book_pressure_avg": float(flow.get("book_pressure_avg", 0.0) or 0.0),
                "book_pressure_trend": float(flow.get("book_pressure_trend", 0.0) or 0.0),
                "large_trader_activity": float(flow.get("large_trader_activity", 0.0) or 0.0),
                "vwap_execution_flow": float(flow.get("vwap_execution_flow", 0.0) or 0.0),
            },
        }
        session.regime_history.append(payload)
        return payload

    def _strategy_edge_adjustment(self, session: TradingSession, strategy_key: str) -> float:
        """Estimate strategy edge from already realized session trades."""
        relevant = [
            t for t in session.trades
            if self._canonical_strategy_key(getattr(t, "strategy", "")) == strategy_key
        ]
        if not relevant:
            return 0.0

        wins = [t for t in relevant if t.pnl_pct > 0]
        losses = [t for t in relevant if t.pnl_pct <= 0]
        n = len(relevant)
        # Smoothed win rate to avoid unstable first trades.
        win_rate = (len(wins) + 1.0) / (n + 2.0)
        avg_win = (sum(t.pnl_pct for t in wins) / len(wins)) if wins else 0.0
        avg_loss = (abs(sum(t.pnl_pct for t in losses)) / len(losses)) if losses else 0.0
        expectancy = (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss)
        pf = (sum(t.pnl_pct for t in wins) / abs(sum(t.pnl_pct for t in losses))) if losses else (2.0 if wins else 0.0)

        edge = ((win_rate - 0.5) * 36.0) + ((pf - 1.0) * 6.0) + (expectancy * 4.0)
        return max(-12.0, min(12.0, edge))
    
    def _process_trading_bar(
        self, 
        session: TradingSession, 
        bar: BarData, 
        timestamp: datetime
    ) -> Dict[str, Any]:
        """Process a trading bar - manage positions and generate signals."""
        result = {'action': 'trading', 'micro_regime': session.micro_regime}
        current_price = bar.close
        current_bar_index = max(0, len(session.bars) - 1)
        session_key = self._get_session_key(session.run_id, session.ticker, session.date)

        # Execute previous-bar signal at current bar open (no same-bar signal fill).
        if session.pending_signal and not session.active_position:
            signal = session.pending_signal
            position = self._open_position(
                session,
                signal,
                entry_price=bar.open,
                entry_time=timestamp,
                signal_bar_index=session.pending_signal_bar_index,
                entry_bar_index=current_bar_index,
                entry_bar_volume=bar.volume,
            )
            session.pending_signal = None
            session.pending_signal_bar_index = -1
            if position.size > 0:
                self.last_trade_bar_index[session_key] = current_bar_index
                result['action'] = 'position_opened'
                result['position'] = position.to_dict()
                result['position_opened'] = {
                    'entry_price': position.entry_price,
                    'side': position.side,
                    'strategy': position.strategy_name,
                    'size': position.size,
                    'fill_ratio': position.fill_ratio,
                    'stop_loss': position.stop_loss,
                    'take_profit': position.take_profit,
                    'partial_take_profit_price': position.partial_take_profit_price,
                    'reasoning': signal.reasoning,
                    'confidence': signal.confidence,
                    'metadata': signal.metadata
                }
            else:
                session.active_position = None
                result['action'] = 'insufficient_fill'
                result['reason'] = 'Position size after risk/fill constraints is zero.'
        
        # If we have an active position, manage it
        if session.active_position:
            pos = session.active_position

            # Check exits using existing stops first (conservative intrabar fill).
            exit = self._resolve_exit_for_bar(pos, bar)
            if exit:
                exit_reason, exit_fill_price = exit
                trade = self._close_position(
                    session,
                    exit_fill_price,
                    timestamp,
                    exit_reason,
                    bar_volume=bar.volume,
                )
                session.last_exit_bar_index = current_bar_index
                result['trade_closed'] = trade.to_dict()
                result['action'] = f'position_closed_{exit_reason}'
                # Include position_closed info for frontend markers with detailed data
                result['position_closed'] = {
                    'exit_price': trade.exit_price,
                    'side': trade.side,
                    'exit_reason': exit_reason,
                    'pnl_pct': trade.pnl_pct,
                    'pnl_dollars': trade.pnl_dollars,
                    'strategy': trade.strategy,
                    'entry_price': trade.entry_price,
                    'entry_time': trade.entry_time.isoformat(),
                    'exit_time': trade.exit_time.isoformat(),
                    'size': trade.size,
                    'bars_held': len([b for b in session.bars if b.timestamp >= trade.entry_time and b.timestamp <= trade.exit_time]),
                    'signal_bar_index': trade.signal_bar_index,
                    'entry_bar_index': trade.entry_bar_index,
                    'signal_timestamp': trade.signal_timestamp,
                    'signal_price': trade.signal_price,
                    'costs': {
                        'slippage': round(trade.slippage, 4),
                        'commission': round(trade.commission, 4),
                        'reg_fee': round(trade.reg_fee, 4),
                        'sec_fee': round(trade.sec_fee, 6),
                        'finra_fee': round(trade.finra_fee, 6),
                        'market_impact': round(trade.market_impact, 4),
                        'total': round(trade.total_costs, 4)
                    }
                }

            # Optional partial scale-out at 1R before final exit.
            if session.active_position:
                partial_trade = self._maybe_take_partial_profit(session, session.active_position, bar, timestamp)
                if partial_trade:
                    result['partial_trade_closed'] = partial_trade.to_dict()
                    result['action'] = 'partial_take_profit'

            # Time-based exit if trade stalls too long.
            if session.active_position and self._should_time_exit(session, session.active_position, current_bar_index):
                trade = self._close_position(
                    session,
                    bar.close,
                    timestamp,
                    "time_exit",
                    bar_volume=bar.volume,
                )
                session.last_exit_bar_index = current_bar_index
                result['trade_closed'] = trade.to_dict()
                result['action'] = 'position_closed_time_exit'
                result['position_closed'] = {
                    'exit_price': trade.exit_price,
                    'side': trade.side,
                    'exit_reason': 'time_exit',
                    'pnl_pct': trade.pnl_pct,
                    'pnl_dollars': trade.pnl_dollars,
                    'strategy': trade.strategy,
                }

            # Adverse flow exit if L2 pressure flips against the position.
            if session.active_position:
                should_exit_adverse, adverse_metrics = self._should_adverse_flow_exit(
                    session,
                    session.active_position,
                    current_bar_index,
                )
                if should_exit_adverse:
                    trade = self._close_position(
                        session,
                        bar.close,
                        timestamp,
                        "adverse_flow",
                        bar_volume=bar.volume,
                    )
                    session.last_exit_bar_index = current_bar_index
                    result['trade_closed'] = trade.to_dict()
                    result['action'] = 'position_closed_adverse_flow'
                    result['position_closed'] = {
                        'exit_price': trade.exit_price,
                        'side': trade.side,
                        'exit_reason': 'adverse_flow',
                        'pnl_pct': trade.pnl_pct,
                        'pnl_dollars': trade.pnl_dollars,
                        'strategy': trade.strategy,
                    }
                    result['adverse_flow'] = adverse_metrics

            # Trailing stop updates are based on close and become effective next bar.
            if session.active_position:
                self._update_trailing_from_close(session, session.active_position, bar)
        
        # Custom Rule: Max Daily Loss Circuit Breaker
        current_pnl = sum(t.pnl_dollars for t in session.trades)
        
        if current_pnl < -self.max_daily_loss:
             if session.active_position:
                 trade = self._close_position(
                     session,
                     current_price,
                     timestamp,
                     "max_daily_loss",
                     bar_volume=bar.volume,
                 )
                 result['trade_closed'] = trade.to_dict()
                 result['action'] = 'max_loss_stop'
                 result['position_closed'] = {
                    'exit_price': trade.exit_price,
                    'side': trade.side,
                    'exit_reason': 'max_daily_loss',
                    'pnl_pct': trade.pnl_pct,
                    'pnl_dollars': trade.pnl_dollars,
                    'strategy': trade.strategy
                 }
            
             # Stop trading for the day
             session.phase = SessionPhase.END_OF_DAY
             session.end_price = current_price
             result['session_summary'] = self._get_session_summary(session)
             return result


        regime_update = self._maybe_refresh_regime(session, current_bar_index, timestamp)
        if regime_update:
            result['regime_update'] = regime_update
            result['regime'] = regime_update.get("regime")
            result['micro_regime'] = regime_update.get("micro_regime")
            result['strategies'] = regime_update.get("strategies", [])
            result['strategy'] = regime_update.get("strategy")
            result['indicators'] = regime_update.get("indicators", {})

        if not session.active_position and session.selected_strategy:
            # Check trade limits
            # Check max trades per day
            if len(session.trades) >= self.max_trades_per_day:
                result['action'] = 'trade_limit_reached'
                result['reason'] = f'Max trades per day ({self.max_trades_per_day}) reached'
                return result
            
            # Check cooldown between trades
            last_trade_bar = self.last_trade_bar_index.get(session_key, -self.trade_cooldown_bars)
            bars_since_last_trade = current_bar_index - last_trade_bar
            if bars_since_last_trade < self.trade_cooldown_bars:
                result['action'] = 'cooldown_active'
                result['reason'] = f'Cooldown: {self.trade_cooldown_bars - bars_since_last_trade} bars remaining'
                return result
            
            # Optional ticker-specific time filter from AOS config.
            bar_time = self._to_market_time(timestamp).time()
            bar_hour = bar_time.hour
            
            ticker_aos_config = self.ticker_params.get(session.ticker.upper(), {})
            trading_hours = ticker_aos_config.get('trading_hours', None)

            if trading_hours and ticker_aos_config.get("time_filter_enabled", True):
                if bar_hour not in trading_hours:
                    result['action'] = 'time_filter'
                    result['reason'] = f'Hour {bar_hour}:00 not in allowed hours {trading_hours}'
                    return result
            
            # ── Multi-Layer Decision Engine ──────────────────────────
            bars_data = session.bars[-100:] if len(session.bars) >= 100 else session.bars
            ohlcv = {
                'open': [b.open for b in bars_data],
                'high': [b.high for b in bars_data],
                'low': [b.low for b in bars_data],
                'close': [b.close for b in bars_data],
                'volume': [b.volume for b in bars_data]
            }
            indicators = self._calculate_indicators(bars_data)
            regime = session.detected_regime or Regime.MIXED
            flow_metrics = indicators.get('order_flow') or {}

            ticker_cfg = self.ticker_params.get(session.ticker.upper(), {})
            is_long_only = bool(ticker_cfg.get("long_only", False))

            # Wrap existing signal generation as a callback for Layer 2
            def gen_signal_fn():
                return self._generate_signal(session, bar, timestamp)

            ml_engine = session.multi_layer or self.multi_layer
            if flow_metrics.get("has_l2_coverage", False):
                ml_engine.strategy_weight = 1.0
            decision = ml_engine.evaluate(
                ohlcv=ohlcv,
                indicators=indicators,
                regime=regime,
                strategies=self.strategies,
                active_strategy_names=session.active_strategies,
                current_price=current_price,
                timestamp=timestamp,
                ticker=session.ticker,
                generate_signal_fn=gen_signal_fn,
                is_long_only=is_long_only,
            )

            # Always report detected patterns for frontend visibility
            if decision.patterns:
                result['patterns_detected'] = [p.to_dict() for p in decision.patterns]

            # Always report layer scores
            result['layer_scores'] = {
                'pattern_score': round(decision.pattern_score, 1),
                'strategy_score': round(decision.strategy_score, 1),
                'combined_score': round(decision.combined_score, 1),
                'threshold': decision.threshold,
                'pattern_weight': round(ml_engine.pattern_weight, 3),
                'strategy_weight': round(ml_engine.strategy_weight, 3),
                'flow_score': round(float(flow_metrics.get('flow_score', 0.0) or 0.0), 1),
                'book_pressure_avg': round(float(flow_metrics.get('book_pressure_avg', 0.0) or 0.0), 3),
                'book_pressure_trend': round(float(flow_metrics.get('book_pressure_trend', 0.0) or 0.0), 3),
                'large_trader_activity': round(float(flow_metrics.get('large_trader_activity', 0.0) or 0.0), 3),
                'vwap_execution_flow': round(float(flow_metrics.get('vwap_execution_flow', 0.0) or 0.0), 3),
                'micro_regime': session.micro_regime,
                'passed': decision.execute,
            }

            if decision.execute and decision.signal:
                signal = decision.signal
                l2_passed, l2_metrics = self._passes_l2_confirmation(session, signal)
                result['l2_confirmation'] = l2_metrics

                if not l2_passed:
                    result['action'] = 'l2_filtered'
                    result['reason'] = l2_metrics.get('reason', 'l2_confirmation_failed')
                    return result

                signal.metadata.setdefault('l2_confirmation', l2_metrics)
                session.signals.append(signal)
                result['signal'] = signal.to_dict()
                result['signals'] = [signal.to_dict()]  # Array format for frontend

                # Queue signal for next bar open (no same-bar execution).
                if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                    session.pending_signal = signal
                    session.pending_signal_bar_index = current_bar_index
                    result['action'] = 'signal_queued'
                    result['queued_for_next_bar'] = True

        return result

    def _effective_stop_for_position(self, pos: Position) -> tuple[Optional[float], Optional[str]]:
        """Return effective stop level and reason for the current side."""
        candidates: List[tuple] = []
        if pos.stop_loss and pos.stop_loss > 0:
            candidates.append(("stop_loss", float(pos.stop_loss)))
        if pos.trailing_stop_active and pos.trailing_stop_price and pos.trailing_stop_price > 0:
            candidates.append(("trailing_stop", float(pos.trailing_stop_price)))

        if not candidates:
            return None, None

        if pos.side == 'long':
            reason, level = max(candidates, key=lambda x: x[1])
        else:
            reason, level = min(candidates, key=lambda x: x[1])
        return level, reason

    def _resolve_exit_for_bar(self, pos: Position, bar: BarData) -> Optional[tuple]:
        """
        Resolve exit for the current bar with conservative tie-break:
        if both stop and target are hit, stop wins.
        """
        stop_level, stop_reason = self._effective_stop_for_position(pos)
        if pos.side == 'long':
            stop_hit = stop_level is not None and bar.low <= stop_level
            tp_hit = pos.take_profit > 0 and bar.high >= pos.take_profit
        else:
            stop_hit = stop_level is not None and bar.high >= stop_level
            tp_hit = pos.take_profit > 0 and bar.low <= pos.take_profit

        if stop_hit:
            return (stop_reason or "stop_loss", float(stop_level))
        if tp_hit:
            return ("take_profit", float(pos.take_profit))
        return None

    def _update_trailing_from_close(self, session: TradingSession, pos: Position, bar: BarData) -> None:
        """Update trailing stop from close; this will be effective on the next bar."""
        if not pos.trailing_stop_active:
            return
        if pos.side == 'long':
            if bar.close > pos.highest_price:
                pos.highest_price = bar.close
            new_stop = pos.highest_price * (1 - session.trailing_stop_pct / 100)
            if new_stop > pos.trailing_stop_price:
                pos.trailing_stop_price = new_stop
        else:
            if bar.close < pos.lowest_price:
                pos.lowest_price = bar.close
            new_stop = pos.lowest_price * (1 + session.trailing_stop_pct / 100)
            if new_stop < pos.trailing_stop_price or pos.trailing_stop_price == 0:
                pos.trailing_stop_price = new_stop

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _bar_has_l2_data(bar: BarData) -> bool:
        return (
            bar.l2_delta is not None
            or bar.l2_imbalance is not None
            or bar.l2_volume is not None
            or bar.l2_book_pressure is not None
            or bar.l2_bid_depth_total is not None
            or bar.l2_ask_depth_total is not None
            or bar.l2_iceberg_bias is not None
        )

    def _passes_l2_confirmation(self, session: TradingSession, signal: Signal) -> tuple[bool, Dict[str, Any]]:
        """
        Optional order-flow confirmation gate.

        Uses only current/past bars (no look-ahead):
        - summed delta over lookback
        - mean imbalance over lookback
        - summed iceberg bias over lookback
        """
        metrics: Dict[str, Any] = {
            "enabled": bool(session.l2_confirm_enabled),
            "lookback_bars": max(1, int(session.l2_lookback_bars)),
            "min_delta": float(session.l2_min_delta),
            "min_imbalance": float(session.l2_min_imbalance),
            "min_iceberg_bias": float(session.l2_min_iceberg_bias),
            "min_participation_ratio": float(session.l2_min_participation_ratio),
            "min_directional_consistency": float(session.l2_min_directional_consistency),
            "min_signed_aggression": float(session.l2_min_signed_aggression),
            "passed": True,
        }

        if not session.l2_confirm_enabled:
            return True, metrics

        lookback = metrics["lookback_bars"]
        window = session.bars[-lookback:] if session.bars else []
        if not window:
            metrics.update({"passed": False, "reason": "l2_window_empty"})
            return False, metrics

        deltas = [self._to_float(b.l2_delta, 0.0) for b in window]
        imbalances = [self._to_float(b.l2_imbalance, 0.0) for b in window]
        iceberg_biases = [self._to_float(b.l2_iceberg_bias, 0.0) for b in window]

        has_any_l2 = any(
            (b.l2_delta is not None)
            or (b.l2_imbalance is not None)
            or (b.l2_iceberg_bias is not None)
            for b in window
        )
        if not has_any_l2:
            metrics.update({"passed": False, "reason": "l2_data_missing"})
            return False, metrics

        delta_sum = float(sum(deltas))
        imbalance_avg = float(sum(imbalances) / len(imbalances)) if imbalances else 0.0
        iceberg_bias_sum = float(sum(iceberg_biases))
        l2_volumes = [max(0.0, self._to_float(b.l2_volume, 0.0)) for b in window]
        bar_volumes = [max(0.0, self._to_float(b.volume, 0.0)) for b in window]

        direction = 1.0 if signal.signal_type == SignalType.BUY else -1.0
        directional_delta = delta_sum * direction
        directional_imbalance = imbalance_avg * direction
        directional_iceberg_bias = iceberg_bias_sum * direction

        participation_samples: List[float] = []
        signed_aggression_samples: List[float] = []
        directional_consistency_base = 0
        directional_consistency_hits = 0

        for b, l2_vol, bar_vol in zip(window, l2_volumes, bar_volumes):
            if l2_vol > 0 and bar_vol > 0:
                participation_samples.append(l2_vol / bar_vol)

            delta_val = self._to_float(b.l2_delta, 0.0)
            if l2_vol > 0:
                signed_aggression_samples.append((delta_val / l2_vol) * direction)

            # Consistency check uses delta sign first, then imbalance sign fallback.
            if abs(delta_val) > 1e-9:
                directional_consistency_base += 1
                if (delta_val * direction) > 0:
                    directional_consistency_hits += 1
                continue

            if b.l2_imbalance is not None:
                imb_val = self._to_float(b.l2_imbalance, 0.0)
                if abs(imb_val) > 1e-9:
                    directional_consistency_base += 1
                    if (imb_val * direction) > 0:
                        directional_consistency_hits += 1

        participation_avg = (
            float(sum(participation_samples) / len(participation_samples))
            if participation_samples else 0.0
        )
        signed_aggression_avg = (
            float(sum(signed_aggression_samples) / len(signed_aggression_samples))
            if signed_aggression_samples else 0.0
        )
        directional_consistency = (
            float(directional_consistency_hits / directional_consistency_base)
            if directional_consistency_base > 0 else 0.0
        )

        passes_delta = directional_delta >= float(session.l2_min_delta)
        passes_imbalance = directional_imbalance >= float(session.l2_min_imbalance)
        passes_iceberg = directional_iceberg_bias >= float(session.l2_min_iceberg_bias)
        passes_participation = participation_avg >= float(session.l2_min_participation_ratio)
        passes_consistency = directional_consistency >= float(session.l2_min_directional_consistency)
        passes_signed_aggression = signed_aggression_avg >= float(session.l2_min_signed_aggression)
        passed = bool(
            passes_delta
            and passes_imbalance
            and passes_iceberg
            and passes_participation
            and passes_consistency
            and passes_signed_aggression
        )

        metrics.update({
            "window_size": len(window),
            "delta_sum": delta_sum,
            "imbalance_avg": imbalance_avg,
            "iceberg_bias_sum": iceberg_bias_sum,
            "participation_avg": participation_avg,
            "directional_consistency": directional_consistency,
            "signed_aggression_avg": signed_aggression_avg,
            "directional_delta": directional_delta,
            "directional_imbalance": directional_imbalance,
            "directional_iceberg_bias": directional_iceberg_bias,
            "passes_delta": passes_delta,
            "passes_imbalance": passes_imbalance,
            "passes_iceberg": passes_iceberg,
            "passes_participation": passes_participation,
            "passes_consistency": passes_consistency,
            "passes_signed_aggression": passes_signed_aggression,
            "passed": passed,
        })

        if not passed:
            metrics["reason"] = "l2_confirmation_failed"
        return passed, metrics
    
    def _generate_signal(
        self, 
        session: TradingSession, 
        bar: BarData, 
        timestamp: datetime
    ) -> Optional[Signal]:
        """
        Generate trading signal using ALL active strategies.
        Returns the signal with highest confidence.
        """
        active_strategies = session.active_strategies
        if not active_strategies:
            # Fallback for old sessions or empty list
            if session.selected_strategy:
                active_strategies = [session.selected_strategy]
            else:
                return None
        
        # Prepare data once
        bars = session.bars[-100:] if len(session.bars) >= 100 else session.bars
        ohlcv = {
            'open': [b.open for b in bars],
            'high': [b.high for b in bars],
            'low': [b.low for b in bars],
            'close': [b.close for b in bars],
            'volume': [b.volume for b in bars]
        }
        indicators = self._calculate_indicators(bars)
        regime = session.detected_regime or Regime.MIXED

        candidate_signals = []
        ticker_cfg = self.ticker_params.get(session.ticker.upper(), {})
        is_long_only = bool(ticker_cfg.get("long_only", False))
        flow_metrics = indicators.get("order_flow") or {}
        flow_available = bool(flow_metrics.get("has_l2_coverage", False))
        flow_strategy_keys = {"absorption_reversal", "momentum_flow", "exhaustion_fade"}

        for strategy_name in active_strategies:
            if strategy_name not in self.strategies:
                continue
            
            strategy = self.strategies[strategy_name]

            signal = self._generate_signal_with_overrides(
                strategy=strategy,
                overrides=self._get_ticker_strategy_overrides(session.ticker, strategy_name),
                current_price=bar.close,
                ohlcv=ohlcv,
                indicators=indicators,
                regime=regime,
                timestamp=timestamp
            )
            
            if signal:
                if is_long_only and signal.signal_type == SignalType.SELL:
                    continue
                strategy_key = self._canonical_strategy_key(signal.strategy_name)
                edge_adjustment = self._strategy_edge_adjustment(session, strategy_key)
                original_confidence = float(signal.confidence)
                signal.confidence = max(1.0, min(100.0, original_confidence + edge_adjustment))
                signal.metadata.setdefault("confidence_adjustment", {
                    "base_confidence": original_confidence,
                    "edge_adjustment": edge_adjustment,
                    "adjusted_confidence": signal.confidence,
                })
                candidate_signals.append(signal)
        
        if not candidate_signals:
            return None
            
        # Select best signal using confidence + strategy preference + recent edge.
        rank = {name: idx for idx, name in enumerate(active_strategies)}

        strategy_trade_history: Dict[str, List[float]] = {}
        for trade in session.trades:
            trade_key = self._canonical_strategy_key(trade.strategy)
            strategy_trade_history.setdefault(trade_key, []).append(trade.gross_pnl_pct)

        scored = []
        for sig in candidate_signals:
            key = self._canonical_strategy_key(sig.strategy_name)
            preference_bonus = max(0.0, 6.0 - (rank.get(key, 99) * 3.0))
            recent = strategy_trade_history.get(key, [])[-3:]
            perf_bonus = 0.0
            if recent:
                perf_bonus = max(-8.0, min(8.0, (sum(recent) / len(recent)) * 2.0))

            flow_bonus = 0.0
            if flow_available:
                flow_bonus = 8.0 if key in flow_strategy_keys else -2.5

            score = float(sig.confidence) + preference_bonus + perf_bonus + flow_bonus
            scored.append((score, sig))
        scored.sort(key=lambda item: item[0], reverse=True)
        best_signal = scored[0][1]
        
        return best_signal
    
    def _calculate_indicators(self, bars: List[BarData]) -> Dict[str, Any]:
        """Calculate indicators from bars."""
        if len(bars) < 5:
            return {"order_flow": self._calculate_order_flow_metrics(bars, lookback=len(bars) or 1)}
        
        closes = [b.close for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        volumes = [b.volume for b in bars]
        
        indicators = {}
        
        # SMA 20
        if len(closes) >= 20:
            indicators['sma'] = [sum(closes[max(0, i-19):i+1]) / min(i+1, 20) for i in range(len(closes))]
        
        # EMA 10
        if len(closes) >= 10:
            ema = []
            multiplier = 2 / (10 + 1)
            for i, c in enumerate(closes):
                if i == 0:
                    ema.append(c)
                else:
                    ema.append((c - ema[-1]) * multiplier + ema[-1])
            indicators['ema'] = ema
            indicators['ema_fast'] = ema
        
        # EMA 20 (slow)
        if len(closes) >= 20:
            ema_slow = []
            multiplier = 2 / (20 + 1)
            for i, c in enumerate(closes):
                if i == 0:
                    ema_slow.append(c)
                else:
                    ema_slow.append((c - ema_slow[-1]) * multiplier + ema_slow[-1])
            indicators['ema_slow'] = ema_slow
        
        # RSI 14
        if len(closes) >= 15:
            gains = []
            losses = []
            for i in range(1, len(closes)):
                change = closes[i] - closes[i-1]
                gains.append(max(change, 0))
                losses.append(abs(min(change, 0)))
            
            rsi = []
            for i in range(13, len(gains)):
                avg_gain = sum(gains[max(0, i-13):i+1]) / 14
                avg_loss = sum(losses[max(0, i-13):i+1]) / 14
                if avg_loss == 0:
                    rsi.append(100)
                else:
                    rs = avg_gain / avg_loss
                    rsi.append(100 - (100 / (1 + rs)))
            
            # Pad with initial values
            indicators['rsi'] = [50] * (len(closes) - len(rsi)) + rsi
        
        # ATR 14
        if len(closes) >= 14:
            tr = []
            for i in range(1, len(bars)):
                hl = highs[i] - lows[i]
                hc = abs(highs[i] - closes[i-1])
                lc = abs(lows[i] - closes[i-1])
                tr.append(max(hl, hc, lc))
            
            atr = []
            for i in range(13, len(tr)):
                atr.append(sum(tr[max(0, i-13):i+1]) / 14)
            
            indicators['atr'] = [atr[0] if atr else 0] * (len(closes) - len(atr)) + atr
        
        # VWAP (if available in bars)
        vwaps = [b.vwap for b in bars if b.vwap is not None]
        if vwaps:
            indicators['vwap'] = vwaps
        else:
            # Calculate simple VWAP approximation
            cum_vol = 0
            cum_pv = 0
            vwap = []
            for b in bars:
                typical_price = (b.high + b.low + b.close) / 3
                cum_vol += b.volume
                cum_pv += typical_price * b.volume
                vwap.append(cum_pv / cum_vol if cum_vol > 0 else typical_price)
            indicators['vwap'] = vwap
        
        # Calculate ADX series incrementally (point-in-time, no look-ahead)
        adx_series = self._calc_adx_series(bars, 14)
        indicators['adx'] = adx_series if adx_series else [0.0] * len(closes)

        # Flow metrics are computed from current/past bars only.
        indicators['order_flow'] = self._calculate_order_flow_metrics(bars, lookback=min(24, len(bars)))
        
        return indicators

    def _calculate_position_size(
        self,
        session: TradingSession,
        signal: Signal,
        entry_price: float,
    ) -> float:
        """
        Volatility/risk-aware position sizing.

        Uses both risk-to-stop and max-notional caps:
        - risk budget = account * risk_per_trade_pct
        - size by risk = risk_budget / stop_distance
        - size by notional = account * max_position_notional_pct / entry_price
        """
        if entry_price <= 0:
            return 0.0

        capital = max(0.0, float(session.account_size_usd))
        if capital <= 0:
            return 0.0

        risk_budget = capital * (max(0.1, float(session.risk_per_trade_pct)) / 100.0)
        flow_confidence_factor = max(0.5, min(1.5, float(signal.confidence) / 80.0))
        risk_budget *= flow_confidence_factor
        max_notional = capital * (max(1.0, float(session.max_position_notional_pct)) / 100.0)
        size_by_notional = max_notional / entry_price

        stop_distance = 0.0
        if signal.stop_loss and signal.stop_loss > 0:
            stop_distance = abs(entry_price - float(signal.stop_loss))
        if stop_distance <= 0:
            # Fallback: 0.5% synthetic stop distance when strategy did not provide stop.
            stop_distance = max(entry_price * 0.005, 0.01)

        size_by_risk = risk_budget / stop_distance
        desired_size = min(size_by_notional, size_by_risk)
        if desired_size <= 0:
            return 0.0

        min_size = max(0.0, float(session.min_position_size))
        if desired_size < min_size and size_by_notional >= min_size:
            desired_size = min_size
        return round(max(0.0, desired_size), 4)

    def _simulate_entry_fill(
        self,
        desired_size: float,
        bar_volume: float,
        session: TradingSession,
    ) -> Tuple[float, float]:
        """
        Deterministic partial-fill simulation.

        If desired size exceeds max allowed participation on this bar, reduce fill size.
        """
        if desired_size <= 0:
            return 0.0, 0.0
        if bar_volume <= 0:
            return desired_size, 1.0

        max_participation = min(1.0, max(0.01, float(session.max_fill_participation_rate)))
        max_fillable = bar_volume * max_participation
        if desired_size <= max_fillable:
            return desired_size, 1.0

        raw_ratio = max_fillable / desired_size if desired_size > 0 else 0.0
        min_ratio = min(1.0, max(0.01, float(session.min_fill_ratio)))
        fill_ratio = min(1.0, max(min_ratio, raw_ratio))
        return round(desired_size * fill_ratio, 4), round(fill_ratio, 4)

    def _bars_held(self, pos: Position, current_bar_index: int) -> int:
        entry_index = pos.entry_bar_index if pos.entry_bar_index is not None else current_bar_index
        return max(0, int(current_bar_index) - int(entry_index))

    def _partial_take_profit_price(self, session: TradingSession, pos: Position) -> float:
        if pos.stop_loss <= 0:
            return 0.0
        rr = max(0.25, float(session.partial_take_profit_rr))
        risk = abs(pos.entry_price - pos.stop_loss)
        if risk <= 0:
            return 0.0
        if pos.side == "long":
            return pos.entry_price + (risk * rr)
        return pos.entry_price - (risk * rr)

    def _build_trade_record(
        self,
        session: TradingSession,
        pos: Position,
        exit_price: float,
        exit_time: datetime,
        reason: str,
        shares: float,
        bar_volume: Optional[float] = None,
    ) -> DayTrade:
        shares = max(0.0, float(shares))
        if shares <= 0:
            raise ValueError("shares must be > 0 when building trade record")

        if pos.side == 'long':
            gross_pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
        else:
            gross_pnl_pct = (pos.entry_price - exit_price) / pos.entry_price * 100

        gross_pnl_dollars = gross_pnl_pct * pos.entry_price / 100 * shares

        costs = self.trading_costs.calculate_costs(
            entry_price=pos.entry_price,
            exit_price=exit_price,
            shares=shares,
            side=pos.side,
            avg_bar_volume=bar_volume,
        )

        net_pnl_dollars = gross_pnl_dollars - costs['total']
        notional = pos.entry_price * shares
        net_pnl_pct = (net_pnl_dollars / notional * 100) if notional > 0 else 0.0

        session.trade_counter += 1
        signal_metadata = pos.signal_metadata if isinstance(pos.signal_metadata, dict) else {}
        order_flow_md = signal_metadata.get("order_flow") if isinstance(signal_metadata, dict) else {}
        if not isinstance(order_flow_md, dict):
            order_flow_md = {}

        flow_snapshot: Dict[str, Any] = {}
        for key in (
            "signed_aggression",
            "directional_consistency",
            "imbalance_avg",
            "sweep_intensity",
            "book_pressure_avg",
            "book_pressure_trend",
            "absorption_rate",
            "delta_price_divergence",
            "delta_zscore",
            "price_change_pct",
            "delta_acceleration",
        ):
            if key in order_flow_md:
                flow_snapshot[key] = self._to_float(order_flow_md.get(key), 0.0)

        l2_md = signal_metadata.get("l2_confirmation") if isinstance(signal_metadata, dict) else None
        if isinstance(l2_md, dict):
            flow_snapshot["l2_confirmation_passed"] = bool(l2_md.get("passed", False))

        trade = DayTrade(
            id=session.trade_counter,
            strategy=pos.strategy_name,
            side=pos.side,
            entry_price=pos.entry_price,
            entry_time=pos.entry_time,
            exit_price=exit_price,
            exit_time=exit_time,
            size=shares,
            pnl_pct=net_pnl_pct,
            pnl_dollars=net_pnl_dollars,
            exit_reason=reason,
            slippage=costs.get('slippage', 0.0),
            commission=costs.get('commission', 0.0),
            reg_fee=costs.get('reg_fee', 0.0),
            sec_fee=costs.get('sec_fee', 0.0),
            finra_fee=costs.get('finra_fee', 0.0),
            market_impact=costs.get('market_impact', 0.0),
            total_costs=costs.get('total', 0.0),
            gross_pnl_pct=gross_pnl_pct,
            signal_bar_index=pos.signal_bar_index,
            entry_bar_index=pos.entry_bar_index,
            signal_timestamp=pos.signal_timestamp,
            signal_price=pos.signal_price,
            signal_metadata=signal_metadata,
            flow_snapshot=flow_snapshot,
        )
        session.trades.append(trade)
        session.total_pnl += net_pnl_pct
        return trade

    def _maybe_take_partial_profit(
        self,
        session: TradingSession,
        pos: Position,
        bar: BarData,
        timestamp: datetime,
    ) -> Optional[DayTrade]:
        if not session.enable_partial_take_profit:
            return None
        if pos.partial_exit_done or pos.size <= 0:
            return None
        if pos.stop_loss <= 0:
            return None

        partial_price = pos.partial_take_profit_price or self._partial_take_profit_price(session, pos)
        if partial_price <= 0:
            return None
        pos.partial_take_profit_price = partial_price

        hit_rr_target = (bar.high >= partial_price) if pos.side == "long" else (bar.low <= partial_price)

        flow = self._calculate_order_flow_metrics(session.bars, lookback=min(8, len(session.bars)))
        signed_aggression = float(flow.get("signed_aggression", 0.0) or 0.0)
        if pos.side == "long":
            flow_deteriorating = signed_aggression < 0.0 and bar.close > pos.entry_price
        else:
            flow_deteriorating = signed_aggression > 0.0 and bar.close < pos.entry_price

        if not hit_rr_target and not flow_deteriorating:
            return None

        close_fraction = 0.5 if flow_deteriorating and not hit_rr_target else float(session.partial_take_profit_fraction)
        close_fraction = min(0.95, max(0.05, close_fraction))
        close_size = max(0.0, min(pos.size, pos.size * close_fraction))
        if close_size <= 0:
            return None

        reason = "partial_take_profit"
        if flow_deteriorating and not hit_rr_target:
            reason = "partial_take_profit_flow_deterioration"

        trade = self._build_trade_record(
            session=session,
            pos=pos,
            exit_price=(bar.close if flow_deteriorating and not hit_rr_target else partial_price),
            exit_time=timestamp,
            reason=reason,
            shares=close_size,
            bar_volume=bar.volume,
        )
        pos.size = max(0.0, pos.size - close_size)
        pos.partial_exit_done = True

        # After partial, protect remaining position at breakeven or better.
        if pos.side == "long":
            pos.stop_loss = max(pos.stop_loss, pos.entry_price)
        else:
            pos.stop_loss = min(pos.stop_loss, pos.entry_price) if pos.stop_loss > 0 else pos.entry_price

        if pos.size <= 0:
            session.active_position = None
        return trade

    def _should_time_exit(self, session: TradingSession, pos: Position, current_bar_index: int) -> bool:
        if session.time_exit_bars <= 0:
            return False
        limit_bars = float(session.time_exit_bars)
        flow = self._calculate_order_flow_metrics(session.bars, lookback=min(8, len(session.bars)))
        signed_aggression = float(flow.get("signed_aggression", 0.0) or 0.0)
        favorable = (
            (pos.side == "long" and signed_aggression >= 0.10)
            or (pos.side == "short" and signed_aggression <= -0.10)
        )
        if favorable:
            limit_bars *= 1.5
        return self._bars_held(pos, current_bar_index) >= int(limit_bars)

    def _should_adverse_flow_exit(
        self,
        session: TradingSession,
        pos: Position,
        current_bar_index: int,
    ) -> Tuple[bool, Dict[str, float]]:
        metrics = {"signed_aggression": 0.0, "directional_consistency": 0.0, "book_pressure_avg": 0.0}
        if not session.adverse_flow_exit_enabled:
            return False, metrics
        if self._bars_held(pos, current_bar_index) < int(session.adverse_flow_min_hold_bars):
            return False, metrics

        flow = self._calculate_order_flow_metrics(session.bars, lookback=min(12, len(session.bars)))
        if not flow.get("has_l2_coverage", False):
            return False, metrics

        signed = float(flow.get("signed_aggression", 0.0) or 0.0)
        consistency = float(flow.get("directional_consistency", 0.0) or 0.0)
        book_pressure_avg = float(flow.get("book_pressure_avg", 0.0) or 0.0)
        threshold = max(0.02, float(session.adverse_flow_threshold))
        metrics = {
            "signed_aggression": signed,
            "directional_consistency": consistency,
            "book_pressure_avg": book_pressure_avg,
        }

        if pos.side == "long":
            adverse_flow = signed <= -threshold and consistency >= 0.45
            adverse_book = book_pressure_avg <= -0.15
            return (adverse_flow or adverse_book), metrics
        adverse_flow = signed >= threshold and consistency >= 0.45
        adverse_book = book_pressure_avg >= 0.15
        return (adverse_flow or adverse_book), metrics
    
    def _open_position(
        self,
        session: TradingSession,
        signal: Signal,
        entry_price: Optional[float] = None,
        entry_time: Optional[datetime] = None,
        signal_bar_index: Optional[int] = None,
        entry_bar_index: Optional[int] = None,
        entry_bar_volume: Optional[float] = None,
    ) -> Position:
        """Open a new position from signal."""
        side = 'long' if signal.signal_type == SignalType.BUY else 'short'
        fill_price = float(entry_price if entry_price is not None else signal.price)
        fill_time = entry_time or signal.timestamp
        desired_size = self._calculate_position_size(session, signal, fill_price)
        size, fill_ratio = self._simulate_entry_fill(
            desired_size=desired_size,
            bar_volume=max(0.0, float(entry_bar_volume or 0.0)),
            session=session,
        )
        
        position = Position(
            strategy_name=signal.strategy_name,
            entry_price=fill_price,
            entry_time=fill_time,
            side=side,
            size=size,
            stop_loss=signal.stop_loss or 0,
            take_profit=signal.take_profit or 0,
            trailing_stop_active=signal.trailing_stop,
            highest_price=fill_price if side == 'long' else 0,
            lowest_price=fill_price if side == 'short' else float('inf'),
            fill_ratio=fill_ratio,
            initial_size=size,
        )
        # Optional audit metadata (keeps backwards compatibility).
        if signal_bar_index is not None:
            position.signal_bar_index = signal_bar_index
        if entry_bar_index is not None:
            position.entry_bar_index = entry_bar_index
        position.signal_timestamp = signal.timestamp.isoformat()
        position.signal_price = signal.price
        signal_metadata = signal.metadata if isinstance(signal.metadata, dict) else {}
        try:
            position.signal_metadata = json.loads(json.dumps(signal_metadata, default=str))
        except Exception:
            position.signal_metadata = dict(signal_metadata)
        if session.enable_partial_take_profit and position.stop_loss > 0:
            position.partial_take_profit_price = self._partial_take_profit_price(session, position)
        
        session.active_position = position
        session.trailing_stop_pct = signal.trailing_stop_pct or 0.8
        
        return position
    
    def _close_position(
        self, 
        session: TradingSession, 
        exit_price: float, 
        exit_time: datetime, 
        reason: str,
        bar_volume: Optional[float] = None,
    ) -> DayTrade:
        """Close position and record trade with trading costs."""
        pos = session.active_position
        trade = self._build_trade_record(
            session=session,
            pos=pos,
            exit_price=exit_price,
            exit_time=exit_time,
            reason=reason,
            shares=pos.size,
            bar_volume=bar_volume,
        )
        session.active_position = None
        
        return trade
    
    def _get_session_summary(self, session: TradingSession) -> Dict[str, Any]:
        """Get summary of trading session."""
        trades = session.trades
        
        if not trades:
            return {
                'ticker': session.ticker,
                'date': session.date,
                'regime': session.detected_regime.value if session.detected_regime else None,
                'micro_regime': session.micro_regime,
                'strategy': session.selected_strategy,
                'total_trades': 0,
                'total_pnl_pct': 0,
                'success': False
            }
        
        winning = [t for t in trades if t.pnl_pct > 0]
        losing = [t for t in trades if t.pnl_pct <= 0]
        
        return {
            'ticker': session.ticker,
            'date': session.date,
            'run_id': session.run_id,
            'regime': session.detected_regime.value if session.detected_regime else None,
            'micro_regime': session.micro_regime,
            'strategy': session.selected_strategy,
            'total_trades': len(trades),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': len(winning) / len(trades) * 100 if trades else 0,
            'trades': [t.to_dict() for t in trades],
            'total_pnl_pct': round(session.total_pnl, 2),
            'avg_pnl_pct': round(session.total_pnl / len(trades), 2) if trades else 0,
            'best_trade': round(max(t.pnl_pct for t in trades), 2) if trades else 0,
            'worst_trade': round(min(t.pnl_pct for t in trades), 2) if trades else 0,
            'bars_processed': len(session.bars),
            'pre_market_bars': len(session.pre_market_bars),
            'regime_history': list(session.regime_history),
            'success': session.total_pnl > 0,
        }
    
    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active sessions."""
        return {k: v.to_dict() for k, v in self.sessions.items()}
    
    def end_session(self, run_id: str, ticker: str, date: str) -> Optional[Dict[str, Any]]:
        """Manually end a session and get summary."""
        session = self.get_session(run_id, ticker, date)
        
        if not session:
            return None
        
        # Close any open position at last price
        if session.active_position and session.bars:
            last_price = session.bars[-1].close
            last_time = session.bars[-1].timestamp
            self._close_position(
                session,
                last_price,
                last_time,
                'manual_close',
                bar_volume=session.bars[-1].volume,
            )
        
        session.phase = SessionPhase.CLOSED
        if session.bars:
            session.end_price = session.bars[-1].close
        
        return self._get_session_summary(session)
    
    def clear_session(self, run_id: str, ticker: str, date: str) -> bool:
        """Clear a session from memory."""
        key = self._get_session_key(run_id, ticker, date)
        if key in self.sessions:
            del self.sessions[key]
            return True
        return False
