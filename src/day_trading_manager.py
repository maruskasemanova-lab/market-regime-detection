"""
Session-based Day Trading Manager.
Manages trading sessions for individual days with regime detection and strategy execution.
"""
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Optional
from enum import Enum
import json
import logging
import re

logger = logging.getLogger(__name__)

from .strategies.base_strategy import BaseStrategy, Signal, SignalType, Position, Regime
from .strategies.trailing_stop import TrailingStopManager, TrailingStopConfig, StopType
from .strategies.mean_reversion import MeanReversionStrategy
from .strategies.pullback import PullbackStrategy
from .strategies.momentum import MomentumStrategy
from .strategies.rotation import RotationStrategy
from .strategies.vwap_magnet import VWAPMagnetStrategy
from .strategies.volume_profile import VolumeProfileStrategy
from .strategies.gap_liquidity import GapLiquidityStrategy


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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'vwap': self.vwap
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
    total_costs: float = 0.0
    gross_pnl_pct: float = 0.0  # PnL before costs
    
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
                'total': round(self.total_costs, 4)
            },
            'gross_pnl_pct': round(self.gross_pnl_pct, 2)
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
        side: str
    ) -> Dict[str, float]:
        """
        Calculate all trading costs.
        Returns dict with commission, reg_fee, slippage, finra_fee, total.
        """
        # Slippage: $0.01 per share on both entry and exit
        slippage_cost = self.slippage_per_share * shares * 2

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

        total = slippage_cost + commission + reg_fee + finra_fee + sec_fee
        
        return {
            'slippage': slippage_cost,
            'commission': commission,
            'reg_fee': reg_fee,
            'sec_fee': sec_fee,
            'finra_fee': finra_fee,
            'total': total
        }


@dataclass
class TradingSession:
    """Single day trading session for a ticker."""
    run_id: str
    ticker: str
    date: str
    regime_detection_minutes: int = 15
    
    # Session state
    phase: SessionPhase = SessionPhase.PRE_MARKET
    detected_regime: Optional[Regime] = None
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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'run_id': self.run_id,
            'ticker': self.ticker,
            'date': self.date,
            'phase': self.phase.value,
            'regime': self.detected_regime.value if self.detected_regime else None,
            'selected_strategy': self.selected_strategy,
            'bars_count': len(self.bars),
            'pre_market_bars_count': len(self.pre_market_bars),
            'active_position': self.active_position.to_dict() if self.active_position else None,
            'trades_count': len(self.trades),
            'signals_count': len(self.signals),
            'total_pnl': round(self.total_pnl, 2),
            'start_price': self.start_price,
            'end_price': self.end_price
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
    ):
        self.sessions: Dict[str, TradingSession] = {}  # session_key -> TradingSession
        self.regime_detection_minutes = regime_detection_minutes
        self.trading_costs = trading_costs or TradingCosts()
        self.max_daily_loss = max_daily_loss  # Stop trading if loss exceeds this amount
        self.max_trades_per_day = max_trades_per_day  # Limit trades to reduce fees
        self.trade_cooldown_bars = trade_cooldown_bars  # Cooldown between trades
        self.market_tz = ZoneInfo("America/New_York")
        self.run_defaults: Dict[tuple, Dict[str, Any]] = {}
        
        self.last_trade_bar_index = {}  # Track last trade bar per session
        
        # Pre-configured strategies
        self.strategies = {
            'mean_reversion': MeanReversionStrategy(),
            'pullback': PullbackStrategy(),
            'momentum': MomentumStrategy(),
            'rotation': RotationStrategy(),
            'vwap_magnet': VWAPMagnetStrategy(),
            'volume_profile': VolumeProfileStrategy(),
            'gap_liquidity': GapLiquidityStrategy()
        }
        
        # Strategy selection by regime
        # Default Preferences - including new volume-focused strategies
        self.default_preference = {
            Regime.TRENDING: ['momentum', 'pullback', 'gap_liquidity', 'volume_profile', 'vwap_magnet'],
            Regime.CHOPPY: ['mean_reversion', 'vwap_magnet', 'volume_profile'],
            Regime.MIXED: ['volume_profile', 'gap_liquidity', 'mean_reversion', 'vwap_magnet', 'rotation']
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
                
                # Load ticker-specific parameters (do not mutate global strategy
                # instances per ticker; apply overrides at signal generation time).
                for ticker, ticker_config in config.get('tickers', {}).items():
                    ticker_key = str(ticker).upper()
                    primary_strategy = self._canonical_strategy_key(ticker_config.get('strategy', ''))
                    backup_strategy = self._canonical_strategy_key(ticker_config.get('backup_strategy', ''))
                    self.ticker_params[ticker_key] = {
                        'strategy': primary_strategy,
                        'backup_strategy': backup_strategy,
                        'params': ticker_config.get('params', {}),
                        'regime_filter': ticker_config.get('regime_filter', []),
                        'avoid_days': ticker_config.get('avoid_days', []),
                        'trading_hours': ticker_config.get('trading_hours', None),  # Time-based filter
                        'long_only': ticker_config.get('long_only', False),
                        'time_filter_enabled': ticker_config.get('time_filter_enabled', True),
                        'min_confidence': ticker_config.get('min_confidence', 65.0),
                        'max_daily_trades': ticker_config.get('max_daily_trades', 2)
                    }
                
                logger.info(f"Loaded AOS config from {config_path}")
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
            self.sessions[key] = TradingSession(
                run_id=run_id,
                ticker=ticker,
                date=date,
                regime_detection_minutes=regime_detection_minutes or self.regime_detection_minutes
            )
            
            # Rule: No Fridays (Day 4)
            # 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun
            if datetime.strptime(date, "%Y-%m-%d").weekday() == 4:
                self.sessions[key].phase = SessionPhase.CLOSED
            elif not self._is_day_allowed(date, ticker):
                self.sessions[key].phase = SessionPhase.CLOSED
        
        return self.sessions[key]

    def set_run_defaults(
        self,
        run_id: str,
        ticker: str,
        regime_detection_minutes: Optional[int] = None,
        account_size_usd: Optional[float] = None
    ) -> None:
        """Set default parameters for all sessions in a run."""
        key = (run_id, ticker)
        defaults = self.run_defaults.get(key, {})
        if regime_detection_minutes is not None:
            defaults["regime_detection_minutes"] = regime_detection_minutes
        if account_size_usd is not None:
            defaults["account_size_usd"] = account_size_usd
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
            vwap=bar_data.get('vwap')
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
            
            if elapsed_minutes >= session.regime_detection_minutes:
                # Detect regime
                regime = self._detect_regime(session)
                session.detected_regime = regime
                
                # Select BEST strategies for this regime
                active_strategies = self._select_strategies(session)
                session.active_strategies = active_strategies
                session.selected_strategy = (
                    "adaptive" if len(active_strategies) > 1 else (active_strategies[0] if active_strategies else None)
                )
                
                # Transition to trading
                session.phase = SessionPhase.TRADING
                result['action'] = 'regime_detected'
                result['regime'] = regime.value
                result['strategies'] = active_strategies
                result['strategy'] = session.selected_strategy
                # print(f"DEBUG: Regime DETECTED {regime} at {timestamp}. Strategy: {active_strategies[0]}", flush=True)
                
                # Include indicator values for frontend decision markers
                indicators = self._calculate_indicators(session.bars)
                result['indicators'] = {
                    'trend_efficiency': self._calc_trend_efficiency(session.bars),
                    'volatility': self._calc_volatility(session.bars),
                    'atr': indicators.get('atr', [0])[-1] if indicators.get('atr') else 0,
                    'adx': indicators.get('adx', [0])[-1] if indicators.get('adx') else 0
                }
            else:
                result['action'] = 'collecting_regime_data'
                minutes_elapsed = int(elapsed_minutes)
                result['minutes_remaining'] = max(
                    0,
                    session.regime_detection_minutes - minutes_elapsed
                )
                
        elif session.phase == SessionPhase.TRADING:
            # print(f"DEBUG: TRADING Phase. Bar: {timestamp}", flush=True)
            session.bars.append(bar)
            
            # Check for end of day
            if bar_time >= session.market_close or bar_time >= time(15, 55):
                # Close any open position
                if session.active_position:
                    trade = self._close_position(session, bar.close, timestamp, 'end_of_day')
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
        return result
    
    def _detect_regime(self, session: TradingSession) -> Regime:
        """Detect market regime from collected bars."""
        # Combine pre-market and regular bars for analysis
        all_bars = session.bars
        
        if len(all_bars) < 20:
            return Regime.MIXED
        
        # Calculate trend efficiency
        closes = [b.close for b in all_bars[-min(len(all_bars), 60):]]
        
        if len(closes) < 10:
            return Regime.MIXED
        
        # Net move vs total move
        net_move = abs(closes[-1] - closes[0])
        total_move = sum(abs(closes[i] - closes[i-1]) for i in range(1, len(closes)))
        
        if total_move == 0:
            return Regime.MIXED
        
        trend_efficiency = net_move / total_move
        
        # Calculate ADX (using all available bars)
        adx = self._calc_adx(all_bars)
        
        # Calculate volatility
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        volatility = variance ** 0.5
        
        # Check for pre-market gap
        gap_factor = 0
        if session.pre_market_bars and session.bars:
            pre_close = session.pre_market_bars[-1].close
            open_price = session.bars[0].open
            gap_pct = abs(open_price - pre_close) / pre_close * 100
            if gap_pct > 1.0:
                gap_factor = 0.1  # Slight trending bias for gap days
        
        # Classification Logic with ADX
        adjusted_efficiency = trend_efficiency + gap_factor
        
        # 1. Strong Trend: High Efficiency OR High ADX
        if adjusted_efficiency >= 0.6 or adx > 30:
            return Regime.TRENDING
            
        # 2. Choppy: Low ADX
        elif adx < 20:
            return Regime.CHOPPY
            
        # 3. Mixed: Anything else
        else:
            return Regime.MIXED
    
    def _select_strategies(self, session: TradingSession) -> List[str]:
        """Select active strategies for the detected regime and ticker."""
        regime = session.detected_regime or Regime.MIXED
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

        ticker_prefs = self.ticker_preferences.get(ticker, {})
        candidates.extend(ticker_prefs.get(regime, []))
        candidates.extend(self.default_preference.get(regime, []))

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
    
    def _process_trading_bar(
        self, 
        session: TradingSession, 
        bar: BarData, 
        timestamp: datetime
    ) -> Dict[str, Any]:
        """Process a trading bar - manage positions and generate signals."""
        result = {'action': 'trading'}
        current_price = bar.close
        
        # If we have an active position, manage it
        if session.active_position:
            pos = session.active_position
            
            # Update trailing stop
            if pos.trailing_stop_active:
                if pos.side == 'long':
                    if current_price > pos.highest_price:
                        pos.highest_price = current_price
                    new_stop = pos.highest_price * (1 - session.trailing_stop_pct / 100)
                    if new_stop > pos.trailing_stop_price:
                        pos.trailing_stop_price = new_stop
                else:
                    if current_price < pos.lowest_price:
                        pos.lowest_price = current_price
                    new_stop = pos.lowest_price * (1 + session.trailing_stop_pct / 100)
                    if new_stop < pos.trailing_stop_price or pos.trailing_stop_price == 0:
                        pos.trailing_stop_price = new_stop
            
            # Check for exit conditions
            exit_reason = None
            
            # Trailing stop check
            if pos.trailing_stop_active and pos.trailing_stop_price > 0:
                if pos.side == 'long' and bar.low <= pos.trailing_stop_price:
                    exit_reason = 'trailing_stop'
                elif pos.side == 'short' and bar.high >= pos.trailing_stop_price:
                    exit_reason = 'trailing_stop'
            
            # Stop loss check
            if not exit_reason and pos.stop_loss > 0:
                if pos.side == 'long' and bar.low <= pos.stop_loss:
                    exit_reason = 'stop_loss'
                elif pos.side == 'short' and bar.high >= pos.stop_loss:
                    exit_reason = 'stop_loss'
            
            # Take profit check
            if not exit_reason and pos.take_profit > 0:
                if pos.side == 'long' and bar.high >= pos.take_profit:
                    exit_reason = 'take_profit'
                elif pos.side == 'short' and bar.low <= pos.take_profit:
                    exit_reason = 'take_profit'
            
            if exit_reason:
                trade = self._close_position(session, current_price, timestamp, exit_reason)
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
                    'costs': {
                        'slippage': round(trade.slippage, 4),
                        'commission': round(trade.commission, 4),
                        'reg_fee': round(trade.reg_fee, 4),
                        'sec_fee': round(trade.sec_fee, 6),
                        'finra_fee': round(trade.finra_fee, 6),
                        'total': round(trade.total_costs, 4)
                    }
                }
        
        # Custom Rule: Max Daily Loss Circuit Breaker
        current_pnl = sum(t.pnl_dollars for t in session.trades)
        
        if current_pnl < -self.max_daily_loss:
             if session.active_position:
                 trade = self._close_position(session, current_price, timestamp, "max_daily_loss")
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


        if not session.active_position and session.selected_strategy:
            # Check trade limits
            session_key = self._get_session_key(session.run_id, session.ticker, session.date)
            current_bar_index = len(session.bars)
            
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
            
            signal = self._generate_signal(session, bar, timestamp)
            
            if signal:
                session.signals.append(signal)
                result['signal'] = signal.to_dict()
                result['signals'] = [signal.to_dict()]  # Array format for frontend
                
                # Execute signal
                if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                    position = self._open_position(session, signal)
                    self.last_trade_bar_index[session_key] = current_bar_index
                    result['action'] = 'position_opened'
                    result['position'] = position.to_dict()
                    # Include position_opened info for frontend markers
                    result['position_opened'] = {
                        'entry_price': position.entry_price,
                        'side': position.side,
                        'strategy': position.strategy_name,
                        'size': position.size,
                        'stop_loss': position.stop_loss,
                        'take_profit': position.take_profit,
                        'reasoning': signal.reasoning,
                        'confidence': signal.confidence,
                        'metadata': signal.metadata
                    }
        
        return result
    
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

            score = float(sig.confidence) + preference_bonus + perf_bonus
            scored.append((score, sig))
        scored.sort(key=lambda item: item[0], reverse=True)
        best_signal = scored[0][1]
        
        return best_signal
    
    def _calculate_indicators(self, bars: List[BarData]) -> Dict[str, Any]:
        """Calculate indicators from bars."""
        if len(bars) < 5:
            return {}
        
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
        
        return indicators
    
    def _open_position(self, session: TradingSession, signal: Signal) -> Position:
        """Open a new position from signal."""
        side = 'long' if signal.signal_type == SignalType.BUY else 'short'
        capital_usd = max(0.0, session.account_size_usd)
        size = round(capital_usd / signal.price, 4) if signal.price > 0 else 0.0
        if size <= 0:
            size = 0.0
        
        position = Position(
            strategy_name=signal.strategy_name,
            entry_price=signal.price,
            entry_time=signal.timestamp,
            side=side,
            size=size,  # Full notional allocation
            stop_loss=signal.stop_loss or 0,
            take_profit=signal.take_profit or 0,
            trailing_stop_active=signal.trailing_stop,
            highest_price=signal.price if side == 'long' else 0,
            lowest_price=signal.price if side == 'short' else float('inf')
        )
        
        session.active_position = position
        session.trailing_stop_pct = signal.trailing_stop_pct or 0.8
        
        return position
    
    def _close_position(
        self, 
        session: TradingSession, 
        exit_price: float, 
        exit_time: datetime, 
        reason: str
    ) -> DayTrade:
        """Close position and record trade with trading costs."""
        pos = session.active_position
        
        # Calculate gross PnL (before costs)
        if pos.side == 'long':
            gross_pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
        else:
            gross_pnl_pct = (pos.entry_price - exit_price) / pos.entry_price * 100
        
        gross_pnl_dollars = gross_pnl_pct * pos.entry_price / 100 * pos.size
        
        # Calculate trading costs
        costs = self.trading_costs.calculate_costs(
            entry_price=pos.entry_price,
            exit_price=exit_price,
            shares=pos.size,
            side=pos.side
        )
        
        # Net PnL after costs
        net_pnl_dollars = gross_pnl_dollars - costs['total']
        net_pnl_pct = net_pnl_dollars / (pos.entry_price * pos.size) * 100
        
        session.trade_counter += 1
        trade = DayTrade(
            id=session.trade_counter,
            strategy=pos.strategy_name,
            side=pos.side,
            entry_price=pos.entry_price,
            entry_time=pos.entry_time,
            exit_price=exit_price,
            exit_time=exit_time,
            size=pos.size,
            pnl_pct=net_pnl_pct,
            pnl_dollars=net_pnl_dollars,
            exit_reason=reason,
            # Cost breakdown
            slippage=costs.get('slippage', 0.0),
            commission=costs.get('commission', 0.0),
            reg_fee=costs.get('reg_fee', 0.0),
            sec_fee=costs.get('sec_fee', 0.0),
            finra_fee=costs.get('finra_fee', 0.0),
            total_costs=costs.get('total', 0.0),
            gross_pnl_pct=gross_pnl_pct
        )
        
        session.trades.append(trade)
        session.total_pnl += net_pnl_pct
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
            'success': session.total_pnl > 0,
            'trades': [t.to_dict() for t in trades]
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
            self._close_position(session, last_price, last_time, 'manual_close')
        
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
