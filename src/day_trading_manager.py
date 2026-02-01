"""
Session-based Day Trading Manager.
Manages trading sessions for individual days with regime detection and strategy execution.
"""
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Dict, Any, List, Optional
from enum import Enum
import json

from .strategies.base_strategy import BaseStrategy, Signal, SignalType, Position, Regime
from .strategies.trailing_stop import TrailingStopManager, TrailingStopConfig, StopType
from .strategies.mean_reversion import MeanReversionStrategy
from .strategies.pullback import PullbackStrategy
from .strategies.momentum import MomentumStrategy
from .strategies.rotation import RotationStrategy
from .strategies.vwap_magnet import VWAPMagnetStrategy


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
    spread_cost: float = 0.0
    commission: float = 0.0
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
                'spread': round(self.spread_cost, 4),
                'commission': round(self.commission, 4),
                'sec_fee': round(self.sec_fee, 6),
                'finra_fee': round(self.finra_fee, 6),
                'total': round(self.total_costs, 4)
            },
            'gross_pnl_pct': round(self.gross_pnl_pct, 2)
        }


@dataclass
class TradingCosts:
    """Trading costs configuration."""
    spread_pct: float = 0.02  # 0.02% spread (half on entry, half on exit)
    commission_per_trade: float = 1.0  # $1 per trade (each side)
    sec_fee_per_dollar: float = 0.0000278  # SEC fee on sale proceeds
    finra_fee_per_share: float = 0.000145  # FINRA TAF per share (capped at $7.27)
    
    def calculate_costs(
        self,
        entry_price: float,
        exit_price: float,
        shares: float,
        side: str
    ) -> Dict[str, float]:
        """
        Calculate all trading costs.
        Returns dict with spread_cost, commission, sec_fee, finra_fee, total.
        """
        # Spread cost: half at entry, half at exit
        half_spread = self.spread_pct / 2 / 100
        if side == 'long':
            # Buy higher, sell lower
            entry_spread = entry_price * half_spread
            exit_spread = exit_price * half_spread
        else:
            # Sell higher, buy lower (short)
            entry_spread = exit_price * half_spread
            exit_spread = entry_price * half_spread
        
        spread_cost = (entry_spread + exit_spread) * shares
        
        # Commission: $1 per trade (entry + exit = $2 total)
        commission = self.commission_per_trade * 2
        
        # SEC fee: only on sale proceeds
        sale_proceeds = exit_price * shares if side == 'long' else entry_price * shares
        sec_fee = sale_proceeds * self.sec_fee_per_dollar
        
        # FINRA TAF: per share, capped at $7.27
        finra_fee = min(shares * self.finra_fee_per_share, 7.27)
        
        total = spread_cost + commission + sec_fee + finra_fee
        
        return {
            'spread_cost': spread_cost,
            'commission': commission,
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
    selected_strategy: Optional[str] = None
    
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
    
    def __init__(self, regime_detection_minutes: int = 15, trading_costs: TradingCosts = None):
        self.sessions: Dict[str, TradingSession] = {}  # session_key -> TradingSession
        self.regime_detection_minutes = regime_detection_minutes
        self.trading_costs = trading_costs or TradingCosts()
        
        # Pre-configured strategies
        self.strategies = {
            'mean_reversion': MeanReversionStrategy(),
            'pullback': PullbackStrategy(),
            'momentum': MomentumStrategy(),
            'rotation': RotationStrategy(),
            'vwap_magnet': VWAPMagnetStrategy()
        }
        
        # Strategy selection by regime
        self.regime_strategy_preference = {
            Regime.TRENDING: ['pullback', 'momentum', 'vwap_magnet'],
            Regime.CHOPPY: ['mean_reversion', 'vwap_magnet'],
            Regime.MIXED: ['rotation', 'vwap_magnet']
        }
        
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
        
        return self.sessions[key]
    
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
        date = timestamp.strftime('%Y-%m-%d')
        session = self.get_or_create_session(run_id, ticker, date)
        
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
        
        bar_time = timestamp.time()
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
            result['action'] = 'started_regime_detection'
            
        elif session.phase == SessionPhase.REGIME_DETECTION:
            session.bars.append(bar)
            
            # Check if we have enough data for regime detection
            minutes_elapsed = len(session.bars)  # Assuming 1-minute bars
            
            if minutes_elapsed >= session.regime_detection_minutes:
                # Detect regime
                regime = self._detect_regime(session)
                session.detected_regime = regime
                
                # Select best strategy for this regime
                strategy_name = self._select_strategy(session)
                session.selected_strategy = strategy_name
                
                # Transition to trading
                session.phase = SessionPhase.TRADING
                result['action'] = 'regime_detected'
                result['regime'] = regime.value
                result['strategy'] = strategy_name
                
                # Include indicator values for frontend decision markers
                indicators = self._calculate_indicators(session.bars)
                result['indicators'] = {
                    'trend_efficiency': self._calc_trend_efficiency(session.bars),
                    'volatility': self._calc_volatility(session.bars),
                    'atr': indicators.get('atr', [0])[-1] if indicators.get('atr') else 0
                }
            else:
                result['action'] = 'collecting_regime_data'
                result['minutes_remaining'] = session.regime_detection_minutes - minutes_elapsed
                
        elif session.phase == SessionPhase.TRADING:
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
        all_bars = session.pre_market_bars + session.bars
        
        if len(all_bars) < 10:
            return Regime.MIXED
        
        # Calculate trend efficiency
        closes = [b.close for b in all_bars[-min(len(all_bars), 30):]]
        
        if len(closes) < 5:
            return Regime.MIXED
        
        # Net move vs total move
        net_move = abs(closes[-1] - closes[0])
        total_move = sum(abs(closes[i] - closes[i-1]) for i in range(1, len(closes)))
        
        if total_move == 0:
            return Regime.MIXED
        
        trend_efficiency = net_move / total_move
        
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
        
        # Classification
        adjusted_efficiency = trend_efficiency + gap_factor
        
        if adjusted_efficiency >= 0.5:
            return Regime.TRENDING
        elif adjusted_efficiency <= 0.25:
            return Regime.CHOPPY
        else:
            return Regime.MIXED
    
    def _select_strategy(self, session: TradingSession) -> str:
        """Select best strategy for the detected regime."""
        regime = session.detected_regime or Regime.MIXED
        
        # Get preferred strategies for this regime
        preferred = self.regime_strategy_preference.get(regime, ['vwap_magnet'])
        
        # For now, return the first preferred strategy
        # Could be enhanced with historical performance analysis
        return preferred[0]
    
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
                # Include position_closed info for frontend markers
                result['position_closed'] = {
                    'exit_price': trade.exit_price,
                    'side': trade.side,
                    'exit_reason': exit_reason,
                    'pnl_pct': trade.pnl_pct,
                    'pnl_dollars': trade.pnl_dollars,
                    'strategy': trade.strategy
                }
        
        # If no position, look for entry signals
        if not session.active_position and session.selected_strategy:
            signal = self._generate_signal(session, bar, timestamp)
            
            if signal:
                session.signals.append(signal)
                result['signal'] = signal.to_dict()
                result['signals'] = [signal.to_dict()]  # Array format for frontend
                
                # Execute signal
                if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                    position = self._open_position(session, signal)
                    result['action'] = 'position_opened'
                    result['position'] = position.to_dict()
                    # Include position_opened info for frontend markers
                    result['position_opened'] = {
                        'entry_price': position.entry_price,
                        'side': position.side,
                        'strategy': position.strategy_name,
                        'size': position.size,
                        'stop_loss': position.stop_loss,
                        'take_profit': position.take_profit
                    }
        
        return result
    
    def _generate_signal(
        self, 
        session: TradingSession, 
        bar: BarData, 
        timestamp: datetime
    ) -> Optional[Signal]:
        """Generate trading signal using selected strategy."""
        strategy_name = session.selected_strategy
        
        if not strategy_name or strategy_name not in self.strategies:
            return None
        
        strategy = self.strategies[strategy_name]
        
        # Prepare OHLCV data
        bars = session.bars[-100:] if len(session.bars) >= 100 else session.bars
        ohlcv = {
            'open': [b.open for b in bars],
            'high': [b.high for b in bars],
            'low': [b.low for b in bars],
            'close': [b.close for b in bars],
            'volume': [b.volume for b in bars]
        }
        
        # Calculate basic indicators
        indicators = self._calculate_indicators(bars)
        
        # Generate signal
        signal = strategy.generate_signal(
            current_price=bar.close,
            ohlcv=ohlcv,
            indicators=indicators,
            regime=session.detected_regime or Regime.MIXED,
            timestamp=timestamp
        )
        
        return signal
    
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
        
        return indicators
    
    def _open_position(self, session: TradingSession, signal: Signal) -> Position:
        """Open a new position from signal."""
        side = 'long' if signal.signal_type == SignalType.BUY else 'short'
        
        position = Position(
            strategy_name=signal.strategy_name,
            entry_price=signal.price,
            entry_time=signal.timestamp,
            side=side,
            size=1.0,
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
            spread_cost=costs['spread_cost'],
            commission=costs['commission'],
            sec_fee=costs['sec_fee'],
            finra_fee=costs['finra_fee'],
            total_costs=costs['total'],
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
