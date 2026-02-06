"""
Strategy Engine - Connects regime detection with trading strategies.
"""
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

from .strategies.base_strategy import BaseStrategy, Signal, SignalType, Position, Regime
from .strategies.trailing_stop import TrailingStopManager, TrailingStopConfig, StopType
from .strategies.mean_reversion import MeanReversionStrategy
from .strategies.pullback import PullbackStrategy
from .strategies.momentum import MomentumStrategy
from .strategies.rotation import RotationStrategy
from .strategies.vwap_magnet import VWAPMagnetStrategy
from .strategies.volume_profile import VolumeProfileStrategy
from .strategies.gap_liquidity import GapLiquidityStrategy
from .strategies.absorption_reversal import AbsorptionReversalStrategy
from .strategies.momentum_flow import MomentumFlowStrategy
from .strategies.exhaustion_fade import ExhaustionFadeStrategy


@dataclass
class Trade:
    """Completed trade record."""
    id: int
    strategy_name: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    side: str
    size: float
    pnl_pct: float
    pnl_absolute: float
    exit_reason: str  # 'stop_loss', 'take_profit', 'trailing_stop', 'signal'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'strategy': self.strategy_name,
            'entry_price': round(self.entry_price, 2),
            'exit_price': round(self.exit_price, 2),
            'entry_time': self.entry_time.isoformat() if isinstance(self.entry_time, datetime) else str(self.entry_time),
            'exit_time': self.exit_time.isoformat() if isinstance(self.exit_time, datetime) else str(self.exit_time),
            'side': self.side,
            'size': self.size,
            'pnl_pct': round(self.pnl_pct, 2),
            'pnl_absolute': round(self.pnl_absolute, 2),
            'exit_reason': self.exit_reason
        }


class StrategyEngine:
    """
    Main engine that orchestrates regime detection and strategy execution.
    """
    
    def __init__(self, backtest_api_url: str = "http://localhost:8000"):
        self.api_url = backtest_api_url
        
        # Initialize all strategies
        self.strategies: Dict[str, BaseStrategy] = {
            'mean_reversion': MeanReversionStrategy(),
            'pullback': PullbackStrategy(),
            'momentum': MomentumStrategy(),
            'rotation': RotationStrategy(),
            'vwap_magnet': VWAPMagnetStrategy(),
            'volume_profile': VolumeProfileStrategy(),
            'gap_liquidity': GapLiquidityStrategy(),
            'absorption_reversal': AbsorptionReversalStrategy(),
            'momentum_flow': MomentumFlowStrategy(),
            'exhaustion_fade': ExhaustionFadeStrategy(),
        }
        
        # Trailing stop manager
        self.trailing_stop_manager = TrailingStopManager(
            TrailingStopConfig(
                stop_type=StopType.PERCENT,
                initial_stop_pct=2.0,
                trailing_pct=0.8
            )
        )
        
        # State
        self.current_regime: Regime = Regime.MIXED
        self.all_signals: List[Signal] = []
        self.all_trades: List[Trade] = []
        self.trade_counter = 0
        self.open_positions: Dict[str, Position] = {}  # strategy_name -> Position
        
        # Cache
        self._cached_data: Dict[str, Any] = {}
        
    def fetch_current_data(self) -> Dict[str, Any]:
        """Fetch current price and bar info from backtest API."""
        try:
            resp = requests.get(f"{self.api_url}/api/current", timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"Error fetching current data: {e}")
            return {}
    
    def fetch_history(self, bars: int = 100) -> Dict[str, List[float]]:
        """Fetch historical OHLCV data."""
        try:
            resp = requests.get(f"{self.api_url}/api/history/ohlcv?bars={bars}", timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"Error fetching history: {e}")
            return {}
    
    def fetch_indicator(self, name: str, period: int = 20) -> Optional[List[float]]:
        """Fetch indicator values."""
        try:
            resp = requests.get(f"{self.api_url}/api/indicator/{name}?period={period}", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            return data.get('values', data.get('value', []))
        except Exception as e:
            print(f"Error fetching indicator {name}: {e}")
            return None
    
    def fetch_all_indicators(self) -> Dict[str, Any]:
        """Fetch all required indicators."""
        indicators = {}
        
        # SMA
        sma = self.fetch_indicator('sma', 20)
        if sma:
            indicators['sma'] = sma
            
        # EMA (fast and slow)
        ema = self.fetch_indicator('ema', 10)
        if ema:
            indicators['ema'] = ema
            indicators['ema_fast'] = ema
            
        ema_slow = self.fetch_indicator('ema', 20)
        if ema_slow:
            indicators['ema_slow'] = ema_slow
        
        # RSI
        rsi = self.fetch_indicator('rsi', 14)
        if rsi:
            indicators['rsi'] = rsi
            
        # ATR
        atr = self.fetch_indicator('atr', 14)
        if atr:
            indicators['atr'] = atr
            
        # VWAP
        vwap = self.fetch_indicator('vwap')
        if vwap:
            indicators['vwap'] = vwap
            
        return indicators
    
    def detect_regime(self, ohlcv: Dict[str, List[float]], indicators: Dict[str, Any]) -> Regime:
        """
        Simple regime detection based on volatility and trend efficiency.
        Uses the existing models logic simplified.
        """
        closes = ohlcv.get('close', [])
        highs = ohlcv.get('high', [])
        lows = ohlcv.get('low', [])
        
        if len(closes) < 20:
            return Regime.MIXED
        
        # Calculate trend efficiency
        # |Close[-1] - Close[-20]| / sum(|Close[i] - Close[i-1]|)
        net_move = abs(closes[-1] - closes[-20])
        total_move = sum(abs(closes[i] - closes[i-1]) for i in range(-19, 0))
        
        if total_move == 0:
            return Regime.MIXED
            
        trend_efficiency = net_move / total_move
        
        # Calculate volatility (using ATR or range)
        atr = indicators.get('atr', [])
        if isinstance(atr, list) and len(atr) > 0:
            current_atr = atr[-1]
            avg_price = sum(closes[-20:]) / 20
            atr_pct = current_atr / avg_price * 100
        else:
            ranges = [highs[i] - lows[i] for i in range(-20, 0)]
            avg_range = sum(ranges) / len(ranges)
            avg_price = sum(closes[-20:]) / 20
            atr_pct = avg_range / avg_price * 100
        
        # Classification
        if trend_efficiency >= 0.5:
            return Regime.TRENDING
        elif trend_efficiency <= 0.25:
            return Regime.CHOPPY
        else:
            return Regime.MIXED
    
    def get_active_strategies(self, regime: Regime) -> List[BaseStrategy]:
        """Get strategies that are allowed in the current regime."""
        active = []
        for strategy in self.strategies.values():
            if strategy.enabled and strategy.is_allowed_in_regime(regime):
                active.append(strategy)
        return active
    
    def process_signals(self, current_price: float, timestamp: datetime) -> List[Signal]:
        """Generate signals from all active strategies."""
        signals = []
        
        # Fetch data
        ohlcv = self.fetch_history(100)
        indicators = self.fetch_all_indicators()
        
        if not ohlcv or 'close' not in ohlcv:
            return signals
        
        # Detect regime
        self.current_regime = self.detect_regime(ohlcv, indicators)
        
        # Get active strategies for this regime
        active_strategies = self.get_active_strategies(self.current_regime)
        
        # Generate signals from each strategy
        for strategy in active_strategies:
            signal = strategy.generate_signal(
                current_price=current_price,
                ohlcv=ohlcv,
                indicators=indicators,
                regime=self.current_regime,
                timestamp=timestamp
            )
            
            if signal:
                signals.append(signal)
                self.all_signals.append(signal)
        
        return signals
    
    def manage_positions(self, current_price: float, timestamp: datetime) -> List[Trade]:
        """Update open positions and check for exits."""
        closed_trades = []
        
        for strategy_name, position in list(self.open_positions.items()):
            if position.status != 'open':
                continue
            
            # Update trailing stop if active
            if position.trailing_stop_active:
                strategy = self.strategies.get(strategy_name)
                if strategy:
                    trailing_pct = 0.8  # Default
                    last_signal = strategy.get_last_signal()
                    if last_signal and last_signal.trailing_stop_pct:
                        trailing_pct = last_signal.trailing_stop_pct
                    
                    position.update_trailing_stop(current_price, trailing_pct)
            
            # Check exit conditions
            exit_reason = None
            
            if position.check_stop_hit(current_price):
                if position.trailing_stop_active and position.trailing_stop_price > 0:
                    if position.side == 'long' and current_price <= position.trailing_stop_price:
                        exit_reason = 'trailing_stop'
                    elif position.side == 'short' and current_price >= position.trailing_stop_price:
                        exit_reason = 'trailing_stop'
                if not exit_reason:
                    exit_reason = 'stop_loss'
                    
            elif position.check_take_profit_hit(current_price):
                exit_reason = 'take_profit'
            
            if exit_reason:
                # Close position
                position.status = 'closed'
                position.calculate_pnl(current_price)
                
                self.trade_counter += 1
                trade = Trade(
                    id=self.trade_counter,
                    strategy_name=strategy_name,
                    entry_price=position.entry_price,
                    exit_price=current_price,
                    entry_time=position.entry_time,
                    exit_time=timestamp,
                    side=position.side,
                    size=position.size,
                    pnl_pct=position.pnl,
                    pnl_absolute=position.pnl * position.entry_price / 100,
                    exit_reason=exit_reason
                )
                
                self.all_trades.append(trade)
                closed_trades.append(trade)
                del self.open_positions[strategy_name]
        
        return closed_trades
    
    def execute_signal(self, signal: Signal, timestamp: datetime) -> Optional[Position]:
        """Execute a signal by opening a position."""
        strategy_name = signal.strategy_name.lower().replace(' ', '_')
        
        # Check if we already have a position for this strategy
        if strategy_name in self.open_positions:
            return None
        
        # Open position
        side = 'long' if signal.signal_type == SignalType.BUY else 'short'
        
        position = Position(
            strategy_name=signal.strategy_name,
            entry_price=signal.price,
            entry_time=timestamp,
            side=side,
            stop_loss=signal.stop_loss or 0,
            take_profit=signal.take_profit or 0,
            trailing_stop_active=signal.trailing_stop,
            highest_price=signal.price if side == 'long' else 0,
            lowest_price=signal.price if side == 'short' else float('inf')
        )
        
        # Use trailing stop manager for initial stop if trailing is active
        if signal.trailing_stop and signal.trailing_stop_pct:
            self.trailing_stop_manager.config.trailing_pct = signal.trailing_stop_pct
        
        self.open_positions[strategy_name] = position
        
        return position
    
    def step(self) -> Dict[str, Any]:
        """
        Execute one step of the backtest.
        Returns state and any signals/trades.
        """
        # Step the backtest forward
        try:
            requests.post(f"{self.api_url}/api/backtest/step", timeout=5)
        except Exception as e:
            print(f"Error stepping backtest: {e}")
        
        # Get current state
        current_data = self.fetch_current_data()
        if not current_data:
            return {'error': 'Failed to fetch current data'}
        
        current_price = current_data.get('current_price', 0)
        bar_index = current_data.get('bar', 0)
        ts = current_data.get('timestamp', datetime.now().isoformat())
        
        try:
            timestamp = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        except:
            timestamp = datetime.now()
        
        # Manage existing positions
        closed_trades = self.manage_positions(current_price, timestamp)
        
        # Generate new signals
        signals = self.process_signals(current_price, timestamp)
        
        # Execute signals (open positions)
        new_positions = []
        for signal in signals:
            if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                pos = self.execute_signal(signal, timestamp)
                if pos:
                    new_positions.append(pos)
        
        return {
            'bar': bar_index,
            'timestamp': ts,
            'price': current_price,
            'regime': self.current_regime.value,
            'signals': [s.to_dict() for s in signals],
            'new_positions': [p.to_dict() for p in new_positions],
            'closed_trades': [t.to_dict() for t in closed_trades],
            'open_positions': {k: v.to_dict() for k, v in self.open_positions.items()}
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current engine state."""
        return {
            'regime': self.current_regime.value,
            'strategies': {name: s.to_dict() for name, s in self.strategies.items()},
            'active_strategies': [s.name for s in self.get_active_strategies(self.current_regime)],
            'open_positions': {k: v.to_dict() for k, v in self.open_positions.items()},
            'total_signals': len(self.all_signals),
            'total_trades': len(self.all_trades),
            'recent_signals': [s.to_dict() for s in self.all_signals[-10:]],
            'recent_trades': [t.to_dict() for t in self.all_trades[-10:]]
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if not self.all_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl_pct': 0,
                'avg_pnl_pct': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'by_strategy': {}
            }
        
        winning = [t for t in self.all_trades if t.pnl_pct > 0]
        losing = [t for t in self.all_trades if t.pnl_pct <= 0]
        
        total_pnl = sum(t.pnl_pct for t in self.all_trades)
        
        # By strategy
        by_strategy = {}
        for trade in self.all_trades:
            strat = trade.strategy_name
            if strat not in by_strategy:
                by_strategy[strat] = {'trades': 0, 'wins': 0, 'pnl': 0}
            by_strategy[strat]['trades'] += 1
            by_strategy[strat]['pnl'] += trade.pnl_pct
            if trade.pnl_pct > 0:
                by_strategy[strat]['wins'] += 1
        
        for strat in by_strategy:
            trades = by_strategy[strat]['trades']
            by_strategy[strat]['win_rate'] = by_strategy[strat]['wins'] / trades * 100 if trades else 0
            by_strategy[strat]['avg_pnl'] = by_strategy[strat]['pnl'] / trades if trades else 0
        
        return {
            'total_trades': len(self.all_trades),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': len(winning) / len(self.all_trades) * 100,
            'total_pnl_pct': round(total_pnl, 2),
            'avg_pnl_pct': round(total_pnl / len(self.all_trades), 2),
            'best_trade': round(max(t.pnl_pct for t in self.all_trades), 2),
            'worst_trade': round(min(t.pnl_pct for t in self.all_trades), 2),
            'by_strategy': by_strategy
        }
