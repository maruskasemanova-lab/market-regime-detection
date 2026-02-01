"""
Pullback Strategy - For TRENDING regime.
Waits for pullback to support/EMA in an established trend.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime

from .base_strategy import BaseStrategy, Signal, SignalType, Regime


class PullbackStrategy(BaseStrategy):
    """
    Pullback Strategy for trending markets.
    
    Logic:
    - Identify trend direction using EMA/SMA crossover
    - Wait for pullback to EMA or VWAP
    - Enter on bounce with volume confirmation
    - Trail stop with trend
    
    Best in: TRENDING regime
    """
    
    def __init__(
        self,
        pullback_threshold_pct: float = 0.5,  # Min pullback to trigger
        ma_fast_period: int = 10,             # Fast MA period
        ma_slow_period: int = 20,             # Slow MA period
        atr_stop_mult: float = 2.0,           # Stop multiplier
        rr_ratio: float = 2.5,                # Risk:Reward ratio
        min_confidence: float = 65.0,         # Min confidence
        trailing_stop_pct: float = 0.8        # Trailing stop pct
    ):
        super().__init__(
            name="Pullback",
            regimes=[Regime.TRENDING]
        )
        self.pullback_threshold_pct = pullback_threshold_pct
        self.ma_fast_period = ma_fast_period
        self.ma_slow_period = ma_slow_period
        self.atr_stop_mult = atr_stop_mult
        self.rr_ratio = rr_ratio
        self.min_confidence = min_confidence
        self.trailing_stop_pct = trailing_stop_pct
        
    def generate_signal(
        self,
        current_price: float,
        ohlcv: Dict[str, List[float]],
        indicators: Dict[str, Any],
        regime: Regime,
        timestamp: datetime
    ) -> Optional[Signal]:
        """
        Generate pullback signal.
        
        Entry conditions:
        - Trend established (EMA fast > EMA slow for uptrend)
        - Price pulls back toward EMA/VWAP
        - Price shows bounce (current bar closing in trend direction)
        - Volume picks up on bounce
        """
        if not self.is_allowed_in_regime(regime):
            return None
            
        if len(self.get_open_positions()) > 0:
            return None
        
        # Get indicators
        ema_fast = indicators.get('ema_fast') or indicators.get('ema')
        ema_slow = indicators.get('ema_slow') or indicators.get('sma')
        vwap = indicators.get('vwap')
        atr = indicators.get('atr')
        rsi = indicators.get('rsi')
        
        if ema_fast is None or ema_slow is None or atr is None:
            return None
            
        # Get latest values
        ema_f = ema_fast[-1] if isinstance(ema_fast, list) else ema_fast
        ema_s = ema_slow[-1] if isinstance(ema_slow, list) else ema_slow
        vwap_val = vwap[-1] if isinstance(vwap, list) else (vwap or current_price)
        atr_val = atr[-1] if isinstance(atr, list) else atr
        rsi_val = rsi[-1] if isinstance(rsi, list) else (rsi or 50)
        
        # Get OHLC
        opens = ohlcv.get('open', [])
        highs = ohlcv.get('high', [])
        lows = ohlcv.get('low', [])
        closes = ohlcv.get('close', [])
        volumes = ohlcv.get('volume', [])
        
        if len(closes) < 5:
            return None
            
        # Calculate average volume
        avg_volume = sum(volumes[-20:]) / min(20, len(volumes)) if volumes else 0
        current_volume = volumes[-1] if volumes else 0
        
        signal = None
        confidence = 50.0
        reasoning_parts = []
        
        # Determine trend
        uptrend = ema_f > ema_s
        downtrend = ema_f < ema_s
        
        # Calculate recent swing high/low
        recent_high = max(highs[-10:]) if len(highs) >= 10 else max(highs)
        recent_low = min(lows[-10:]) if len(lows) >= 10 else min(lows)
        
        # LONG SIGNAL: Uptrend with pullback
        if uptrend:
            reasoning_parts.append(f"Uptrend (EMA{self.ma_fast_period} > EMA{self.ma_slow_period})")
            
            # Check for pullback to EMA or VWAP
            pullback_target = max(ema_f, vwap_val)
            distance_to_target = (current_price - pullback_target) / pullback_target * 100
            
            # Want price near or just above the pullback target
            if -self.pullback_threshold_pct <= distance_to_target <= self.pullback_threshold_pct * 2:
                confidence += 15
                reasoning_parts.append(f"At pullback zone ({distance_to_target:.2f}% from EMA/VWAP)")
                
                # Check for bounce - current close > open
                if len(closes) >= 2 and closes[-1] > opens[-1]:
                    confidence += 15
                    reasoning_parts.append("Bullish bounce candle")
                    
                    # Check if bouncing off low
                    if lows[-1] <= pullback_target <= closes[-1]:
                        confidence += 10
                        reasoning_parts.append("Bounce confirmed from support")
                
                # RSI not overbought
                if 40 <= rsi_val <= 60:
                    confidence += 10
                    reasoning_parts.append(f"RSI neutral ({rsi_val:.1f})")
                elif rsi_val < 40:
                    confidence += 5
                    reasoning_parts.append(f"RSI weak but still uptrend")
                
                # Volume confirmation
                if current_volume > avg_volume:
                    confidence += 10
                    reasoning_parts.append("Volume increasing on bounce")
                
                if confidence >= self.min_confidence:
                    stop_loss = min(recent_low, current_price - atr_val * self.atr_stop_mult)
                    take_profit = self.calculate_take_profit(current_price, stop_loss, self.rr_ratio, 'long')
                    
                    signal = Signal(
                        strategy_name=self.name,
                        signal_type=SignalType.BUY,
                        price=current_price,
                        timestamp=timestamp,
                        confidence=min(confidence, 100),
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        trailing_stop=True,
                        trailing_stop_pct=self.trailing_stop_pct,
                        reasoning=" | ".join(reasoning_parts),
                        metadata={
                            'ema_fast': ema_f,
                            'ema_slow': ema_s,
                            'vwap': vwap_val,
                            'atr': atr_val,
                            'rsi': rsi_val,
                            'recent_low': recent_low,
                            'regime': regime.value
                        }
                    )
        
        # SHORT SIGNAL: Downtrend with pullback
        elif downtrend:
            reasoning_parts.append(f"Downtrend (EMA{self.ma_fast_period} < EMA{self.ma_slow_period})")
            
            pullback_target = min(ema_f, vwap_val)
            distance_to_target = (current_price - pullback_target) / pullback_target * 100
            
            # Want price near or just below the pullback target
            if -self.pullback_threshold_pct * 2 <= distance_to_target <= self.pullback_threshold_pct:
                confidence += 15
                reasoning_parts.append(f"At pullback zone ({distance_to_target:.2f}% from EMA/VWAP)")
                
                # Check for rejection - current close < open
                if len(closes) >= 2 and closes[-1] < opens[-1]:
                    confidence += 15
                    reasoning_parts.append("Bearish rejection candle")
                    
                    if highs[-1] >= pullback_target >= closes[-1]:
                        confidence += 10
                        reasoning_parts.append("Rejection confirmed from resistance")
                
                # RSI not oversold
                if 40 <= rsi_val <= 60:
                    confidence += 10
                    reasoning_parts.append(f"RSI neutral ({rsi_val:.1f})")
                elif rsi_val > 60:
                    confidence += 5
                    reasoning_parts.append(f"RSI strong but still downtrend")
                
                if current_volume > avg_volume:
                    confidence += 10
                    reasoning_parts.append("Volume increasing on rejection")
                
                if confidence >= self.min_confidence:
                    stop_loss = max(recent_high, current_price + atr_val * self.atr_stop_mult)
                    take_profit = self.calculate_take_profit(current_price, stop_loss, self.rr_ratio, 'short')
                    
                    signal = Signal(
                        strategy_name=self.name,
                        signal_type=SignalType.SELL,
                        price=current_price,
                        timestamp=timestamp,
                        confidence=min(confidence, 100),
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        trailing_stop=True,
                        trailing_stop_pct=self.trailing_stop_pct,
                        reasoning=" | ".join(reasoning_parts),
                        metadata={
                            'ema_fast': ema_f,
                            'ema_slow': ema_s,
                            'vwap': vwap_val,
                            'atr': atr_val,
                            'rsi': rsi_val,
                            'recent_high': recent_high,
                            'regime': regime.value
                        }
                    )
        
        if signal:
            self.add_signal(signal)
            
        return signal
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'pullback_threshold_pct': self.pullback_threshold_pct,
            'ma_fast_period': self.ma_fast_period,
            'ma_slow_period': self.ma_slow_period,
            'atr_stop_mult': self.atr_stop_mult,
            'rr_ratio': self.rr_ratio,
            'trailing_stop_pct': self.trailing_stop_pct
        })
        return base
