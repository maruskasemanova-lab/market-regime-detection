"""
Rotation Strategy - For MIXED regime.
Rotates between positions based on relative strength.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime

from .base_strategy import BaseStrategy, Signal, SignalType, Regime


class RotationStrategy(BaseStrategy):
    """
    Rotation Strategy for mixed/uncertain markets.
    
    Logic:
    - Monitor relative strength between assets
    - Rotate from weak to strong performers
    - Use balanced position sizing
    - More conservative stops
    
    Best in: MIXED regime
    """
    
    def __init__(
        self,
        lookback_period: int = 20,            # Period for relative strength
        rotation_threshold: float = 2.0,      # Min % diff to trigger rotation
        atr_stop_mult: float = 2.5,           # Wider stop for mixed markets
        min_confidence: float = 55.0,         # Lower bar (mixed is uncertain)
        trailing_stop_pct: float = 1.0        # Wider trailing for chop
    ):
        super().__init__(
            name="Rotation",
            regimes=[Regime.MIXED, Regime.CHOPPY]
        )
        self.lookback_period = lookback_period
        self.rotation_threshold = rotation_threshold
        self.atr_stop_mult = atr_stop_mult
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
        Generate rotation signal based on relative strength and VWAP position.
        In single-asset mode, uses VWAP position as rotation metric.
        """
        if not self.is_allowed_in_regime(regime):
            return None
            
        if len(self.get_open_positions()) > 0:
            return None
        
        # Get indicators
        vwap = indicators.get('vwap')
        atr = indicators.get('atr')
        rsi = indicators.get('rsi')
        sma = indicators.get('sma')
        ema = indicators.get('ema')
        
        if atr is None or vwap is None:
            return None
            
        atr_val = atr[-1] if isinstance(atr, list) else atr
        vwap_val = vwap[-1] if isinstance(vwap, list) else vwap
        rsi_val = rsi[-1] if isinstance(rsi, list) else (rsi or 50)
        sma_val = sma[-1] if isinstance(sma, list) else (sma or current_price)
        ema_val = ema[-1] if isinstance(ema, list) else (ema or current_price)
        
        closes = ohlcv.get('close', [])
        volumes = ohlcv.get('volume', [])
        
        if len(closes) < self.lookback_period:
            return None
        
        # Calculate relative performance metrics
        lookback_close = closes[-self.lookback_period]
        performance = (current_price - lookback_close) / lookback_close * 100
        
        # VWAP distance as rotation signal
        vwap_distance = (current_price - vwap_val) / vwap_val * 100
        
        # Moving average relationship
        ma_bullish = ema_val > sma_val
        ma_bearish = ema_val < sma_val
        
        # Volume trend
        avg_volume = sum(volumes[-10:]) / 10 if len(volumes) >= 10 else sum(volumes) / len(volumes)
        recent_volume = sum(volumes[-3:]) / 3 if len(volumes) >= 3 else volumes[-1]
        volume_increasing = recent_volume > avg_volume
        
        signal = None
        confidence = 50.0
        reasoning_parts = []
        
        # ROTATION TO LONG
        # Conditions: Price recovering toward VWAP from below, MAs turning up
        if (performance > self.rotation_threshold and 
            -1 <= vwap_distance <= 1 and 
            ma_bullish):
            
            reasoning_parts.append(f"Positive rotation ({performance:.1f}% in {self.lookback_period} bars)")
            confidence += 15
            
            reasoning_parts.append("Price at VWAP - potential breakout zone")
            confidence += 10
            
            if ma_bullish:
                confidence += 10
                reasoning_parts.append("EMA > SMA (bullish structure)")
            
            if 45 <= rsi_val <= 65:
                confidence += 10
                reasoning_parts.append(f"RSI balanced ({rsi_val:.1f})")
            
            if volume_increasing:
                confidence += 10
                reasoning_parts.append("Volume building")
            
            if confidence >= self.min_confidence:
                stop_loss = self.calculate_atr_stop(current_price, atr_val, self.atr_stop_mult, 'long')
                take_profit = self.calculate_take_profit(current_price, stop_loss, 2.0, 'long')
                
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
                        'performance': performance,
                        'vwap_distance': vwap_distance,
                        'vwap': vwap_val,
                        'atr': atr_val,
                        'rsi': rsi_val,
                        'regime': regime.value
                    }
                )
        
        # ROTATION TO SHORT
        # Conditions: Price weakening from above VWAP, MAs turning down
        elif (performance < -self.rotation_threshold and 
              -1 <= vwap_distance <= 1 and 
              ma_bearish):
            
            reasoning_parts.append(f"Negative rotation ({performance:.1f}% in {self.lookback_period} bars)")
            confidence += 15
            
            reasoning_parts.append("Price at VWAP - potential breakdown zone")
            confidence += 10
            
            if ma_bearish:
                confidence += 10
                reasoning_parts.append("EMA < SMA (bearish structure)")
            
            if 35 <= rsi_val <= 55:
                confidence += 10
                reasoning_parts.append(f"RSI balanced ({rsi_val:.1f})")
            
            if volume_increasing:
                confidence += 10
                reasoning_parts.append("Volume building")
            
            if confidence >= self.min_confidence:
                stop_loss = self.calculate_atr_stop(current_price, atr_val, self.atr_stop_mult, 'short')
                take_profit = self.calculate_take_profit(current_price, stop_loss, 2.0, 'short')
                
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
                        'performance': performance,
                        'vwap_distance': vwap_distance,
                        'vwap': vwap_val,
                        'atr': atr_val,
                        'rsi': rsi_val,
                        'regime': regime.value
                    }
                )
        
        if signal:
            self.add_signal(signal)
            
        return signal
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'lookback_period': self.lookback_period,
            'rotation_threshold': self.rotation_threshold,
            'atr_stop_mult': self.atr_stop_mult,
            'trailing_stop_pct': self.trailing_stop_pct
        })
        return base
