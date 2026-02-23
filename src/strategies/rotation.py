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
        lookback_period: int = 10,            # Lowered from 20 for faster signal
        rotation_threshold: float = 0.5,      # Lowered from 2.0% (too high for 20 mins)
        volume_lookback: int = 10,
        volume_increase_ratio: float = 1.05,
        volume_stop_pct: float = 0.9,
        min_confidence: float = 50.0,         # Lowered from 55.0
        trailing_stop_pct: float = 1.0
    ):
        super().__init__(
            name="Rotation",
            regimes=[Regime.MIXED, Regime.CHOPPY]
        )
        self.lookback_period = lookback_period
        self.rotation_threshold = rotation_threshold
        self.volume_lookback = volume_lookback
        self.volume_increase_ratio = volume_increase_ratio
        self.volume_stop_pct = volume_stop_pct
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
        rsi = indicators.get('rsi')
        sma = indicators.get('sma')
        ema = indicators.get('ema')
        
        if vwap is None:
            return None
            
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
        
        # Debug logging
        # print(f"DEBUG: Rotation - Perf: {performance:.2f}%, Threshold: {self.rotation_threshold}%")
        
        # VWAP distance as rotation signal
        vwap_distance = (current_price - vwap_val) / vwap_val * 100
        
        # Moving average relationship
        ma_bullish = ema_val > sma_val
        ma_bearish = ema_val < sma_val
        
        # Volume trend
        volume_stats = self.get_volume_stats(volumes, self.volume_lookback)
        avg_volume = volume_stats["avg"]
        current_volume = volume_stats["current"]
        volume_ratio = volume_stats["ratio"]
        effective_volume_stop_pct = self.get_effective_volume_stop_pct() or self.volume_stop_pct
        recent_volume = sum(volumes[-3:]) / 3 if len(volumes) >= 3 else current_volume
        volume_increasing = avg_volume and recent_volume > avg_volume * self.volume_increase_ratio
        
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
                stop_pct = self.volume_adjusted_pct(effective_volume_stop_pct, volume_ratio)
                stop_loss = self.calculate_percent_stop(current_price, stop_pct, 'long')
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
                    trailing_stop_pct=self.get_effective_trailing_stop_pct(),
                    reasoning=" | ".join(reasoning_parts),
                    metadata={
                        'performance': performance,
                        'vwap_distance': vwap_distance,
                        'vwap': vwap_val,
                        'volume_ratio': volume_ratio,
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
                stop_pct = self.volume_adjusted_pct(effective_volume_stop_pct, volume_ratio)
                stop_loss = self.calculate_percent_stop(current_price, stop_pct, 'short')
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
                    trailing_stop_pct=self.get_effective_trailing_stop_pct(),
                    reasoning=" | ".join(reasoning_parts),
                    metadata={
                        'performance': performance,
                        'vwap_distance': vwap_distance,
                        'vwap': vwap_val,
                        'volume_ratio': volume_ratio,
                        'rsi': rsi_val,
                        'regime': regime.value
                    }
                )
        
        if signal:
            signal = self.apply_l2_flow_boost(signal, indicators)
            self.add_signal(signal)
            
        return signal
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'lookback_period': self.lookback_period,
            'rotation_threshold': self.rotation_threshold,
            'volume_lookback': self.volume_lookback,
            'volume_increase_ratio': self.volume_increase_ratio,
            'volume_stop_pct': self.volume_stop_pct,
            'trailing_stop_pct': self.trailing_stop_pct
        })
        return base
