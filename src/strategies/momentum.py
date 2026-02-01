"""
Momentum Strategy - For TRENDING regime.
Trades breakouts from consolidation with strong volume.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime

from .base_strategy import BaseStrategy, Signal, SignalType, Regime


class MomentumStrategy(BaseStrategy):
    """
    Momentum/Breakout Strategy for trending markets.
    
    Logic:
    - Detect consolidation (low volatility, narrow range)
    - Enter on breakout with high volume
    - Aggressive trailing stop
    - Pyramiding supported
    
    Best in: TRENDING regime
    """
    
    def __init__(
        self,
        consolidation_bars: int = 5,          # Bars to check for consolidation
        breakout_atr_mult: float = 0.5,       # Breakout threshold as ATR multiple
        volume_threshold: float = 1.5,        # Volume must be X times average
        atr_stop_mult: float = 1.5,           # Tight stop
        rr_ratio: float = 3.0,                # High R:R
        min_confidence: float = 70.0,         # High bar for entry
        trailing_stop_pct: float = 0.6        # Aggressive trailing
    ):
        super().__init__(
            name="Momentum",
            regimes=[Regime.TRENDING]
        )
        self.consolidation_bars = consolidation_bars
        self.breakout_atr_mult = breakout_atr_mult
        self.volume_threshold = volume_threshold
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
        Generate momentum/breakout signal.
        
        Entry conditions:
        - Price breaks above/below consolidation range
        - Volume confirms the breakout
        - RSI confirms momentum direction
        """
        if not self.is_allowed_in_regime(regime):
            return None
            
        if len(self.get_open_positions()) >= 2:  # Allow pyramiding up to 2
            return None
        
        # Get indicators
        atr = indicators.get('atr')
        rsi = indicators.get('rsi')
        vwap = indicators.get('vwap')
        ema = indicators.get('ema') or indicators.get('ema_fast')
        
        if atr is None:
            return None
            
        atr_val = atr[-1] if isinstance(atr, list) else atr
        rsi_val = rsi[-1] if isinstance(rsi, list) else (rsi or 50)
        vwap_val = vwap[-1] if isinstance(vwap, list) else (vwap or current_price)
        ema_val = ema[-1] if isinstance(ema, list) else (ema or current_price)
        
        # Get OHLC
        opens = ohlcv.get('open', [])
        highs = ohlcv.get('high', [])
        lows = ohlcv.get('low', [])
        closes = ohlcv.get('close', [])
        volumes = ohlcv.get('volume', [])
        
        if len(closes) < self.consolidation_bars + 2:
            return None
        
        # Calculate consolidation range (excluding current bar)
        consol_highs = highs[-(self.consolidation_bars + 1):-1]
        consol_lows = lows[-(self.consolidation_bars + 1):-1]
        consol_high = max(consol_highs)
        consol_low = min(consol_lows)
        consol_range = consol_high - consol_low
        
        # Check if range is "tight" (consolidation)
        is_consolidation = consol_range < atr_val * 2  # Range less than 2x ATR
        
        # Volume analysis
        avg_volume = sum(volumes[-20:-1]) / min(19, len(volumes) - 1) if len(volumes) > 1 else 0
        current_volume = volumes[-1] if volumes else 0
        volume_surge = current_volume >= avg_volume * self.volume_threshold
        
        signal = None
        confidence = 50.0
        reasoning_parts = []
        
        # LONG BREAKOUT
        if current_price > consol_high + (atr_val * self.breakout_atr_mult):
            reasoning_parts.append(f"Breakout above {consol_high:.2f}")
            confidence += 15
            
            if is_consolidation:
                confidence += 15
                reasoning_parts.append(f"Breaking tight consolidation (range: {consol_range:.2f})")
            
            if volume_surge:
                confidence += 20
                reasoning_parts.append(f"Volume surge ({current_volume/avg_volume:.1f}x avg)")
            
            if rsi_val > 50:
                confidence += 10
                reasoning_parts.append(f"RSI confirms momentum ({rsi_val:.1f})")
            
            if current_price > vwap_val:
                confidence += 5
                reasoning_parts.append("Above VWAP")
            
            if current_price > ema_val:
                confidence += 5
                reasoning_parts.append("Above EMA")
            
            # Check if breakout bar is strong (close near high)
            bar_body = closes[-1] - opens[-1]
            bar_range = highs[-1] - lows[-1]
            if bar_range > 0 and bar_body / bar_range > 0.6:
                confidence += 10
                reasoning_parts.append("Strong breakout candle")
            
            if confidence >= self.min_confidence:
                stop_loss = max(consol_low, current_price - atr_val * self.atr_stop_mult)
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
                        'consol_high': consol_high,
                        'consol_low': consol_low,
                        'consol_range': consol_range,
                        'atr': atr_val,
                        'volume_ratio': current_volume / avg_volume if avg_volume else 0,
                        'rsi': rsi_val,
                        'regime': regime.value
                    }
                )
        
        # SHORT BREAKDOWN
        elif current_price < consol_low - (atr_val * self.breakout_atr_mult):
            reasoning_parts.append(f"Breakdown below {consol_low:.2f}")
            confidence += 15
            
            if is_consolidation:
                confidence += 15
                reasoning_parts.append(f"Breaking tight consolidation (range: {consol_range:.2f})")
            
            if volume_surge:
                confidence += 20
                reasoning_parts.append(f"Volume surge ({current_volume/avg_volume:.1f}x avg)")
            
            if rsi_val < 50:
                confidence += 10
                reasoning_parts.append(f"RSI confirms momentum ({rsi_val:.1f})")
            
            if current_price < vwap_val:
                confidence += 5
                reasoning_parts.append("Below VWAP")
            
            if current_price < ema_val:
                confidence += 5
                reasoning_parts.append("Below EMA")
            
            bar_body = opens[-1] - closes[-1]
            bar_range = highs[-1] - lows[-1]
            if bar_range > 0 and bar_body / bar_range > 0.6:
                confidence += 10
                reasoning_parts.append("Strong breakdown candle")
            
            if confidence >= self.min_confidence:
                stop_loss = min(consol_high, current_price + atr_val * self.atr_stop_mult)
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
                        'consol_high': consol_high,
                        'consol_low': consol_low,
                        'consol_range': consol_range,
                        'atr': atr_val,
                        'volume_ratio': current_volume / avg_volume if avg_volume else 0,
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
            'consolidation_bars': self.consolidation_bars,
            'breakout_atr_mult': self.breakout_atr_mult,
            'volume_threshold': self.volume_threshold,
            'atr_stop_mult': self.atr_stop_mult,
            'rr_ratio': self.rr_ratio,
            'trailing_stop_pct': self.trailing_stop_pct
        })
        return base
