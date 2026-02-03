"""
Pullback Strategy - For TRENDING regime.
Trades pullbacks to EMA/VWAP in established trends.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime

from .base_strategy import BaseStrategy, Signal, SignalType, Regime


class PullbackStrategy(BaseStrategy):
    """
    Pullback Strategy for trending markets.
    
    Logic:
    - Identify strong trend (ADX > 25, Price vs VWAP, EMA align)
    - Wait for pullback to EMA/VWAP zone
    - Enter on CONFIRMED bounce (break of previous candle high)
    
    Best in: TRENDING regime
    """
    
    def __init__(
        self,
        pullback_threshold_pct: float = 0.5,  # Tighter zone relative to slower MA
        ma_fast_period: int = 50,             # ~10 EMA on 5m chart
        ma_slow_period: int = 100,            # ~20 EMA on 5m chart
        atr_stop_mult: float = 3.0,           # Wider stop for noise
        rr_ratio: float = 1.5,                # Lower R:R target (high win rate focus)
        min_confidence: float = 65.0,         # Standard confidence (was 70.0)
        trailing_stop_pct: float = 1.0        # Looser trail
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
        """
        if not self.is_allowed_in_regime(regime):
            return None
            
        if len(self.get_open_positions()) > 0:
            return None
        
        # Get indicators
        ema_f = indicators.get('ema') or indicators.get('ema_fast')
        ema_s = indicators.get('ema_slow')
        vwap = indicators.get('vwap')
        atr = indicators.get('atr')
        rsi = indicators.get('rsi')
        adx = indicators.get('adx')
        
        if ema_f is None or ema_s is None or atr is None:
            return None
            
        ema_f = ema_f[-1] if isinstance(ema_f, list) else (ema_f or current_price)
        ema_s = ema_s[-1] if isinstance(ema_s, list) else (ema_s or current_price)
        vwap_val = vwap[-1] if isinstance(vwap, list) else (vwap or current_price)
        atr_val = atr[-1] if isinstance(atr, list) else atr
        rsi_val = rsi[-1] if isinstance(rsi, list) else (rsi or 50)
        adx_val = adx[-1] if isinstance(adx, list) else (adx or 0)
        
        # Get OHLC
        opens = ohlcv.get('open', [])
        highs = ohlcv.get('high', [])
        lows = ohlcv.get('low', [])
        closes = ohlcv.get('close', [])
        volumes = ohlcv.get('volume', [])
        
        if len(closes) < 20:
            return None
            
        avg_volume = sum(volumes[-20:]) / min(20, len(volumes)) if volumes else 0
        current_volume = volumes[-1] if volumes else 0
        
        signal = None
        confidence = 50.0
        reasoning_parts = []
        
        # Determine trend (Relaxed ADX)
        uptrend = ema_f > ema_s and current_price > vwap_val and adx_val > 20
        downtrend = ema_f < ema_s and current_price < vwap_val and adx_val > 20
        
        # Calculate recent swing high/low
        recent_high = max(highs[-10:]) if len(highs) >= 10 else max(highs)
        recent_low = min(lows[-10:]) if len(lows) >= 10 else min(lows)
        
        # LONG SIGNAL: Uptrend + Pullback + Bounce
        if uptrend:
            reasoning_parts.append(f"Uptrend (ADX {adx_val:.1f})")
            
            # Check if we are in a pullback (Price < EMA Fast)
            # OR if we are just finding support at EMA Slow/VWAP
            if current_price < ema_f or current_price < ema_s:
                confidence += 15
                reasoning_parts.append("In pullback zone")
                
                # Check for bounce: Green candle
                if closes[-1] > opens[-1]:
                    # Quality Check: Strong close (upper half) OR Volume surge
                    midpoint = (highs[-1] + lows[-1]) / 2
                    is_strong_close = closes[-1] > midpoint
                    is_volume_surge = current_volume > avg_volume * 1.2
                    
                    if is_strong_close or is_volume_surge:
                        confidence += 25
                        if is_strong_close: reasoning_parts.append("Strong close")
                        if is_volume_surge: reasoning_parts.append("Volume surge")
                        
                        # RSI Check (room to run)
                        if rsi_val < 70:
                            confidence += 10
                            reasoning_parts.append(f"RSI ok ({rsi_val:.1f})")
                    
                    # RSI hooking up
                    if rsi_val > 40 and rsi_val < 60:
                        confidence += 10
                        reasoning_parts.append(f"RSI in buy zone ({rsi_val:.1f})")

                    if confidence >= self.min_confidence:
                        stop_loss = min(lows[-5:], default=current_price * 0.99)
                        # Ensure stop is not too far
                        max_stop_dist = atr_val * self.atr_stop_mult
                        if current_price - stop_loss > max_stop_dist:
                            stop_loss = current_price - max_stop_dist
                            
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
                                'vwap': vwap_val,
                                'adx': adx_val,
                                'regime': regime.value
                            }
                        )

        # SHORT SIGNAL: Downtrend + Pullback + Rejection
        elif downtrend:
            reasoning_parts.append(f"Downtrend (ADX {adx_val:.1f})")
            
            # Check if we are in a pullback (Price > EMA Fast)
            if current_price > ema_f or current_price > ema_s:
                confidence += 15
                reasoning_parts.append("In pullback zone")
                
                # Check for rejection: Red candle
                if closes[-1] < opens[-1]:
                    # Quality Check: Weak close (lower half) OR Volume surge
                    midpoint = (highs[-1] + lows[-1]) / 2
                    is_weak_close = closes[-1] < midpoint
                    is_volume_surge = current_volume > avg_volume * 1.2
                    
                    if is_weak_close or is_volume_surge:
                        confidence += 25
                        if is_weak_close: reasoning_parts.append("Weak close")
                        if is_volume_surge: reasoning_parts.append("Volume surge")

                        # RSI Check (room to run)
                        if rsi_val > 30:
                            confidence += 10
                            reasoning_parts.append(f"RSI ok ({rsi_val:.1f})")
                    
                    # RSI hooking down
                    if rsi_val < 60 and rsi_val > 40:
                        confidence += 10
                        reasoning_parts.append(f"RSI in sell zone ({rsi_val:.1f})")

                    if confidence >= self.min_confidence:
                        stop_loss = max(highs[-5:], default=current_price * 1.01)
                        # Ensure stop is not too far
                        max_stop_dist = atr_val * self.atr_stop_mult
                        if stop_loss - current_price > max_stop_dist:
                            stop_loss = current_price + max_stop_dist

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
                                'vwap': vwap_val,
                                'adx': adx_val,
                                'regime': regime.value
                            }
                        )
        
        if signal:
            self.add_signal(signal)
            
        return signal
