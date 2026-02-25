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
        consolidation_bars: int = 15,         # 15 minute consolidation (better base)
        volume_threshold: float = 1.5,        # Moderate volume confirmation
        volume_lookback: int = 20,
        consolidation_range_pct: float = 0.5, # % of price — 0.5% for tight flag
        breakout_pct: float = 0.15,           # Breakout buffer
        volume_stop_pct: float = 1.5,         # Wider stop for volatile stocks
        rr_ratio: float = 2.5,                # Good R:R
        min_confidence: float = 60.0,         # Moderate bar
        trailing_stop_pct: float = 1.5        # Trail to catch runners
    ):
        super().__init__(
            name="Momentum",
            regimes=[Regime.TRENDING]
        )
        self._uses_l2_internally = True
        self.consolidation_bars = consolidation_bars
        self.volume_threshold = volume_threshold
        self.volume_lookback = volume_lookback
        self.consolidation_range_pct = consolidation_range_pct
        self.breakout_pct = breakout_pct
        self.volume_stop_pct = volume_stop_pct
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
        rsi = indicators.get('rsi')
        vwap = indicators.get('vwap')
        ema = indicators.get('ema') or indicators.get('ema_fast')
        adx = indicators.get('adx')
        
        rsi_val = rsi[-1] if isinstance(rsi, list) else (rsi or 50)
        vwap_val = vwap[-1] if isinstance(vwap, list) else (vwap or current_price)
        ema_val = ema[-1] if isinstance(ema, list) else (ema or current_price)
        adx_val = adx[-1] if isinstance(adx, list) else adx
        
        # ADX is required for momentum strategy - skip if unavailable (warmup)
        if adx_val is None:
            return None
        
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
        is_consolidation = (consol_range / current_price * 100) < self.consolidation_range_pct
        
        # Volume analysis
        volume_stats = self.get_volume_stats(volumes, self.volume_lookback)
        avg_volume = volume_stats["avg"]
        current_volume = volume_stats["current"]
        volume_ratio = volume_stats["ratio"]
        effective_volume_stop_pct = self.get_effective_volume_stop_pct() or self.volume_stop_pct
        effective_rr_ratio = self.get_effective_rr_ratio() or self.rr_ratio
        volume_surge = avg_volume and current_volume >= avg_volume * self.volume_threshold
        
        # Filter: Strong Momentum using ADX (lowered from 35 for more trade opportunities)
        if adx_val < 25:  # ADX > 25 indicates trending conditions
            return None
            
        # Get L2 Flow
        flow = indicators.get("order_flow") or {}
        signed_aggr = float(flow.get("signed_aggression", 0.0) or 0.0)
        
        signal = None
        confidence = 30.0  # Lowered from 50 to enforce requiring strong volume/consolidation
        reasoning_parts = []
        
        # LONG BREAKOUT
        breakout_buffer = current_price * (self.volume_adjusted_pct(self.breakout_pct, volume_ratio) / 100)
        if current_price > consol_high + breakout_buffer:
            # Order Flow Filter
            if signed_aggr < 0.02:
                return None
                
            # Trend Filter: Price must be aligned with SMA (Trend)
            sma_val = indicators.get('sma')[-1] if indicators.get('sma') else None
            if sma_val and current_price < sma_val:
                return None
                
            # RSI Filter: Don't buy overbought
            if rsi_val > 70:
                return None
            
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
                stop_pct = self.volume_adjusted_pct(effective_volume_stop_pct, volume_ratio)
                stop_loss = max(consol_low, current_price * (1 - stop_pct / 100))
                take_profit = self.calculate_take_profit(current_price, stop_loss, effective_rr_ratio, 'long')
                
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
                        'consol_high': consol_high,
                        'consol_low': consol_low,
                        'consol_range': consol_range,
                        'volume_ratio': volume_ratio,
                        'rsi': rsi_val,
                        'regime': regime.value,
                        'adx': adx_val
                    }
                )
        
        # SHORT BREAKDOWN
        elif current_price < consol_low - breakout_buffer:
            # Order Flow Filter
            if signed_aggr > -0.02:
                return None
                
            # Trend Filter: Price must be aligned with SMA (Trend)
            sma_val = indicators.get('sma')[-1] if indicators.get('sma') else None
            if sma_val and current_price > sma_val:
                return None

            # RSI Filter: Don't sell oversold
            if rsi_val < 30:
                return None

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
                stop_pct = self.volume_adjusted_pct(effective_volume_stop_pct, volume_ratio)
                stop_loss = min(consol_high, current_price * (1 + stop_pct / 100))
                take_profit = self.calculate_take_profit(current_price, stop_loss, effective_rr_ratio, 'short')
                
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
                        'consol_high': consol_high,
                        'consol_low': consol_low,
                        'consol_range': consol_range,
                        'volume_ratio': volume_ratio,
                        'rsi': rsi_val,
                        'regime': regime.value,
                        'adx': adx_val
                    }
                )
        
        if signal:
            signal = self.apply_l2_flow_boost(signal, indicators)
            self.add_signal(signal)
            
        return signal
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'consolidation_bars': self.consolidation_bars,
            'volume_threshold': self.volume_threshold,
            'volume_lookback': self.volume_lookback,
            'consolidation_range_pct': self.consolidation_range_pct,
            'breakout_pct': self.breakout_pct,
            'volume_stop_pct': self.volume_stop_pct,
            'rr_ratio': self.rr_ratio,
            'trailing_stop_pct': self.trailing_stop_pct
        })
        return base
