"""
Mean Reversion Strategy - For CHOPPY regime.
Trades price returning to VWAP/mean after extended moves.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime

from .base_strategy import BaseStrategy, Signal, SignalType, Regime


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy for choppy/range-bound markets.
    
    Logic:
    - Enter when price deviates significantly from VWAP
    - Target return to VWAP
    - Use tight stop beyond the extreme
    
    Best in: CHOPPY regime
    """
    
    def __init__(
        self,
        entry_deviation_pct: float = 0.3,    # Lowered from 1.2 for active scalping
        atr_stop_mult: float = 2.0,          # Lowered from 3.0
        min_confidence: float = 60.0,        # Higher bar (was 50)
        volume_confirmation: bool = True,
        trailing_stop_pct: float = 0.3       # Very tight trail
    ):
        super().__init__(
            name="MeanReversion",
            regimes=[Regime.CHOPPY, Regime.MIXED]
        )
        self.entry_deviation_pct = entry_deviation_pct
        self.atr_stop_mult = atr_stop_mult
        self.min_confidence = min_confidence
        self.volume_confirmation = volume_confirmation
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
        Generate mean reversion signal.
        
        Entry conditions:
        - Price is > entry_deviation_pct away from VWAP
        - Volume is not extremely high (would indicate breakout)
        - RSI confirms oversold/overbought
        """
        if not self.is_allowed_in_regime(regime):
            return None
            
        # Check if we already have open positions
        if len(self.get_open_positions()) > 0:
            return None
        
        # Get indicators
        vwap = indicators.get('vwap')
        atr = indicators.get('atr')
        rsi = indicators.get('rsi')
        adx = indicators.get('adx')
        
        if vwap is None or atr is None:
            return None
            
        # Get latest values
        vwap_val = vwap[-1] if isinstance(vwap, list) else vwap
        atr_val = atr[-1] if isinstance(atr, list) else atr
        rsi_val = rsi[-1] if isinstance(rsi, list) else (rsi or 50)
        adx_val = adx[-1] if isinstance(adx, list) else (adx or 0)
        
        # Determine threshold based on regime
        threshold = self.entry_deviation_pct  # Default (0.3 for Choppy)
        
        if regime == Regime.TRENDING:
            threshold = 2.0  # Much stricter for Trending (only fade extremes)
        elif regime == Regime.MIXED:
            threshold = 1.0  # Moderate for Mixed
            
        # Calculate deviation from VWAP
        vwap_distance_pct = self.get_vwap_distance(current_price, vwap_val)
        
        # Debug logging
        # print(f"DEBUG: MeanReversion - Price: {current_price:.2f}, VWAP: {vwap_val:.2f}, Dist: {vwap_distance_pct:.2f}%, Threshold: {self.entry_deviation_pct}%")
        
        # Calculate volume metrics
        volumes = ohlcv.get('volume', [])
        if len(volumes) >= 20:
            avg_volume = sum(volumes[-20:]) / 20
            current_volume = volumes[-1]
        else:
            avg_volume = sum(volumes) / len(volumes) if volumes else 0
            current_volume = volumes[-1] if volumes else 0
        
        signal = None
        confidence = 50.0
        reasoning_parts = []
        
        # Filter: Ensure market is not strongly trending against us
        # In TRENDING regime, we allow high ADX (that's the point of fading the extreme)
        # But we must be very careful.
        if regime != Regime.TRENDING and adx_val > 30:
            return None
        
        # LONG SIGNAL: Price below VWAP (oversold)
        if vwap_distance_pct <= -threshold:
            # Confirm with RSI
            if rsi_val < 35:
                confidence += 20
                reasoning_parts.append(f"RSI oversold ({rsi_val:.1f})")
            elif rsi_val < 45:
                confidence += 20
                reasoning_parts.append(f"RSI weak ({rsi_val:.1f})")
            
            # Deviation scoring
            if abs(vwap_distance_pct) > threshold * 2.0:
                confidence += 20
                reasoning_parts.append(f"Strong deviation ({vwap_distance_pct:.2f}%)")
            else:
                confidence += 10
                reasoning_parts.append(f"Deviation from VWAP ({vwap_distance_pct:.2f}%)")
            
            # Volume check - want declining volume at extreme
            if current_volume < avg_volume:
                confidence += 10
                reasoning_parts.append("Declining volume (exhaustion)")
            
            # Regime bonus
            if regime == Regime.CHOPPY:
                confidence += 10
                reasoning_parts.append("CHOPPY regime favors mean reversion")
            
            if confidence >= self.min_confidence:
                stop_loss = self.calculate_atr_stop(current_price, atr_val, self.atr_stop_mult, 'long')
                take_profit = vwap_val  # Target VWAP
                
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
                        'vwap_distance_pct': vwap_distance_pct,
                        'atr': atr_val,
                        'rsi': rsi_val,
                        'regime': regime.value,
                        'adx': adx_val
                    }
                )
        
        # SHORT SIGNAL: Price above VWAP (overbought)
        elif vwap_distance_pct >= threshold:
            print(f"DEBUG: Checking SHORT signal. Dist: {vwap_distance_pct}")
            # Confirm with RSI
            if rsi_val > 65:
                confidence += 20
                reasoning_parts.append(f"RSI overbought ({rsi_val:.1f})")
            elif rsi_val > 55:
                confidence += 10
                reasoning_parts.append(f"RSI strong ({rsi_val:.1f})")
            
            # Deviation scoring
            if abs(vwap_distance_pct) > threshold * 2.0:
                confidence += 20
                reasoning_parts.append(f"Strong deviation ({vwap_distance_pct:.2f}%)")
            else:
                confidence += 10
                reasoning_parts.append(f"Deviation from VWAP ({vwap_distance_pct:.2f}%)")
            
            # Volume check
            if current_volume < avg_volume:
                confidence += 10
                reasoning_parts.append("Declining volume (exhaustion)")
            
            if regime == Regime.CHOPPY:
                confidence += 10
                reasoning_parts.append("CHOPPY regime favors mean reversion")
            
            if confidence >= self.min_confidence:
                stop_loss = self.calculate_atr_stop(current_price, atr_val, self.atr_stop_mult, 'short')
                take_profit = vwap_val  # Target VWAP
                
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
                        'vwap_distance_pct': vwap_distance_pct,
                        'atr': atr_val,
                        'rsi': rsi_val,
                        'regime': regime.value,
                        'adx': adx_val
                    }
                )
        
        if signal:
            self.add_signal(signal)
            
        return signal
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'entry_deviation_pct': self.entry_deviation_pct,
            'atr_stop_mult': self.atr_stop_mult,
            'min_confidence': self.min_confidence,
            'trailing_stop_pct': self.trailing_stop_pct
        })
        return base
