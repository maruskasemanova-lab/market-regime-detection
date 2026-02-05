"""
VWAP Magnet Strategy - Universal strategy across all regimes.
Uses VWAP as price attractor for both entries and exits.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime

from .base_strategy import BaseStrategy, Signal, SignalType, Regime


class VWAPMagnetStrategy(BaseStrategy):
    """
    VWAP Magnet Strategy - works in all regimes.
    
    Core concept: VWAP acts as a "magnet" - price tends to return to it.
    
    Logic:
    - Track distance from VWAP
    - Score the "magnet pull" based on:
      * Distance from VWAP
      * Time since last VWAP touch
      * Volume profile around VWAP
    - Enter when pull is strong, exit at VWAP
    
    Works in: ALL regimes
    """
    
    def __init__(
        self,
        min_distance_pct: float = 0.4,         # Lowered to 0.4 for active scalping (was 0.8)
        max_distance_pct: float = 3.0,         # Max distance
        bars_since_vwap_threshold: int = 5,    # Faster trigger (was 10)
        volume_confirm: bool = True,           # Require volume confirmation
        volume_lookback: int = 20,
        volume_stop_pct: float = 0.7,
        min_confidence: float = 60.0,          # Min confidence
        trailing_stop_pct: float = 0.4,        # Tight trail to VWAP
        long_only: bool = False                # Only take long trades (shorts lose $233)
    ):
        super().__init__(
            name="VWAPMagnet",
            regimes=[Regime.TRENDING, Regime.CHOPPY, Regime.MIXED]  # All regimes
        )
        self.min_distance_pct = min_distance_pct
        self.max_distance_pct = max_distance_pct
        self.bars_since_vwap_threshold = bars_since_vwap_threshold
        self.volume_confirm = volume_confirm
        self.volume_lookback = volume_lookback
        self.volume_stop_pct = volume_stop_pct
        self.min_confidence = min_confidence
        self.trailing_stop_pct = trailing_stop_pct
        self.long_only = long_only
        
    def _bars_since_vwap_touch(
        self, 
        closes: List[float], 
        highs: List[float],
        lows: List[float],
        vwap: float, 
        tolerance_pct: float = 0.1
    ) -> int:
        """Count bars since price last touched VWAP."""
        tolerance = vwap * tolerance_pct / 100
        
        for i in range(len(closes) - 1, -1, -1):
            # Check if VWAP was within the bar's range
            if lows[i] - tolerance <= vwap <= highs[i] + tolerance:
                return len(closes) - 1 - i
        
        return len(closes)  # Never touched in lookback
    
    def _calculate_vwap_volume_profile(
        self,
        closes: List[float],
        volumes: List[float],
        vwap: float,
        range_pct: float = 0.5
    ) -> Dict[str, float]:
        """Calculate volume distribution around VWAP."""
        above_vwap_vol = 0
        below_vwap_vol = 0
        at_vwap_vol = 0
        
        range_val = vwap * range_pct / 100
        
        for close, vol in zip(closes[-20:], volumes[-20:]):
            if close > vwap + range_val:
                above_vwap_vol += vol
            elif close < vwap - range_val:
                below_vwap_vol += vol
            else:
                at_vwap_vol += vol
        
        total = above_vwap_vol + below_vwap_vol + at_vwap_vol
        if total == 0:
            return {'above': 0.33, 'below': 0.33, 'at': 0.33}
            
        return {
            'above': above_vwap_vol / total,
            'below': below_vwap_vol / total,
            'at': at_vwap_vol / total
        }
        
    def generate_signal(
        self,
        current_price: float,
        ohlcv: Dict[str, List[float]],
        indicators: Dict[str, Any],
        regime: Regime,
        timestamp: datetime
    ) -> Optional[Signal]:
        """
        Generate VWAP magnet signal.
        
        Entry conditions:
        - Price is stretched away from VWAP
        - Time since VWAP touch increases pull
        - Volume supports the move back
        """
        if not self.is_allowed_in_regime(regime):
            return None
            
        if len(self.get_open_positions()) > 0:
            return None
        
        # Get indicators
        vwap = indicators.get('vwap')
        rsi = indicators.get('rsi')
        
        if vwap is None:
            return None
            
        vwap_val = vwap[-1] if isinstance(vwap, list) else vwap
        rsi_val = rsi[-1] if isinstance(rsi, list) else (rsi or 50)
        
        # Get OHLC
        highs = ohlcv.get('high', [])
        lows = ohlcv.get('low', [])
        closes = ohlcv.get('close', [])
        volumes = ohlcv.get('volume', [])
        
        if len(closes) < 20:
            return None
        
        # Calculate VWAP distance
        vwap_distance_pct = abs((current_price - vwap_val) / vwap_val * 100)
        price_above_vwap = current_price > vwap_val
        
        # Check if distance is in valid range
        if vwap_distance_pct < self.min_distance_pct:
            return None  # Too close, no trade
        if vwap_distance_pct > self.max_distance_pct:
            return None  # Too stretched, might trend further
        
        # Calculate bars since VWAP touch
        bars_since = self._bars_since_vwap_touch(closes, highs, lows, vwap_val)
        
        # Volume profile
        vol_profile = self._calculate_vwap_volume_profile(closes, volumes, vwap_val)
        
        # Current volume vs average
        volume_stats = self.get_volume_stats(volumes, self.volume_lookback)
        avg_volume = volume_stats["avg"]
        current_volume = volume_stats["current"]
        volume_ratio = volume_stats["ratio"]
        volume_declining = avg_volume and current_volume < avg_volume * 0.8
        
        signal = None
        confidence = 50.0
        reasoning_parts = []
        
        # "MAGNET PULL" SCORING
        
        # Distance score (further = stronger pull, up to a point)
        distance_score = min(vwap_distance_pct / self.min_distance_pct * 10, 20)
        confidence += distance_score
        reasoning_parts.append(f"VWAP distance: {vwap_distance_pct:.2f}%")
        
        # Time score (longer away = stronger pull)
        if bars_since >= self.bars_since_vwap_threshold:
            time_score = min((bars_since - self.bars_since_vwap_threshold) * 2, 15)
            confidence += time_score
            reasoning_parts.append(f"{bars_since} bars since VWAP touch")
        
        # Volume exhaustion score
        if self.volume_confirm and volume_declining:
            confidence += 10
            reasoning_parts.append("Volume declining (exhaustion)")
        
        # LONG SIGNAL: Price below VWAP, expect pull up
        if not price_above_vwap:
            # Volume profile: high volume below = support
            if vol_profile['below'] > 0.4:
                confidence += 10
                reasoning_parts.append("High volume below VWAP (support)")
            
            # RSI oversold helps
            if rsi_val < 40:
                confidence += 10
                reasoning_parts.append(f"RSI oversold ({rsi_val:.1f})")
            
            # Regime adjustment
            if regime == Regime.CHOPPY:
                confidence += 5
                reasoning_parts.append("CHOPPY regime favors VWAP return")
            
            if confidence >= self.min_confidence:
                stop_pct = self.volume_adjusted_pct(self.volume_stop_pct, volume_ratio)
                stop_loss = self.calculate_percent_stop(current_price, stop_pct, 'long')
                take_profit = vwap_val  # Target VWAP exactly
                
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
                        'bars_since_vwap': bars_since,
                        'volume_profile': vol_profile,
                        'volume_ratio': volume_ratio,
                        'rsi': rsi_val,
                        'regime': regime.value
                    }
                )
        # SHORT SIGNAL: Price above VWAP, expect pull down
        # Skip if long_only mode is enabled
        elif not self.long_only:
            # Volume profile: high volume above = resistance
            if vol_profile['above'] > 0.4:
                confidence += 10
                reasoning_parts.append("High volume above VWAP (resistance)")
            
            # RSI overbought helps
            if rsi_val > 60:
                confidence += 10
                reasoning_parts.append(f"RSI overbought ({rsi_val:.1f})")
            
            if regime == Regime.CHOPPY:
                confidence += 5
                reasoning_parts.append("CHOPPY regime favors VWAP return")
            
            if confidence >= self.min_confidence:
                stop_pct = self.volume_adjusted_pct(self.volume_stop_pct, volume_ratio)
                stop_loss = self.calculate_percent_stop(current_price, stop_pct, 'short')
                take_profit = vwap_val
                
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
                        'bars_since_vwap': bars_since,
                        'volume_profile': vol_profile,
                        'volume_ratio': volume_ratio,
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
            'min_distance_pct': self.min_distance_pct,
            'max_distance_pct': self.max_distance_pct,
            'bars_since_vwap_threshold': self.bars_since_vwap_threshold,
            'volume_confirm': self.volume_confirm,
            'volume_lookback': self.volume_lookback,
            'volume_stop_pct': self.volume_stop_pct,
            'trailing_stop_pct': self.trailing_stop_pct
        })
        return base
