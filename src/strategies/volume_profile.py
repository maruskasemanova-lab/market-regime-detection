"""
Volume Profile Strategy - Based on Ludvík Turek's Volume Symmetry concepts.

Key Concepts:
- P-Profile: High volume at lows → expect resistance at 161.8%/200% extension
- B-Profile: High volume at highs → expect support at 161.8%/200% extension  
- Trade the symmetry levels (1:1, 161.8%, 200%)
"""
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from collections import defaultdict

from .base_strategy import BaseStrategy, Signal, SignalType, Regime


class VolumeProfileStrategy(BaseStrategy):
    """
    Volume Profile Strategy based on Ludvík Turek's concepts.
    
    Core Logic:
    1. Calculate Volume Profile (POC, Value Area)
    2. Detect P-profile (high volume at lows) vs B-profile (high volume at highs)
    3. Calculate symmetry levels (161.8%, 200%)
    4. Trade reactions at these levels
    
    Works best in: TRENDING, MIXED regimes
    """
    
    def __init__(
        self,
        profile_lookback: int = 60,           # Bars to build profile (1 hour on 1m)
        num_bins: int = 20,                   # Number of price bins
        pb_threshold: float = 0.6,            # Ratio threshold for P/B detection
        symmetry_tolerance_pct: float = 0.15, # How close to symmetry level to trigger
        min_confidence: float = 65.0,
        atr_stop_mult: float = 2.5,           # Wider stops
        rr_ratio: float = 2.0,
        trailing_stop_pct: float = 0.8,
        volume_lookback: int = 20
    ):
        super().__init__(
            name="VolumeProfile",
            regimes=[Regime.TRENDING, Regime.MIXED]
        )
        self.profile_lookback = profile_lookback
        self.num_bins = num_bins
        self.pb_threshold = pb_threshold
        self.symmetry_tolerance_pct = symmetry_tolerance_pct
        self.min_confidence = min_confidence
        self.atr_stop_mult = atr_stop_mult
        self.rr_ratio = rr_ratio
        self.trailing_stop_pct = trailing_stop_pct
        self.volume_lookback = volume_lookback
        
        # Cache for profile calculations
        self._last_profile: Optional[Dict] = None
        self._profile_timestamp: Optional[datetime] = None
        
    def calculate_volume_profile(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float]
    ) -> Dict[str, Any]:
        """
        Calculate Volume Profile metrics.
        
        Returns:
            Dict with:
            - poc: Point of Control (price with highest volume)
            - vah: Value Area High (upper 70% boundary)
            - val: Value Area Low (lower 70% boundary)
            - profile_type: 'P' or 'B' or 'D' (double distribution)
            - bins: Volume distribution by price bins
        """
        if len(closes) < 10 or len(volumes) < 10:
            return {}
            
        # Get price range
        price_high = max(highs)
        price_low = min(lows)
        price_range = price_high - price_low
        
        if price_range <= 0:
            return {}
            
        bin_size = price_range / self.num_bins
        
        # Build volume distribution
        volume_by_bin = defaultdict(float)
        
        for i, (close, volume) in enumerate(zip(closes, volumes)):
            bin_idx = int((close - price_low) / bin_size)
            bin_idx = min(bin_idx, self.num_bins - 1)
            volume_by_bin[bin_idx] += volume
            
        # Find POC (bin with highest volume)
        poc_bin = max(volume_by_bin.keys(), key=lambda k: volume_by_bin[k])
        poc_price = price_low + (poc_bin + 0.5) * bin_size
        
        # Calculate Value Area (70% of volume)
        total_volume = sum(volume_by_bin.values())
        target_volume = total_volume * 0.70
        
        accumulated = 0
        val_bin = poc_bin
        vah_bin = poc_bin
        
        # Expand from POC until 70% reached
        while accumulated < target_volume:
            # Check which side to expand
            lower_vol = volume_by_bin.get(val_bin - 1, 0)
            upper_vol = volume_by_bin.get(vah_bin + 1, 0)
            
            if lower_vol >= upper_vol and val_bin > 0:
                val_bin -= 1
                accumulated += volume_by_bin[val_bin]
            elif vah_bin < self.num_bins - 1:
                vah_bin += 1
                accumulated += volume_by_bin[vah_bin]
            else:
                break
                
        val_price = price_low + val_bin * bin_size
        vah_price = price_low + (vah_bin + 1) * bin_size
        
        # Determine P/B profile type
        # P = high volume at bottom, B = high volume at top
        lower_half_volume = sum(v for k, v in volume_by_bin.items() if k < self.num_bins / 2)
        upper_half_volume = sum(v for k, v in volume_by_bin.items() if k >= self.num_bins / 2)
        
        if lower_half_volume > upper_half_volume * (1 + self.pb_threshold):
            profile_type = 'P'  # High volume at lows
        elif upper_half_volume > lower_half_volume * (1 + self.pb_threshold):
            profile_type = 'B'  # High volume at highs
        else:
            profile_type = 'D'  # Double/balanced distribution
            
        return {
            'poc': poc_price,
            'vah': vah_price,
            'val': val_price,
            'profile_type': profile_type,
            'price_high': price_high,
            'price_low': price_low,
            'total_volume': total_volume,
            'lower_volume_ratio': lower_half_volume / (total_volume + 0.001),
            'upper_volume_ratio': upper_half_volume / (total_volume + 0.001)
        }
    
    def calculate_symmetry_levels(
        self, 
        profile: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate Fibonacci symmetry levels from volume profile.
        
        Based on Turek's concept:
        - P-profile: Measure from high volume zone (bottom) to top
        - B-profile: Measure from high volume zone (top) to bottom
        
        Returns levels at 100% (1:1), 161.8%, and 200%
        """
        poc = profile['poc']
        price_high = profile['price_high']
        price_low = profile['price_low']
        profile_type = profile['profile_type']
        
        levels = {}
        
        if profile_type == 'P':
            # P-profile: High volume at bottom
            # Measure from POC up to high, project resistance above
            base_move = price_high - poc
            
            levels['resistance_100'] = price_high + base_move
            levels['resistance_161'] = price_high + base_move * 1.618
            levels['resistance_200'] = price_high + base_move * 2.0
            
            # Support is at the high volume zone
            levels['support'] = poc
            levels['direction'] = 'up'  # Expected reaction is down from resistance
            
        elif profile_type == 'B':
            # B-profile: High volume at top
            # Measure from POC down to low, project support below
            base_move = poc - price_low
            
            levels['support_100'] = price_low - base_move
            levels['support_161'] = price_low - base_move * 1.618
            levels['support_200'] = price_low - base_move * 2.0
            
            # Resistance is at the high volume zone
            levels['resistance'] = poc
            levels['direction'] = 'down'  # Expected reaction is up from support
            
        else:
            # Balanced - use both
            levels['resistance'] = profile['vah']
            levels['support'] = profile['val']
            levels['direction'] = 'neutral'
            
        return levels
    
    def generate_signal(
        self,
        current_price: float,
        ohlcv: Dict[str, List[float]],
        indicators: Dict[str, Any],
        regime: Regime,
        timestamp: datetime
    ) -> Optional[Signal]:
        """Generate signal based on volume profile and symmetry levels."""
        
        if not self.is_allowed_in_regime(regime):
            return None
            
        if len(self.get_open_positions()) > 0:
            return None
            
        # Get OHLCV
        opens = ohlcv.get('open', [])
        highs = ohlcv.get('high', [])
        lows = ohlcv.get('low', [])
        closes = ohlcv.get('close', [])
        volumes = ohlcv.get('volume', [])
        
        if len(closes) < self.profile_lookback:
            return None
            
        # Get recent data for profile
        recent_highs = highs[-self.profile_lookback:]
        recent_lows = lows[-self.profile_lookback:]
        recent_closes = closes[-self.profile_lookback:]
        recent_volumes = volumes[-self.profile_lookback:]
        
        # Calculate volume profile
        profile = self.calculate_volume_profile(
            recent_highs, recent_lows, recent_closes, recent_volumes
        )
        
        if not profile:
            return None
            
        # Calculate symmetry levels
        levels = self.calculate_symmetry_levels(profile)
        
        # Get indicators
        atr = indicators.get('atr')
        atr_val = atr[-1] if isinstance(atr, list) else (atr or current_price * 0.01)
        
        vwap = indicators.get('vwap')
        vwap_val = vwap[-1] if isinstance(vwap, list) else (vwap or current_price)
        
        # Volume confirmation
        volume_stats = self.get_volume_stats(volumes, self.volume_lookback)
        volume_ratio = volume_stats['ratio']
        effective_rr_ratio = self.get_effective_rr_ratio() or self.rr_ratio
        
        signal = None
        confidence = 30.0  # Lowered from 50 to force candle confirmation
        reasoning_parts = [f"Profile: {profile['profile_type']}"]
        
        # P-PROFILE LOGIC: Trade SHORT at resistance levels
        if profile['profile_type'] == 'P':
            # Check if price is near resistance levels
            for level_name in ['resistance_200', 'resistance_161', 'resistance_100']:
                level = levels.get(level_name)
                if level is None:
                    continue
                    
                distance_pct = abs(current_price - level) / current_price * 100
                
                if distance_pct < self.symmetry_tolerance_pct:
                    confidence += 20
                    reasoning_parts.append(f"At {level_name}: {level:.2f}")
                    
                    # Need bearish candle confirmation
                    if closes[-1] < opens[-1]:
                        confidence += 15
                        reasoning_parts.append("Bearish candle")
                        
                        # Volume surge adds confidence
                        if volume_ratio > 1.3:
                            confidence += 10
                            reasoning_parts.append(f"Volume surge ({volume_ratio:.1f}x)")
                            
                        if confidence >= self.min_confidence:
                            stop_loss = current_price + atr_val * self.atr_stop_mult
                            take_profit = self.calculate_take_profit(
                                current_price, stop_loss, effective_rr_ratio, 'short'
                            )
                            
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
                                    'profile_type': profile['profile_type'],
                                    'poc': profile['poc'],
                                    'symmetry_level': level,
                                    'level_name': level_name,
                                    'volume_ratio': volume_ratio,
                                    'regime': regime.value
                                }
                            )
                            break
                            
        # B-PROFILE LOGIC: Trade LONG at support levels
        elif profile['profile_type'] == 'B':
            # Check if price is near support levels
            for level_name in ['support_200', 'support_161', 'support_100']:
                level = levels.get(level_name)
                if level is None:
                    continue
                    
                distance_pct = abs(current_price - level) / current_price * 100
                
                if distance_pct < self.symmetry_tolerance_pct:
                    confidence += 20
                    reasoning_parts.append(f"At {level_name}: {level:.2f}")
                    
                    # Need bullish candle confirmation
                    if closes[-1] > opens[-1]:
                        confidence += 15
                        reasoning_parts.append("Bullish candle")
                        
                        # Volume surge adds confidence
                        if volume_ratio > 1.3:
                            confidence += 10
                            reasoning_parts.append(f"Volume surge ({volume_ratio:.1f}x)")
                            
                        if confidence >= self.min_confidence:
                            stop_loss = current_price - atr_val * self.atr_stop_mult
                            take_profit = self.calculate_take_profit(
                                current_price, stop_loss, effective_rr_ratio, 'long'
                            )
                            
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
                                    'profile_type': profile['profile_type'],
                                    'poc': profile['poc'],
                                    'symmetry_level': level,
                                    'level_name': level_name,
                                    'volume_ratio': volume_ratio,
                                    'regime': regime.value
                                }
                            )
                            break
        
        # BALANCED PROFILE: Trade mean reversion at VAH/VAL
        else:
            # Filter: Do not mean-revert if the market is strongly trending
            adx_val = indicators.get('adx', [0])[-1] if indicators.get('adx') else 0
            if adx_val > 25:
                return None
                
            # Near Value Area High - look for short
            if current_price >= profile['vah']:
                distance_pct = (current_price - profile['vah']) / current_price * 100
                if distance_pct < self.symmetry_tolerance_pct * 2:
                    confidence += 15
                    reasoning_parts.append(f"Near VAH: {profile['vah']:.2f}")
                    
                    if closes[-1] < opens[-1]:
                        confidence += 10
                        reasoning_parts.append("Bearish candle")
                        if volume_ratio > 1.2:
                            confidence += 10
                            reasoning_parts.append("Bearish rejection volume")
                        
                        if confidence >= self.min_confidence:
                            stop_loss = current_price + atr_val * self.atr_stop_mult
                            take_profit = profile['poc']
                            
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
                                    'profile_type': 'D',
                                    'poc': profile['poc'],
                                    'vah': profile['vah'],
                                    'val': profile['val'],
                                    'regime': regime.value
                                }
                            )
            
            # Near Value Area Low - look for long
            elif current_price <= profile['val']:
                distance_pct = (profile['val'] - current_price) / current_price * 100
                if distance_pct < self.symmetry_tolerance_pct * 2:
                    confidence += 15
                    reasoning_parts.append(f"Near VAL: {profile['val']:.2f}")
                    
                    if closes[-1] > opens[-1]:
                        confidence += 10
                        reasoning_parts.append("Bullish candle")
                        if volume_ratio > 1.2:
                            confidence += 10
                            reasoning_parts.append("Bullish bounce volume")
                        
                        if confidence >= self.min_confidence:
                            stop_loss = current_price - atr_val * self.atr_stop_mult
                            take_profit = profile['poc']
                            
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
                                    'profile_type': 'D',
                                    'poc': profile['poc'],
                                    'vah': profile['vah'],
                                    'val': profile['val'],
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
            'profile_lookback': self.profile_lookback,
            'num_bins': self.num_bins,
            'pb_threshold': self.pb_threshold,
            'symmetry_tolerance_pct': self.symmetry_tolerance_pct,
            'atr_stop_mult': self.atr_stop_mult,
            'rr_ratio': self.rr_ratio,
            'trailing_stop_pct': self.trailing_stop_pct
        })
        return base
