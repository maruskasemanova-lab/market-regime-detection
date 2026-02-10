"""
Gap & Swing Liquidity Strategy - Based on Ludvík Turek's swing trading concepts.

Key Concepts from Turek's livestream:
- Daily GAPs as entry signals
- Swing Liquidity zones (previous highs/lows) for stop placement
- 1-2-3 market structure patterns
- Limit entries at predefined levels
"""
from typing import Optional, Dict, Any, List
from datetime import datetime

from .base_strategy import BaseStrategy, Signal, SignalType, Regime


class GapLiquidityStrategy(BaseStrategy):
    """
    Gap and Swing Liquidity Strategy for intraday trading.
    
    Core Logic:
    1. Identify daily/overnight GAPs
    2. Find Swing Liquidity zones (clusters of highs/lows)
    3. Trade reactions at GAP fills with swing liquidity stops
    4. Follow 1-2-3 structure for entries
    
    Works best in: TRENDING, MIXED regimes
    """
    
    def __init__(
        self,
        gap_threshold_pct: float = 0.3,       # Min gap size to trade
        swing_lookback: int = 20,             # Bars to find swing points
        liquidity_cluster_bars: int = 5,      # Bars for liquidity cluster
        gap_fill_tolerance_pct: float = 0.1,  # How close for gap fill signal
        min_confidence: float = 65.0,
        atr_stop_mult: float = 2.5,
        rr_ratio: float = 2.5,
        trailing_stop_pct: float = 1.0,
        volume_lookback: int = 20
    ):
        super().__init__(
            name="GapLiquidity",
            regimes=[Regime.TRENDING, Regime.MIXED]
        )
        self.gap_threshold_pct = gap_threshold_pct
        self.swing_lookback = swing_lookback
        self.liquidity_cluster_bars = liquidity_cluster_bars
        self.gap_fill_tolerance_pct = gap_fill_tolerance_pct
        self.min_confidence = min_confidence
        self.atr_stop_mult = atr_stop_mult
        self.rr_ratio = rr_ratio
        self.trailing_stop_pct = trailing_stop_pct
        self.volume_lookback = volume_lookback
        
        # Track gap levels
        self._gap_up_level: Optional[float] = None
        self._gap_down_level: Optional[float] = None
        
    def find_swing_levels(
        self,
        highs: List[float],
        lows: List[float],
        lookback: int = 20
    ) -> Dict[str, List[float]]:
        """
        Find significant swing highs and lows (liquidity pools).
        
        Returns:
            Dict with 'swing_highs' and 'swing_lows' lists
        """
        swing_highs = []
        swing_lows = []
        
        if len(highs) < lookback:
            return {'swing_highs': [], 'swing_lows': []}
            
        # Find local maxima/minima
        for i in range(2, len(highs) - 2):
            # Swing high: higher than neighbors
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append(highs[i])
                
            # Swing low: lower than neighbors
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append(lows[i])
                
        return {
            'swing_highs': sorted(swing_highs, reverse=True)[:5],  # Top 5
            'swing_lows': sorted(swing_lows)[:5]  # Bottom 5
        }
    
    def detect_gap(
        self,
        opens: List[float],
        closes: List[float],
        highs: List[float],
        lows: List[float]
    ) -> Dict[str, Any]:
        """
        Detect overnight/intraday gaps.
        
        A gap occurs when open is outside previous bar's range.
        """
        if len(opens) < 2:
            return {'type': None, 'size_pct': 0, 'level': None}
            
        prev_close = closes[-2]
        prev_high = highs[-2]
        prev_low = lows[-2]
        current_open = opens[-1]
        
        gap_size = 0
        gap_type = None
        gap_level = None
        
        # Gap up: Open above previous high
        if current_open > prev_high:
            gap_size = (current_open - prev_high) / prev_close * 100
            if gap_size >= self.gap_threshold_pct:
                gap_type = 'up'
                gap_level = prev_high  # Gap fill target
                
        # Gap down: Open below previous low
        elif current_open < prev_low:
            gap_size = (prev_low - current_open) / prev_close * 100
            if gap_size >= self.gap_threshold_pct:
                gap_type = 'down'
                gap_level = prev_low  # Gap fill target
                
        return {
            'type': gap_type,
            'size_pct': gap_size,
            'level': gap_level,
            'prev_close': prev_close
        }
    
    def find_123_pattern(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float]
    ) -> Dict[str, Any]:
        """
        Detect 1-2-3 market structure pattern.
        
        Bullish 1-2-3:
        1. Swing low (point 1)
        2. Rally to swing high (point 2)
        3. Pullback higher low (point 3)
        -> Entry on break of point 2
        
        Bearish 1-2-3:
        1. Swing high (point 1)
        2. Drop to swing low (point 2)
        3. Rally to lower high (point 3)
        -> Entry on break of point 2
        """
        if len(closes) < 10:
            return {'type': None}
            
        # Look at last 10 bars
        recent_highs = highs[-10:]
        recent_lows = lows[-10:]
        
        # Find potential pattern
        # Bullish: Low -> High -> Higher Low
        min_idx = recent_lows.index(min(recent_lows))
        max_idx = recent_highs.index(max(recent_highs))
        
        if min_idx < max_idx:  # Potential bullish
            point1 = min(recent_lows)
            point2 = max(recent_highs)
            # Check for higher low after point2
            if max_idx < len(recent_lows) - 1:
                point3 = min(recent_lows[max_idx:])
                if point3 > point1:
                    return {
                        'type': 'bullish',
                        'point1': point1,
                        'point2': point2,
                        'point3': point3,
                        'entry_level': point2  # Break above
                    }
                    
        elif max_idx < min_idx:  # Potential bearish
            point1 = max(recent_highs)
            point2 = min(recent_lows)
            # Check for lower high after point2
            if min_idx < len(recent_highs) - 1:
                point3 = max(recent_highs[min_idx:])
                if point3 < point1:
                    return {
                        'type': 'bearish',
                        'point1': point1,
                        'point2': point2,
                        'point3': point3,
                        'entry_level': point2  # Break below
                    }
                    
        return {'type': None}
    
    def generate_signal(
        self,
        current_price: float,
        ohlcv: Dict[str, List[float]],
        indicators: Dict[str, Any],
        regime: Regime,
        timestamp: datetime
    ) -> Optional[Signal]:
        """Generate signal based on gaps and swing liquidity."""
        
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
        
        if len(closes) < self.swing_lookback:
            return None
            
        # Get indicators
        atr = indicators.get('atr')
        atr_val = atr[-1] if isinstance(atr, list) else (atr or current_price * 0.01)
        
        vwap = indicators.get('vwap')
        vwap_val = vwap[-1] if isinstance(vwap, list) else (vwap or current_price)
        
        # Detect gap
        gap = self.detect_gap(opens, closes, highs, lows)
        
        # Find swing levels
        swings = self.find_swing_levels(
            highs[-self.swing_lookback:],
            lows[-self.swing_lookback:],
            self.swing_lookback
        )
        
        # Detect 1-2-3 pattern
        pattern = self.find_123_pattern(highs, lows, closes)
        
        # Volume confirmation
        volume_stats = self.get_volume_stats(volumes, self.volume_lookback)
        volume_ratio = volume_stats['ratio']
        
        signal = None
        confidence = 50.0
        reasoning_parts = []
        
        # STRATEGY 1: Gap Fill Trade
        if gap['type'] is not None and gap['level'] is not None:
            gap_level = gap['level']
            distance_to_gap = abs(current_price - gap_level) / current_price * 100
            
            # Check if approaching gap fill
            if distance_to_gap < self.gap_fill_tolerance_pct:
                confidence += 15
                reasoning_parts.append(f"Gap fill at {gap_level:.2f}")
                
                # Gap Up -> Short on fill (mean reversion)
                if gap['type'] == 'up' and current_price <= gap_level * 1.001:
                    reasoning_parts.append(f"Gap up filled (size: {gap['size_pct']:.2f}%)")
                    confidence += 10
                    
                    # Add confidence for larger gaps
                    if gap['size_pct'] > 0.5:
                        confidence += 10
                        reasoning_parts.append("Large gap")
                    
                    # Volume confirmation
                    if volume_ratio > 1.3:
                        confidence += 10
                        reasoning_parts.append(f"Volume confirmation ({volume_ratio:.1f}x)")
                        
                    if confidence >= self.min_confidence:
                        # Stop above gap open
                        stop_loss = current_price + atr_val * self.atr_stop_mult
                        
                        # Target: Next swing low or VWAP
                        take_profit = min(swings.get('swing_lows', [vwap_val])[:1] or [vwap_val])
                        if take_profit >= current_price:
                            take_profit = current_price - atr_val * self.rr_ratio
                        
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
                                'gap_type': gap['type'],
                                'gap_level': gap_level,
                                'gap_size_pct': gap['size_pct'],
                                'volume_ratio': volume_ratio,
                                'regime': regime.value
                            }
                        )
                
                # Gap Down -> Long on fill (mean reversion)
                elif gap['type'] == 'down' and current_price >= gap_level * 0.999:
                    reasoning_parts.append(f"Gap down filled (size: {gap['size_pct']:.2f}%)")
                    confidence += 10
                    
                    if gap['size_pct'] > 0.5:
                        confidence += 10
                        reasoning_parts.append("Large gap")
                    
                    if volume_ratio > 1.3:
                        confidence += 10
                        reasoning_parts.append(f"Volume confirmation ({volume_ratio:.1f}x)")
                        
                    if confidence >= self.min_confidence:
                        stop_loss = current_price - atr_val * self.atr_stop_mult
                        
                        take_profit = max(swings.get('swing_highs', [vwap_val])[:1] or [vwap_val])
                        if take_profit <= current_price:
                            take_profit = current_price + atr_val * self.rr_ratio
                        
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
                                'gap_type': gap['type'],
                                'gap_level': gap_level,
                                'gap_size_pct': gap['size_pct'],
                                'volume_ratio': volume_ratio,
                                'regime': regime.value
                            }
                        )
        
        # STRATEGY 2: 1-2-3 Pattern Breakout
        if signal is None and pattern['type'] is not None:
            entry_level = pattern.get('entry_level')
            
            if pattern['type'] == 'bullish' and entry_level:
                # Check for breakout above point 2
                if current_price > entry_level and closes[-2] <= entry_level:
                    confidence += 25
                    reasoning_parts.append(f"1-2-3 Bullish breakout at {entry_level:.2f}")
                    
                    # Volume confirmation
                    if volume_ratio > 1.5:
                        confidence += 15
                        reasoning_parts.append(f"Strong volume ({volume_ratio:.1f}x)")
                        
                    if confidence >= self.min_confidence:
                        stop_loss = pattern['point3'] - atr_val * 0.5  # Below point 3
                        take_profit = self.calculate_take_profit(
                            current_price, stop_loss, self.rr_ratio, 'long'
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
                            trailing_stop_pct=self.trailing_stop_pct,
                            reasoning=" | ".join(reasoning_parts),
                            metadata={
                                'pattern': '1-2-3 bullish',
                                'point1': pattern['point1'],
                                'point2': pattern['point2'],
                                'point3': pattern['point3'],
                                'volume_ratio': volume_ratio,
                                'regime': regime.value
                            }
                        )
                        
            elif pattern['type'] == 'bearish' and entry_level:
                # Check for breakdown below point 2
                if current_price < entry_level and closes[-2] >= entry_level:
                    confidence += 25
                    reasoning_parts.append(f"1-2-3 Bearish breakdown at {entry_level:.2f}")
                    
                    if volume_ratio > 1.5:
                        confidence += 15
                        reasoning_parts.append(f"Strong volume ({volume_ratio:.1f}x)")
                        
                    if confidence >= self.min_confidence:
                        stop_loss = pattern['point3'] + atr_val * 0.5  # Above point 3
                        take_profit = self.calculate_take_profit(
                            current_price, stop_loss, self.rr_ratio, 'short'
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
                            trailing_stop_pct=self.trailing_stop_pct,
                            reasoning=" | ".join(reasoning_parts),
                            metadata={
                                'pattern': '1-2-3 bearish',
                                'point1': pattern['point1'],
                                'point2': pattern['point2'],
                                'point3': pattern['point3'],
                                'volume_ratio': volume_ratio,
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
            'gap_threshold_pct': self.gap_threshold_pct,
            'swing_lookback': self.swing_lookback,
            'liquidity_cluster_bars': self.liquidity_cluster_bars,
            'gap_fill_tolerance_pct': self.gap_fill_tolerance_pct,
            'atr_stop_mult': self.atr_stop_mult,
            'rr_ratio': self.rr_ratio,
            'trailing_stop_pct': self.trailing_stop_pct
        })
        return base
