"""Trend and volatility indicator helpers for regime detection."""

from __future__ import annotations

from typing import List, Optional

from ..day_trading_models import BarData

def regime_calc_trend_efficiency(self, bars: List[BarData]) -> float:
    """Calculate trend efficiency (net move / total move)."""
    if len(bars) < 5:
        return 0.0
    
    closes = [b.close for b in bars[-min(len(bars), 30):]]
    net_move = abs(closes[-1] - closes[0])
    total_move = sum(abs(closes[i] - closes[i-1]) for i in range(1, len(closes)))
    
    if total_move == 0:
        return 0.0
    
    return round(net_move / total_move, 3)

def regime_calc_volatility(self, bars: List[BarData]) -> float:
    """Calculate volatility as standard deviation of returns in percent."""
    if len(bars) < 5:
        return 0.0
    
    closes = [b.close for b in bars[-min(len(bars), 30):]]
    returns = [(closes[i] - closes[i-1]) / closes[i-1] * 100 for i in range(1, len(closes))]
    
    avg_return = sum(returns) / len(returns)
    variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
    
    return round(variance ** 0.5, 3)

def regime_calc_adx(self, bars: List[BarData], period: int = 14) -> Optional[float]:
    """Calculate ADX from bar data. Returns None during warmup."""
    if len(bars) < period * 2:
        return None
        
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    closes = [b.close for b in bars]
    
    tr_list = []
    dm_plus_list = []
    dm_minus_list = []
    
    for i in range(1, len(bars)):
        # True Range
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i-1])
        lc = abs(lows[i] - closes[i-1])
        tr = max(hl, hc, lc)
        tr_list.append(tr)
        
        # Directional Movement
        up_move = highs[i] - highs[i-1]
        down_move = lows[i-1] - lows[i]
        
        if up_move > down_move and up_move > 0:
            dm_plus_list.append(up_move)
        else:
            dm_plus_list.append(0.0)
            
        if down_move > up_move and down_move > 0:
            dm_minus_list.append(down_move)
        else:
            dm_minus_list.append(0.0)
    
    # Need enough data for Wilder's smoothing
    if len(tr_list) < period:
        return None
        
    # Initial smoothed averages (using simple average for first period)
    tr_smooth = sum(tr_list[:period])
    dm_plus_smooth = sum(dm_plus_list[:period])
    dm_minus_smooth = sum(dm_minus_list[:period])
    
    adx_values = []
    
    # Calculate smoothed values for the rest
    for i in range(period, len(tr_list)):
        tr_smooth = tr_smooth - (tr_smooth / period) + tr_list[i]
        dm_plus_smooth = dm_plus_smooth - (dm_plus_smooth / period) + dm_plus_list[i]
        dm_minus_smooth = dm_minus_smooth - (dm_minus_smooth / period) + dm_minus_list[i]
        
        if tr_smooth == 0:
            di_plus = 0
            di_minus = 0
        else:
            di_plus = 100 * (dm_plus_smooth / tr_smooth)
            di_minus = 100 * (dm_minus_smooth / tr_smooth)
        
        dx = 0
        if di_plus + di_minus > 0:
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        
        # ADX Smoothing
        if len(adx_values) == 0:
            adx_values.append(dx) # Initial ADX is just DX
        else:
            # Wilder's smoothing for ADX
            prev_adx = adx_values[-1]
            adx = (prev_adx * (period - 1) + dx) / period
            adx_values.append(adx)
            
    return round(adx_values[-1], 2) if adx_values else None

def regime_calc_adx_series(self, bars: List[BarData], period: int = 14) -> List[float]:
    """Calculate ADX series incrementally (point-in-time, no look-ahead bias).
    
    Returns a list of ADX values, one for each bar, where each value
    only uses data up to that point in time.
    """
    if len(bars) < 2:
        return [None] * len(bars)
        
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    closes = [b.close for b in bars]
    
    # Pre-fill with None for bars that don't have enough data
    adx_series: List[Optional[float]] = [None] * len(bars)
    
    tr_list = []
    dm_plus_list = []
    dm_minus_list = []
    
    # First bar has no ADX
    for i in range(1, len(bars)):
        # True Range
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i-1])
        lc = abs(lows[i] - closes[i-1])
        tr = max(hl, hc, lc)
        tr_list.append(tr)
        
        # Directional Movement
        up_move = highs[i] - highs[i-1]
        down_move = lows[i-1] - lows[i]
        
        if up_move > down_move and up_move > 0:
            dm_plus_list.append(up_move)
        else:
            dm_plus_list.append(0.0)
            
        if down_move > up_move and down_move > 0:
            dm_minus_list.append(down_move)
        else:
            dm_minus_list.append(0.0)
        
        # Need at least period bars of TR data
        if len(tr_list) < period:
            continue
        
        # Calculate ADX at this point using data up to bar i
        if len(tr_list) == period:
            # Initial smoothed averages
            tr_smooth = sum(tr_list[:period])
            dm_plus_smooth = sum(dm_plus_list[:period])
            dm_minus_smooth = sum(dm_minus_list[:period])
            
            if tr_smooth > 0:
                di_plus = 100 * (dm_plus_smooth / tr_smooth)
                di_minus = 100 * (dm_minus_smooth / tr_smooth)
            else:
                di_plus = 0.0
                di_minus = 0.0
            
            if di_plus + di_minus > 0:
                dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
            else:
                dx = 0.0
            
            adx_series[i] = round(dx, 2)
        else:
            # Wilder's smoothing
            j = len(tr_list) - 1  # Current index in tr_list
            start_idx = j - period
            
            # Calculate smoothed values up to this point
            tr_smooth = sum(tr_list[start_idx:start_idx + period])
            dm_plus_smooth = sum(dm_plus_list[start_idx:start_idx + period])
            dm_minus_smooth = sum(dm_minus_list[start_idx:start_idx + period])
            
            # Apply Wilder's smoothing for remaining bars
            for k in range(start_idx + period, j + 1):
                tr_smooth = tr_smooth - (tr_smooth / period) + tr_list[k]
                dm_plus_smooth = dm_plus_smooth - (dm_plus_smooth / period) + dm_plus_list[k]
                dm_minus_smooth = dm_minus_smooth - (dm_minus_smooth / period) + dm_minus_list[k]
            
            if tr_smooth > 0:
                di_plus = 100 * (dm_plus_smooth / tr_smooth)
                di_minus = 100 * (dm_minus_smooth / tr_smooth)
            else:
                di_plus = 0.0
                di_minus = 0.0
            
            if di_plus + di_minus > 0:
                dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
            else:
                dx = 0.0
            
            # Use previous ADX for smoothing
            prev_adx = adx_series[i - 1]
            if prev_adx is not None and prev_adx > 0:
                adx = (prev_adx * (period - 1) + dx) / period
            else:
                adx = dx
            
            adx_series[i] = round(adx, 2)
    
    return adx_series
