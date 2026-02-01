"""
Trailing Stop-Loss Manager with multiple stop types.
"""
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any


class StopType(Enum):
    """Types of stop-loss strategies."""
    FIXED = "FIXED"           # Fixed price
    PERCENT = "PERCENT"       # Percentage from entry
    ATR = "ATR"               # ATR-based
    CHANDELIER = "CHANDELIER" # Highest high - N*ATR
    BREAKEVEN = "BREAKEVEN"   # Move to breakeven after X% profit
    PARABOLIC = "PARABOLIC"   # Accelerating stop (SAR-like)


@dataclass
class TrailingStopConfig:
    """Configuration for trailing stop."""
    stop_type: StopType
    initial_stop_pct: float = 2.0      # Initial stop percentage
    trailing_pct: float = 1.0          # Trailing percentage
    atr_multiplier: float = 2.0        # ATR multiplier
    breakeven_trigger_pct: float = 1.0 # When to move to breakeven
    acceleration_factor: float = 0.02  # For parabolic
    max_acceleration: float = 0.2      # Max acceleration for parabolic


class TrailingStopManager:
    """
    Manages trailing stop-loss for positions.
    Supports multiple stop types and dynamic adjustments.
    """
    
    def __init__(self, config: Optional[TrailingStopConfig] = None):
        self.config = config or TrailingStopConfig(stop_type=StopType.PERCENT)
        self.af = self.config.acceleration_factor  # Current acceleration (for parabolic)
        
    def calculate_initial_stop(
        self,
        entry_price: float,
        side: str,
        atr: Optional[float] = None,
        highest_high: Optional[float] = None,
        lowest_low: Optional[float] = None
    ) -> float:
        """
        Calculate initial stop-loss price.
        
        Args:
            entry_price: Entry price
            side: 'long' or 'short'
            atr: Current ATR value (for ATR-based stops)
            highest_high: Recent highest high (for Chandelier)
            lowest_low: Recent lowest low (for Chandelier)
        """
        if self.config.stop_type == StopType.FIXED:
            return self._fixed_stop(entry_price, side)
            
        elif self.config.stop_type == StopType.PERCENT:
            return self._percent_stop(entry_price, side, self.config.initial_stop_pct)
            
        elif self.config.stop_type == StopType.ATR:
            if atr is None:
                return self._percent_stop(entry_price, side, self.config.initial_stop_pct)
            return self._atr_stop(entry_price, side, atr)
            
        elif self.config.stop_type == StopType.CHANDELIER:
            if highest_high is None or lowest_low is None or atr is None:
                return self._percent_stop(entry_price, side, self.config.initial_stop_pct)
            return self._chandelier_stop(side, atr, highest_high, lowest_low)
            
        elif self.config.stop_type == StopType.BREAKEVEN:
            return self._percent_stop(entry_price, side, self.config.initial_stop_pct)
            
        elif self.config.stop_type == StopType.PARABOLIC:
            # Initial parabolic stop same as percent
            self.af = self.config.acceleration_factor  # Reset AF
            return self._percent_stop(entry_price, side, self.config.initial_stop_pct)
            
        return self._percent_stop(entry_price, side, self.config.initial_stop_pct)
    
    def update_trailing_stop(
        self,
        current_price: float,
        current_stop: float,
        entry_price: float,
        side: str,
        highest_price: float,
        lowest_price: float,
        atr: Optional[float] = None,
        highest_high: Optional[float] = None,
        lowest_low: Optional[float] = None
    ) -> float:
        """
        Update trailing stop based on price movement.
        
        Returns:
            New stop price (never moves against the position)
        """
        if self.config.stop_type == StopType.FIXED:
            return current_stop
            
        elif self.config.stop_type == StopType.PERCENT:
            return self._update_percent_trailing(
                current_price, current_stop, side, highest_price, lowest_price
            )
            
        elif self.config.stop_type == StopType.ATR:
            if atr:
                return self._update_atr_trailing(
                    current_price, current_stop, side, atr
                )
            return current_stop
            
        elif self.config.stop_type == StopType.CHANDELIER:
            if all([atr, highest_high, lowest_low]):
                return self._update_chandelier_trailing(
                    current_stop, side, atr, highest_high, lowest_low
                )
            return current_stop
            
        elif self.config.stop_type == StopType.BREAKEVEN:
            return self._update_breakeven_trailing(
                current_price, current_stop, entry_price, side
            )
            
        elif self.config.stop_type == StopType.PARABOLIC:
            return self._update_parabolic_trailing(
                current_price, current_stop, side, highest_price, lowest_price
            )
            
        return current_stop
    
    def _fixed_stop(self, entry: float, side: str) -> float:
        """Fixed stop - never trails."""
        pct = self.config.initial_stop_pct / 100
        if side == 'long':
            return entry * (1 - pct)
        return entry * (1 + pct)
    
    def _percent_stop(self, price: float, side: str, pct: float) -> float:
        """Percentage-based stop."""
        pct_decimal = pct / 100
        if side == 'long':
            return price * (1 - pct_decimal)
        return price * (1 + pct_decimal)
    
    def _atr_stop(self, price: float, side: str, atr: float) -> float:
        """ATR-based stop."""
        distance = atr * self.config.atr_multiplier
        if side == 'long':
            return price - distance
        return price + distance
    
    def _chandelier_stop(
        self, side: str, atr: float,
        highest_high: float, lowest_low: float
    ) -> float:
        """Chandelier exit stop."""
        distance = atr * self.config.atr_multiplier
        if side == 'long':
            return highest_high - distance
        return lowest_low + distance
    
    def _update_percent_trailing(
        self,
        current_price: float,
        current_stop: float,
        side: str,
        highest_price: float,
        lowest_price: float
    ) -> float:
        """Update percentage trailing stop."""
        pct = self.config.trailing_pct / 100
        
        if side == 'long':
            # Trail from highest price
            new_stop = highest_price * (1 - pct)
            return max(current_stop, new_stop)  # Never lower the stop
        else:
            # Trail from lowest price
            new_stop = lowest_price * (1 + pct)
            return min(current_stop, new_stop) if current_stop > 0 else new_stop
    
    def _update_atr_trailing(
        self,
        current_price: float,
        current_stop: float,
        side: str,
        atr: float
    ) -> float:
        """Update ATR trailing stop."""
        distance = atr * self.config.atr_multiplier
        
        if side == 'long':
            new_stop = current_price - distance
            return max(current_stop, new_stop)
        else:
            new_stop = current_price + distance
            return min(current_stop, new_stop) if current_stop > 0 else new_stop
    
    def _update_chandelier_trailing(
        self,
        current_stop: float,
        side: str,
        atr: float,
        highest_high: float,
        lowest_low: float
    ) -> float:
        """Update Chandelier trailing stop."""
        distance = atr * self.config.atr_multiplier
        
        if side == 'long':
            new_stop = highest_high - distance
            return max(current_stop, new_stop)
        else:
            new_stop = lowest_low + distance
            return min(current_stop, new_stop) if current_stop > 0 else new_stop
    
    def _update_breakeven_trailing(
        self,
        current_price: float,
        current_stop: float,
        entry_price: float,
        side: str
    ) -> float:
        """Move stop to breakeven after profit threshold."""
        trigger_pct = self.config.breakeven_trigger_pct / 100
        
        if side == 'long':
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct >= trigger_pct and current_stop < entry_price:
                # Add small buffer above breakeven
                return entry_price * 1.001
            return current_stop
        else:
            profit_pct = (entry_price - current_price) / entry_price
            if profit_pct >= trigger_pct and (current_stop > entry_price or current_stop == 0):
                return entry_price * 0.999
            return current_stop
    
    def _update_parabolic_trailing(
        self,
        current_price: float,
        current_stop: float,
        side: str,
        highest_price: float,
        lowest_price: float
    ) -> float:
        """
        Parabolic SAR-like trailing stop.
        Accelerates as price moves in favor.
        """
        if side == 'long':
            # Check if we have new high
            if current_price > highest_price:
                self.af = min(
                    self.af + self.config.acceleration_factor,
                    self.config.max_acceleration
                )
            
            # SAR formula: new_sar = old_sar + af * (ep - old_sar)
            new_stop = current_stop + self.af * (highest_price - current_stop)
            
            # Never let stop go above current price
            new_stop = min(new_stop, current_price * 0.99)
            
            return max(current_stop, new_stop)
        else:
            if current_price < lowest_price:
                self.af = min(
                    self.af + self.config.acceleration_factor,
                    self.config.max_acceleration
                )
            
            new_stop = current_stop - self.af * (current_stop - lowest_price)
            new_stop = max(new_stop, current_price * 1.01)
            
            return min(current_stop, new_stop) if current_stop > 0 else new_stop
    
    def check_stop_hit(self, current_price: float, stop_price: float, side: str) -> bool:
        """Check if stop is triggered."""
        if side == 'long':
            return current_price <= stop_price
        return current_price >= stop_price
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration."""
        return {
            'stop_type': self.config.stop_type.value,
            'initial_stop_pct': self.config.initial_stop_pct,
            'trailing_pct': self.config.trailing_pct,
            'atr_multiplier': self.config.atr_multiplier,
            'breakeven_trigger_pct': self.config.breakeven_trigger_pct,
            'acceleration_factor': self.config.acceleration_factor,
            'max_acceleration': self.config.max_acceleration,
            'current_af': self.af
        }
