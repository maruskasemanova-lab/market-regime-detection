import re

with open('/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_trade_impl.py', 'r') as f:
    content = f.read()

# Top level insertions
content = content.replace(
    'def _to_finite_float(value: Any) -> Optional[float]:',
    '''class TradeExecutionEngine:
    """Encapsulates position sizing, risk management, and trade lifecycle."""
    def __init__(self, manager: Any):
        self.manager = manager

def _to_finite_float(value: Any) -> Optional[float]:'''
)

lines = content.split('\n')
new_lines = []
in_trade_func = False

for line in lines:
    if line.startswith('def trade_'):
        in_trade_func = True
        
        # Determine if it needs @staticmethod
        if '(self' not in line:
            new_lines.append('    @staticmethod')
            line = '    def ' + line[10:]
        else:
            line = '    def ' + line[10:]
            
        new_lines.append(line)
    elif in_trade_func and (line.startswith(' ') or line == ''):
        # Indent everything inside trade functions by 4 spaces
        if line == '':
            new_lines.append('')
        else:
            new_lines.append('    ' + line)
    else:
        in_trade_func = False
        new_lines.append(line)

content = '\n'.join(new_lines)

# Method calls replacements
replacements = {
    'self.trading_costs': 'self.manager.trading_costs',
    'self.exit_engine': 'self.manager.exit_engine',
    'self._calculate_order_flow_metrics': 'self.manager._calculate_order_flow_metrics',
    'self._calculate_indicators': 'self.manager._calculate_indicators',
    'self._latest_indicator_value': 'self.manager._latest_indicator_value',
    'self.ticker_params': 'self.manager.ticker_params',
    'self._normalize_stop_loss_mode': 'self.manager.config_service.normalize_stop_loss_mode',
    'self._extract_raw_confidence_from_metadata': 'self.manager._extract_raw_confidence_from_metadata',
    'self._extract_confirming_source_keys_from_metadata': 'self.manager._extract_confirming_source_keys_from_metadata',
    'self._to_float': 'self.manager._safe_float',
    'getattr(self, "consecutive_loss_limit"': 'getattr(self.manager, "consecutive_loss_limit"',
    'getattr(self, "consecutive_loss_cooldown_bars"': 'getattr(self.manager, "consecutive_loss_cooldown_bars"',
    
    # Internal method renames
    'self._extract_confirming_sources': 'self.extract_confirming_sources',
    'self._trailing_multiplier': 'self.trailing_multiplier',
    'self._calculate_position_size': 'self.calculate_position_size',
    'self._effective_trailing_stop_pct': 'self.effective_trailing_stop_pct',
    'self._resolve_stop_loss_for_entry': 'self.resolve_stop_loss_for_entry',
    'self._simulate_entry_fill': 'self.simulate_entry_fill',
    'self._fixed_stop_price': 'self.fixed_stop_price',
    'self._partial_take_profit_price': 'self.partial_take_profit_price',
    'self._build_trade_record': 'self.build_trade_record',
}

for old, new in replacements.items():
    content = content.replace(old, new)

with open('/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_trade_impl.py', 'w') as f:
    f.write(content)

print("Refactored day_trading_trade_impl.py successfully.")
