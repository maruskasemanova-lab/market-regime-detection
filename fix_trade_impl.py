import re

with open('/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_trade_impl.py', 'r') as f:
    original_code = f.read()

lines = original_code.splitlines()

part1 = lines[:140]
part2 = lines[140:]

# Indent part2
part2_indented = []
for line in part2:
    if line == '':
        part2_indented.append('')
    else:
        part2_indented.append('    ' + line)

content2 = '\n'.join(part2_indented)

# Remove 'trade_' prefix from method definitions
content2 = re.sub(r'^    def trade_([a-zA-Z0-9_]+)', r'    def \1', content2, flags=re.MULTILINE)

# Add @staticmethod to the pure functions
statics = ['extract_confirming_sources', 'agreement_risk_multiplier', 'trailing_multiplier', 'fixed_stop_price']
for s in statics:
    content2 = content2.replace(f'    def {s}(', f'    @staticmethod\n    def {s}(')

class_def = '''class TradeExecutionEngine:
    """Encapsulates position sizing, risk management, and trade lifecycle."""
    def __init__(self, manager: Any):
        self.manager = manager
'''

# Rejoin
content = '\n'.join(part1) + '\n\n' + class_def + '\n' + content2 + '\n'

# Fix references
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
