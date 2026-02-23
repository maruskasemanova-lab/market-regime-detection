with open('/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_gate_impl.py', 'r') as f:
    original = f.read()

lines = original.splitlines()

# find where imports end (first def)
first_def_idx = 0
for i, line in enumerate(lines):
    if line.startswith('def '):
        first_def_idx = i
        break

part1 = lines[:first_def_idx]
part2 = lines[first_def_idx:]

class_def = '''
class GateEvaluationEngine:
    """Evaluates various gates to confirm or reject trading signals."""
    def __init__(self, manager):
        self.manager = manager
'''

new_part2 = []
for line in part2:
    if line == '':
        new_part2.append('')
    else:
        new_part2.append('    ' + line)

content2 = '\n'.join(new_part2)

import re
content2 = re.sub(r'^    def gate_([a-zA-Z0-9_]+)', r'    def \1', content2, flags=re.MULTILINE)

statics = ['time_of_day_threshold_boost', 'cross_asset_headwind_threshold_boost', 'bar_has_l2_data']
for s in statics:
    content2 = content2.replace(f'    def {s}(', f'    @staticmethod\n    def {s}(')

content = '\n'.join(part1) + class_def + '\n' + content2 + '\n'

replacements = {
    'self.ticker_params': 'self.manager.ticker_params',
    'self.exit_engine': 'self.manager.exit_engine',
    'self._normalize_adaptive_config': 'self.manager._normalize_adaptive_config',
    'self._resolve_momentum_diversification': 'self.manager._resolve_momentum_diversification',
    'self._normalize_strategy_list': 'self.manager._normalize_strategy_list',
    'self._canonical_strategy_key': 'self.manager._canonical_strategy_key',
    'self._normalize_momentum_sleeve_id': 'self.manager._normalize_momentum_sleeve_id',
    'self._evaluate_momentum_diversification_gate_candidate': 'self.evaluate_momentum_diversification_gate_candidate',
    'self._calculate_order_flow_metrics': 'self.manager._calculate_order_flow_metrics',
    'self._select_momentum_sleeve': 'self.manager._select_momentum_sleeve',
    'self._to_float': 'self.manager._safe_float',
    'self.DEFAULT_MOMENTUM_STRATEGIES': 'self.manager.DEFAULT_MOMENTUM_STRATEGIES',
}

for old, new in replacements.items():
    content = content.replace(old, new)


with open('/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_gate_impl.py', 'w') as f:
    f.write(content)
