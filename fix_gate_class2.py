import re

with open('/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_gate_impl.py', 'r') as f:
    text = f.read()

old_init = '''def gate_evaluate_momentum_diversification_gate_candidate(
    self,
    *,'''

new_init = '''class GateEvaluationEngine:
    def __init__(self, exit_engine, config_service, evidence_service, default_momentum_strategies):
        self.exit_engine = exit_engine
        self.config_service = config_service
        self.evidence_service = evidence_service
        self.default_momentum_strategies = default_momentum_strategies

    def evaluate_momentum_diversification_gate_candidate(
        self,
        *,'''

text = text.replace(old_init, new_init)

# Now we need to fix the indentation for the whole file, since it was exported as module-level functions initially.
lines = text.split('\n')
new_lines = []

in_class = False
for line in lines:
    if line.startswith('class GateEvaluationEngine:'):
        in_class = True
        new_lines.append(line)
        continue
        
    if in_class:
        if line.strip() == '':
            new_lines.append(line)
        elif line.startswith('    def __init__') or line.startswith('        self.') or line.startswith('    def evaluate_momentum'):
            new_lines.append(line)
        else:
            new_lines.append('    ' + line)
    else:
        new_lines.append(line)

final_text = '\n'.join(new_lines)
final_text = final_text.replace('    def gate_passes_momentum_diversification_gate(', '    def passes_momentum_diversification_gate(')
final_text = final_text.replace('    def gate_should_momentum_fail_fast_exit(', '    def should_momentum_fail_fast_exit(')
final_text = final_text.replace('    def gate_time_of_day_threshold_boost(', '    @staticmethod\n    def time_of_day_threshold_boost(')
final_text = final_text.replace('    def gate_cross_asset_headwind_threshold_boost(', '    @staticmethod\n    def cross_asset_headwind_threshold_boost(')
final_text = final_text.replace('    def gate_build_position_closed_payload(', '    def build_position_closed_payload(')
final_text = final_text.replace('    def gate_bar_has_l2_data(', '    @staticmethod\n    def bar_has_l2_data(')
final_text = final_text.replace('    def gate_passes_l2_confirmation(', '    def passes_l2_confirmation(')

with open('/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_gate_impl.py', 'w') as f:
    f.write(final_text)
