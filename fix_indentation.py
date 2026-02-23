import re

with open('/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_gate_impl.py', 'r') as f:
    text = f.read()

# Add the class definition back, then indent everything below it except the imports
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
        # If it's already indented by 4 spaces (like the init we just added), keep it
        elif line.startswith('    def __init__') or line.startswith('        self.') or line.startswith('    def evaluate_momentum'):
            new_lines.append(line)
        else:
            new_lines.append('    ' + line)
    else:
        new_lines.append(line)

with open('/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_gate_impl.py', 'w') as f:
    f.write('\n'.join(new_lines))
