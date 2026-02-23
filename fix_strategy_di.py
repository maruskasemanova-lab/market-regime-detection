with open('/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_strategy_evaluator.py', 'r') as f:
    text = f.read()

# Replace init
old_init = '''    def __init__(self, manager: Any):
        self.manager = manager'''

new_init = '''    def __init__(self, evidence_service):
        self.evidence_service = evidence_service'''

text = text.replace(old_init, new_init)

# Replace manager attributes
replacements = {
    'self.manager._safe_float': 'self.evidence_service.safe_float',
    'self.manager._bars_held': 'self.evidence_service.bars_held',
    'self.manager._latest_indicator_value': 'self.evidence_service.latest_indicator_value',
}

for old, new in replacements.items():
    text = text.replace(old, new)


with open('/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_strategy_evaluator.py', 'w') as f:
    f.write(text)
