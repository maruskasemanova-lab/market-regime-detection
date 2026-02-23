with open('/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_gate_impl.py', 'r') as f:
    original = f.read()

# Replace init
original = original.replace('    def __init__(self, manager):\n        self.manager = manager',
                            '    def __init__(self, exit_engine, default_momentum_strategies):\n        self.exit_engine = exit_engine\n        self.default_momentum_strategies = default_momentum_strategies')

# Look for self.manager. method calls and replace them or we handle them as static methods if appropriate
replacements = {
    'self.manager.exit_engine': 'self.exit_engine',
    'self.manager.DEFAULT_MOMENTUM_STRATEGIES': 'self.default_momentum_strategies',
}
for k, v in replacements.items():
    original = original.replace(k, v)

with open('/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_gate_impl.py', 'w') as f:
    f.write(original)
