import re

with open('/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_gate_impl.py', 'r') as f:
    text = f.read()

# Replace config access
text = text.replace('self.manager._normalize_strategy_list', 'self.config_service.normalize_strategy_list')
text = text.replace('self.manager._normalize_momentum_sleeve_id', 'self.config_service.normalize_momentum_sleeve_id')
text = text.replace('self.manager.ticker_params', 'self.config_service.ticker_params')
text = text.replace('self.manager._normalize_adaptive_config', 'self.config_service.normalize_adaptive_config')
text = text.replace('self.manager._resolve_momentum_diversification', 'self.config_service.resolve_momentum_diversification')
text = text.replace('self.manager._select_momentum_sleeve', 'self.config_service.select_momentum_sleeve')

# Replace flow access
text = text.replace('self.manager._calculate_order_flow_metrics', 'self.evidence_service.calculate_order_flow_metrics')
text = text.replace('self.manager._canonical_strategy_key', 'self.evidence_service.canonical_strategy_key')
text = text.replace('self.manager._safe_float', 'self.evidence_service.safe_float')

# Fix init
old_init = '''    def __init__(self, manager):
        self.manager = manager'''

new_init = '''    def __init__(self, exit_engine, config_service, evidence_service, default_momentum_strategies):
        self.exit_engine = exit_engine
        self.config_service = config_service
        self.evidence_service = evidence_service
        self.default_momentum_strategies = default_momentum_strategies'''

text = text.replace(old_init, new_init)

# Replace remaining manager calls
text = text.replace('self.manager.exit_engine', 'self.exit_engine')
text = text.replace('self.manager.DEFAULT_MOMENTUM_STRATEGIES', 'self.default_momentum_strategies')

with open('/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_gate_impl.py', 'w') as f:
    f.write(text)
