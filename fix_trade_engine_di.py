with open('/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_trade_impl.py', 'r') as f:
    text = f.read()

# Replace init
old_init = '''    def __init__(self, manager):
        self.manager = manager'''

new_init = '''    def __init__(self, config_service, evidence_service, exit_engine, ticker_params, get_session_key):
        self.config_service = config_service
        self.evidence_service = evidence_service
        self.exit_engine = exit_engine
        self.ticker_params = ticker_params
        self.get_session_key = get_session_key'''

text = text.replace(old_init, new_init)

# Replace manager attributes
replacements = {
    'self.manager.ticker_params': 'self.ticker_params',
    'self.manager._get_session_key': 'self.get_session_key',
    'self.manager.exit_engine': 'self.exit_engine',
    
    'self.manager._normalize_adaptive_config': 'self.config_service.normalize_adaptive_config',
    'self.manager._normalize_advanced_execution': 'self.config_service.normalize_advanced_execution',
    
    'self.manager._safe_float': 'self.evidence_service.safe_float',
    'self.manager._extract_raw_confidence_from_metadata': 'self.evidence_service.extract_raw_confidence_from_metadata',
    'self.manager._bars_held': 'self.evidence_service.bars_held',
}

for old, new in replacements.items():
    text = text.replace(old, new)


with open('/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_trade_impl.py', 'w') as f:
    f.write(text)
