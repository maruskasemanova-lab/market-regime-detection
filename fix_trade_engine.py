with open('/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_trade_impl.py', 'r') as f:
    text = f.read()

# Replace init
old_init = '''    def __init__(self, manager: Any):
        self.manager = manager'''

new_init = '''    def __init__(self, config_service, evidence_service, exit_engine, ticker_params, get_session_key):
        self.config_service = config_service
        self.evidence_service = evidence_service
        self.exit_engine = exit_engine
        self.ticker_params = ticker_params
        self.get_session_key = get_session_key'''

text = text.replace(old_init, new_init)

with open('/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_trade_impl.py', 'w') as f:
    f.write(text)
