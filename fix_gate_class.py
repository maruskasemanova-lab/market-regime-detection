with open('/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_gate_impl.py', 'r') as f:
    text = f.read()

# I accidentally stripped the GateEvaluationEngine class definition when fixing DI, because it was at the very top of those methods
if "class GateEvaluationEngine" not in text:
    old_init = '''def gate_evaluate_momentum_diversification_gate_candidate('''
    new_init = '''class GateEvaluationEngine:
    def __init__(self, exit_engine, config_service, evidence_service, default_momentum_strategies):
        self.exit_engine = exit_engine
        self.config_service = config_service
        self.evidence_service = evidence_service
        self.default_momentum_strategies = default_momentum_strategies

    def evaluate_momentum_diversification_gate_candidate('''
    text = text.replace(old_init, new_init)
    
    # fix the function defs
    text = text.replace('def gate_passes_momentum_diversification_gate(', '    def passes_momentum_diversification_gate(')
    text = text.replace('def gate_should_momentum_fail_fast_exit(', '    def should_momentum_fail_fast_exit(')
    text = text.replace('def gate_time_of_day_threshold_boost(', '    @staticmethod\n    def time_of_day_threshold_boost(')
    text = text.replace('def gate_cross_asset_headwind_threshold_boost(', '    @staticmethod\n    def cross_asset_headwind_threshold_boost(')
    text = text.replace('def gate_build_position_closed_payload(', '    def build_position_closed_payload(')
    text = text.replace('def gate_bar_has_l2_data(', '    @staticmethod\n    def bar_has_l2_data(')
    text = text.replace('def gate_passes_l2_confirmation(', '    def passes_l2_confirmation(')
    
with open('/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_gate_impl.py', 'w') as f:
    f.write(text)
