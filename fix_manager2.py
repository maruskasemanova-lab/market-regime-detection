import re

with open('/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_manager.py', 'r') as f:
    text = f.read()

# I need to completely remove the proxy methods that were delegated directly in day_trading_runtime_impl.py
methods_to_remove = [
    '_extract_raw_confidence_from_metadata',
    '_evaluate_momentum_diversification_gate_candidate',
    '_passes_momentum_diversification_gate',
    '_should_momentum_fail_fast_exit',
    '_time_of_day_threshold_boost',
    '_cross_asset_headwind_threshold_boost',
    '_build_position_closed_payload',
    '_bar_has_l2_data',
    '_passes_l2_confirmation',
    '_canonical_trading_config',
    '_build_strategy_formula_context',
    '_evaluate_strategy_custom_formula',
    '_generate_signal_with_overrides',
    '_resolve_stop_loss_for_entry',
    '_calculate_position_size',
    '_simulate_fill',
    '_process_exits',
    '_build_trade_record',
    '_evaluate_intraday_levels_entry_quality',
    '_strategy_formula_fields',
]

def remove_method(text, method_name):
    # Match def _method_name(self...): taking into account typing until the next def or end of class
    pattern = r'    (?:@staticmethod\n    )?def ' + re.escape(method_name) + r'\b.*?(?=\n    (?:@staticmethod\n    )?def |\Z)'
    return re.sub(pattern, '', text, flags=re.DOTALL)

for method in methods_to_remove:
    text = remove_method(text, method)

with open('/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_manager.py', 'w') as f:
    f.write(text)
