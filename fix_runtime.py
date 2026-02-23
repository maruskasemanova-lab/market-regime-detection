with open('/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_runtime_impl.py', 'r') as f:
    text = f.read()

# Replace manager proxy calls with direct engine calls
replacements = {
    'self._extract_raw_confidence_from_metadata(': 'self.evidence_service.extract_raw_confidence_from_metadata(',
    'self._evaluate_momentum_diversification_gate_candidate(': 'self.gate_engine.evaluate_momentum_diversification_gate_candidate(',
    'self._passes_momentum_diversification_gate(': 'self.gate_engine.passes_momentum_diversification_gate(',
    'self._should_momentum_fail_fast_exit(': 'self.gate_engine.should_momentum_fail_fast_exit(',
    'self._time_of_day_threshold_boost(': 'self.gate_engine.time_of_day_threshold_boost(',
    'self._cross_asset_headwind_threshold_boost(': 'self.gate_engine.cross_asset_headwind_threshold_boost(',
    'self._build_position_closed_payload(': 'self.gate_engine.build_position_closed_payload(',
    'self._bar_has_l2_data(': 'self.gate_engine.bar_has_l2_data(',
    'self._passes_l2_confirmation(': 'self.gate_engine.passes_l2_confirmation(',
    'self._canonical_trading_config(': 'self.config_service.canonical_trading_config(',
    'self._build_strategy_formula_context(': 'self.strategy_evaluator.build_strategy_formula_context(',
    'self._evaluate_strategy_custom_formula(': 'self.strategy_evaluator.evaluate_strategy_custom_formula(',
    'self._generate_signal_with_overrides(': 'self.strategy_evaluator.generate_signal_with_overrides(',
    'self._resolve_stop_loss_for_entry(': 'self.trade_engine.resolve_stop_loss_for_entry(',
    'self._calculate_position_size(': 'self.trade_engine.calculate_position_size(',
    'self._simulate_fill(': 'self.trade_engine.simulate_fill(',
    'self._process_exits(': 'self.trade_engine.process_exits(',
    'self._build_trade_record(': 'self.trade_engine.build_trade_record(',
    'self._evaluate_intraday_levels_entry_quality(': 'self.evidence_service.evaluate_intraday_levels_entry_quality(',
}

for old, new in replacements.items():
    text = text.replace(old, new)


with open('/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_runtime_impl.py', 'w') as f:
    f.write(text)
