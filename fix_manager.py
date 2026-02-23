with open('/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_manager.py', 'r') as f:
    text = f.read()

# Update the instantiation section
old_lines = '''        # Helper services: config normalization + evidence parsing.
        self.config_service = DayTradingConfigService(self)
        self.trade_engine = TradeExecutionEngine(self)
        self.strategy_evaluator = StrategyEvaluatorEngine(self)
        self.gate_engine = GateEvaluationEngine(self)
        self.evidence_service = DayTradingEvidenceService(
            canonical_strategy_key=self._canonical_strategy_key,
            safe_float=self._safe_float,
        )'''

new_lines = '''        # Helper services: config normalization + evidence parsing.
        self.config_service = DayTradingConfigService(self)
        self.evidence_service = DayTradingEvidenceService(
            canonical_strategy_key=self._canonical_strategy_key,
            safe_float=self._safe_float,
        )
        self.trade_engine = TradeExecutionEngine(
            config_service=self.config_service,
            evidence_service=self.evidence_service,
            exit_engine=self.exit_engine,
            ticker_params=self.ticker_params,
            get_session_key=self._get_session_key,
        )
        self.strategy_evaluator = StrategyEvaluatorEngine(
            evidence_service=self.evidence_service,
        )
        self.gate_engine = GateEvaluationEngine(
            exit_engine=self.exit_engine,
            config_service=self.config_service,
            evidence_service=self.evidence_service,
            default_momentum_strategies=self.DEFAULT_MOMENTUM_STRATEGIES,
        )'''

text = text.replace(old_lines, new_lines)

with open('/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_manager.py', 'w') as f:
    f.write(text)
