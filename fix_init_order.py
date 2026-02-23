with open('/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_manager.py', 'r') as f:
    text = f.read()

# self.ticker_params is currently defined way down on line 285 but is needed at line 241
# Let's move self.ticker_params initialization earlier in __init__ just after self.strategies
ticker_params_block = '''        # Ticker-Specific Preferences (AOS Optimized)
        self.ticker_preferences = {
            "NVDA": {
                Regime.TRENDING: ['pullback', 'momentum', 'volume_profile'],
                Regime.CHOPPY: [],  # Skip choppy - losses historically
                Regime.MIXED: ['volume_profile', 'vwap_magnet']
            },
            "TSLA": {
                Regime.TRENDING: ['momentum', 'gap_liquidity'],
                Regime.CHOPPY: [],  # Skip choppy - high volatility losses
                Regime.MIXED: []  # Only trade clear trends
            },
            "AAPL": {
                Regime.TRENDING: ['mean_reversion', 'vwap_magnet', 'pullback'],
                Regime.CHOPPY: ['mean_reversion', 'vwap_magnet'],
                Regime.MIXED: ['mean_reversion', 'vwap_magnet']
            },
            "SPY": {
                Regime.TRENDING: ['momentum_flow', 'gap_liquidity'],
                Regime.CHOPPY: ['vwap_magnet', 'mean_reversion'],
                Regime.MIXED: ['vwap_magnet']
            },
            "QQQ": {
                Regime.TRENDING: ['momentum_flow', 'scalp_l2_intrabar', 'gap_liquidity'],
                Regime.CHOPPY: ['scalp_l2_intrabar', 'absorption_reversal', 'mean_reversion'],
                Regime.MIXED: ['scalp_l2_intrabar', 'vwap_magnet']
            }
        }

        # Initialize ticker_params dictionary
        self.ticker_params: Dict[str, Dict[str, Any]] = {}
        target_file = "/Users/hotovo/.gemini/antigravity/scratch/backtest-runner/aos_optimization/aos_config.json"
        
        # Load local settings mapping (if provided)
        try:
            self._load_aos_config(target_file)
        except Exception as e:
            logger.warning(f"Could not load AOS config from {target_file}: {e}")
'''

import re
# Remove the block from its current location
pattern = r'        # Ticker-Specific Preferences \(AOS Optimized\).*?logger\.warning\(f"Could not load AOS config from \{target_file\}: \{e\}"\)\n'
text = re.sub(pattern, '', text, flags=re.DOTALL)

# Insert it before trade_engine
insert_point = '        self.config_service = DayTradingConfigService(self)'
text = text.replace(insert_point, ticker_params_block + '\n' + insert_point)

with open('/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_manager.py', 'w') as f:
    f.write(text)
