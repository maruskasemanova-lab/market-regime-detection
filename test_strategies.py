from src.day_trading_manager import DayTradingManager
manager = DayTradingManager()
print(list(manager.strategies.keys()))
for name, strat in manager.strategies.items():
    allowed = getattr(strat, 'allowed_regimes', [])
    print(f"{name}: enabled={getattr(strat, 'enabled', True)}, allowed={allowed}")
