from datetime import datetime, timezone
import json
from src.day_trading_models import TradingSession, BarData, Signal, SignalType
from src.day_trading_runtime_intrabar import calculate_intrabar_1s_snapshot

ts = datetime(2026, 2, 12, 15, 32, tzinfo=timezone.utc)
quotes = [
    {"s": 1, "bid": 99.99, "ask": 100.01},  # 100.00
    {"s": 2, "bid": 100.00, "ask": 100.02}, # 100.01
    {"s": 3, "bid": 100.01, "ask": 100.03}, # 100.02
    {"s": 4, "bid": 100.02, "ask": 100.04}, # 100.03
    {"s": 5, "bid": 100.01, "ask": 100.03}, # 100.02
    {"s": 6, "bid": 100.00, "ask": 100.02}, # 100.01
    {"s": 7, "bid": 100.01, "ask": 100.03}, # 100.02
    {"s": 8, "bid": 100.02, "ask": 100.04}, # 100.03
    {"s": 9, "bid": 100.03, "ask": 100.05}, # 100.04
]
bar = BarData(
    timestamp=ts,
    open=100.0,
    high=100.0,
    low=100.0,
    close=100.0,
    volume=100,
    intrabar_quotes_1s=quotes,
)

snapshot = calculate_intrabar_1s_snapshot(bar, intrabar_window_seconds=10)
print(json.dumps(snapshot, indent=2))
