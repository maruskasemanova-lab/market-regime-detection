import sys
import os

from src.day_trading_manager import DayTradingManager
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class MockConfig(BaseModel):
    run_id: str = "direct_test"
    ticker: str = "MU"
    liquidity_sweep_detection_enabled: bool = True
    intraday_levels_enabled: bool = True

manager = DayTradingManager()
manager.active_runs["direct_test"] = {"id": "direct_test"} # Hack to let process_bar pass if needed

from datetime import datetime, timedelta

base_time = datetime.fromisoformat("2026-02-20T14:30:00+00:00")
for i in range(20):
    t = base_time + timedelta(minutes=i)
    bar_data = {
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
        "volume": 1000, "vwap": 100.25,
        "l2_delta": 50, "l2_buy_volume": 600, "l2_sell_volume": 400, "l2_volume": 1000,
        "l2_imbalance": 0.2, "l2_bid_depth_total": 5000, "l2_ask_depth_total": 4500,
        "l2_book_pressure": 0.1, "l2_book_pressure_change": 0.05,
        "l2_iceberg_buy_count": 0, "l2_iceberg_sell_count": 0, "l2_iceberg_bias": 0.0,
        "l2_quality_flags": [], "l2_quality": {}
    }
    # Pass mock config inside process_bar kwargs
    # We will let manager handle it or inject config if needed.
    res = manager.process_bar(
        run_id="direct_test",
        ticker="MU",
        timestamp=t,
        bar_data=bar_data,
        config=MockConfig().dict()
    )
    if 'liquidity_sweep' in res:
        print(f"Bar {i} Sweep:", res['liquidity_sweep'])

