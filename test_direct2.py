import sys
import os
from datetime import datetime, timedelta

from src.day_trading_manager import DayTradingManager
from src.day_trading_models import TradingConfig

manager = DayTradingManager()
run_id = "test_run_1"
ticker = "MU"
config = TradingConfig(
    liquidity_sweep_detection_enabled=True,
    l2_requested=True
)

manager.active_runs[run_id] = {"id": run_id, "config": config}

base_time = datetime.fromisoformat("2026-02-20T14:30:00+00:00")
for i in range(200):
    t = base_time + timedelta(minutes=i)
    bar_data = {
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
        "volume": 1000, "vwap": 100.25,
        "l2_delta": -50, "l2_buy_volume": 400, "l2_sell_volume": 600, "l2_volume": 1000,
        "l2_imbalance": -0.2, "l2_bid_depth_total": 5000, "l2_ask_depth_total": 4500,
        "l2_book_pressure": -0.1, "l2_book_pressure_change": -0.05,
        "l2_iceberg_buy_count": 0, "l2_iceberg_sell_count": 0, "l2_iceberg_bias": 0.0,
        "l2_quality_flags": [], "l2_quality": {}
    }
    # Create artificial sweep conditions on bar 100
    if i == 100:
        bar_data["low"] = 98.0  # Sweep low
        bar_data["close"] = 100.0
        bar_data["l2_delta"] = 5000 # Strong buying
        bar_data["l2_imbalance"] = 0.8
        bar_data["l2_book_pressure"] = 0.5
        
    res = manager.process_bar(
        run_id=run_id,
        ticker=ticker,
        timestamp=t,
        bar_data=bar_data
    )
    if 'liquidity_sweep' in res:
        sw = res['liquidity_sweep']
        if sw.get('sweep_detected') or sw.get('reason') != 'disabled':
            print(f"Bar {i} Sweep: {sw}")

