import requests
import json
from datetime import datetime, timedelta

start_url = "http://localhost:8001/api/session/config"
bar_url = "http://localhost:8001/api/session/bar"

base_time = datetime.fromisoformat("2026-02-20T14:30:00+00:00") # 9:30 AM EST start

run_id = "test_session_bar_sweep_12"
ticker = "MU"

# Start Session
start_params = {
    "run_id": run_id,
    "ticker": ticker,
    "date": "2026-02-20",
    "liquidity_sweep_detection_enabled": True,
    "strategy_selection_mode": "all_enabled",
    "regime_filter_json": json.dumps([]),
    "market_open": "09:30",
    "market_close": "16:00",
}

resp = requests.post(start_url, params=start_params, headers={"x-internal-token": "test-token-123"})
if resp.status_code != 200:
    print("Start fail:", resp.text)
    exit(1)

# Post bars
for i in range(200):
    t = base_time + timedelta(minutes=i)
    payload = {
        "run_id": run_id,
        "ticker": ticker,
        "timestamp": t.isoformat(),
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000,
        "vwap": 100.25,
        "l2_delta": 50,
        "l2_buy_volume": 600,
        "l2_sell_volume": 400,
        "l2_volume": 1000,
        "l2_imbalance": 0.2,
        "l2_bid_depth_total": 5000,
        "l2_ask_depth_total": 4500,
        "l2_book_pressure": 0.1,
        "l2_book_pressure_change": 0.05,
        "l2_iceberg_buy_count": 0,
        "l2_iceberg_sell_count": 0,
        "l2_iceberg_bias": 0.0,
        "l2_quality_flags": [],
        "l2_quality": {},
        "intrabar_quotes_1s": []
    }
    
    # Try to trigger a sweep at bar 150
    if i == 150:
        # Give it strong bearish l2 aggression and bullish book pressure 
        payload["l2_delta"] = -5000
        payload["l2_book_pressure"] = 1.0  # high bullish proxy
    
    response = requests.post(bar_url, json=payload, headers={"x-internal-token": "test-token-123"})
    if response.status_code == 200:
        res = response.json()
        if res.get('action') == 'regime_detected':
             print(f"Bar {i} Regime Detected: {res.get('regime')} -> strategies: {res.get('strategies')}")
        # Only print sweep dict if it exists or when an action occurred
        if res.get('action') == 'trading' or res.get('liquidity_sweep'):
             print(f"Bar {i} Sweep: {res.get('liquidity_sweep')}")
    else:
        print(response.status_code, response.text)

print("Done posting.")
