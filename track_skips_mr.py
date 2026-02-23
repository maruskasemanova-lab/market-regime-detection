import os
log_path = "/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/api.log"
if os.path.exists(log_path):
    with open(log_path) as f:
        print("API log found")
        content = f.read()
        if "20:09" in content:
            print("Found 20:09")
else:
    print("No api.log")
    
import sqlite3
db_paths = [
    "/Users/hotovo/.gemini/antigravity/scratch/backtest-runner/data/backtest_runner.db",
    "/Users/hotovo/.gemini/antigravity/scratch/backtest-runner/backtest_runner.db"
]
for p in db_paths:
    if os.path.exists(p):
        print("Found DB:", p)
        conn = sqlite3.connect(p)
        c = conn.cursor()
        c.execute("SELECT run_id, report_json FROM run_reports ORDER BY created_at DESC LIMIT 1")
        row = c.fetchone()
        if row:
            run_id, data = row
            print("Latest DB run:", run_id)
            import json
            j = json.loads(data)
            if "trades" in j:
                print("Trades:", len(j["trades"]))
