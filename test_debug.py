import math
from typing import List

raw_quotes = [
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
intrabar_window_seconds = 10


rows = []
for item in raw_quotes:
    second = int(float(item.get("s", 0) or 0))
    bid = float(item.get("bid", 0.0) or 0.0)
    ask = float(item.get("ask", 0.0) or 0.0)
    mid = (bid + ask) / 2.0
    spread_bps = ((ask - bid) / mid * 10000.0) if mid > 0 else 0.0
    rows.append((second, mid, spread_bps))

print("Mids:", [r[1] for r in rows])

window_long_move_pct = 0.0
window_long_push_ratio = 0.0
window_max_coverage_points = 0
n_rows = len(rows)
cum_push = [0] * (n_rows + 1)
cum_pull = [0] * (n_rows + 1)
cum_abs = [0.0] * (n_rows + 1)

for idx in range(1, n_rows):
    step = rows[idx][1] - rows[idx - 1][1]
    push = 1 if step > 0 else 0
    pull = 1 if step < 0 else 0
    cum_push[idx + 1] = cum_push[idx] + push
    cum_pull[idx + 1] = cum_pull[idx] + pull
    cum_abs[idx + 1] = cum_abs[idx] + abs(step)

for i in range(n_rows):
    start_sec = rows[i][0]
    start_mid = rows[i][1]
    
    for j in range(i + 1, n_rows):
        end_sec = rows[j][0]
        if end_sec - start_sec > intrabar_window_seconds:
            break
            
        end_mid = rows[j][1]
        net_move = end_mid - start_mid
        move_pct = (net_move / start_mid) * 100.0
        
        p_hits = cum_push[j + 1] - cum_push[i + 1]
        n_hits = cum_pull[j + 1] - cum_pull[i + 1]
        tot_abs = cum_abs[j + 1] - cum_abs[i + 1]
        
        d_base = p_hits + n_hits
        push_ratio = (p_hits - n_hits) / d_base if d_base > 0 else 0.0
        
        print(f"i={i}, j={j}, move_pct={move_pct:.6f}, p_hits={p_hits}, n_hits={n_hits}, push_ratio={push_ratio}")
        
        if move_pct > window_long_move_pct:
            window_long_move_pct = move_pct
            window_long_push_ratio = push_ratio

print("FINAL:", window_long_move_pct, window_long_push_ratio)
