import os

target = "/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_runtime_impl.py"
with open(target, "r") as f:
    lines = f.readlines()

# Find start of block to replace
start_idx = -1
for i, line in enumerate(lines):
    if "def gen_signal_fn():" in line:
        start_idx = i
        break

# Find end of block 
end_idx = -1
for i in range(start_idx, len(lines)):
    line = lines[i]
    if "if decision.execute and decision.signal and passed_trade_threshold:" in line:
        end_idx = i
        break

if start_idx == -1 or end_idx == -1:
    print("Could not find start or end index. start:", start_idx, "end:", end_idx)
    exit(1)

new_code = """        # ── 5s Intrabar Checkpoint Loop ──────────────────────
        step = int(getattr(session.config, "intrabar_eval_step_seconds", 5)) if hasattr(session, "config") else 5
        raw_quotes = getattr(bar, "intrabar_quotes_1s", None)
        checkpoints_meta = []
        import copy
        if isinstance(raw_quotes, list) and raw_quotes:
            normalized_quotes = []
            for item in raw_quotes:
                if isinstance(item, dict):
                    try:
                        sec = int(float(item.get("s", 0) or 0))
                        if 0 <= sec <= 59:
                            normalized_quotes.append(dict(item, s=sec))
                    except (TypeError, ValueError):
                        pass
            normalized_quotes.sort(key=lambda x: int(x.get("s")))
            
            prefix_quotes = []
            last_sec = None
            for idx, quote in enumerate(normalized_quotes):
                sec = int(quote.get("s"))
                prefix_quotes.append(quote)
                if last_sec == sec:
                    prefix_quotes[-2] = quote if len(prefix_quotes) >= 2 else quote
                    if len(prefix_quotes) >= 2:
                        prefix_quotes.pop()
                    continue
                last_sec = sec
                
                is_boundary = (sec % step == 0) or (sec == 59) or (idx == len(normalized_quotes) - 1)
                if is_boundary:
                    cp_bar = copy.copy(bar)
                    cp_bar.intrabar_quotes_1s = [dict(r) for r in prefix_quotes]
                    cp_ts = timestamp.replace(second=sec, microsecond=0)
                    checkpoints_meta.append((cp_bar, cp_ts, sec))
                    
        if not checkpoints_meta:
            checkpoints_meta.append((bar, timestamp, 0))
            
        intrabar_eval_trace = {
            "schema_version": 1,
            "source": "intrabar_quote_checkpoints",
            "minute_timestamp": timestamp.replace(second=0, microsecond=0).isoformat(),
            "checkpoints": []
        }
        
        final_slice_res = None
        for cp_bar, cp_ts, sec in checkpoints_meta:
            slice_res = _runtime_evaluate_intrabar_slice_impl(self, session, cp_bar, cp_ts)
            
            cp_payload = {
                "timestamp": slice_res.get("timestamp"),
                "offset_sec": sec,
                "layer_scores": slice_res.get("layer_scores"),
                "intrabar_1s": _calculate_intrabar_1s_snapshot_impl(cp_bar),
                "provisional": True,
            }
            if "signal_rejected" in slice_res:
                cp_payload["signal_rejected"] = slice_res["signal_rejected"]
            if "candidate_diagnostics" in slice_res:
                cp_payload["candidate_diagnostics"] = slice_res["candidate_diagnostics"]
                
            intrabar_eval_trace["checkpoints"].append(cp_payload)
            final_slice_res = slice_res
            
            # If the strategy natively triggered a signal (passed base thresholds + evidence),
            # stop evaluating remaining intra-minute slices and proceed to the micro confirmation gates!
            if slice_res.get("_raw_signal"):
                break
                
        if intrabar_eval_trace["checkpoints"]:
            intrabar_eval_trace["checkpoints"][-1]["provisional"] = False
            intrabar_eval_trace["checkpoint_count"] = len(intrabar_eval_trace["checkpoints"])
            
        result["intrabar_eval_trace"] = intrabar_eval_trace
        if final_slice_res:
            if "layer_scores" in final_slice_res:
                result["layer_scores"] = final_slice_res["layer_scores"]
            if "signal_rejected" in final_slice_res:
                result["signal_rejected"] = final_slice_res["signal_rejected"]
            if "candidate_diagnostics" in final_slice_res:
                result["candidate_diagnostics"] = final_slice_res["candidate_diagnostics"]
        else:
            final_slice_res = {}

        # Synthesize a mock Decision object so the subsequent gate sequence behaves identically.
        _ls = final_slice_res.get("layer_scores", {})
        combined_score = float(_ls.get("combined_score", 0.0))
        effective_trade_threshold = float(_ls.get("threshold_used", 0.0))
        passed_trade_threshold = bool(_ls.get("passed", False))
        tod_boost = float(_ls.get("tod_threshold_boost", 0.0))
        headwind_boost = float(_ls.get("headwind_threshold_boost", 0.0))
        required_confirming_sources = int(_ls.get("required_confirming_sources", 2))
        
        class DecisionProxy:
            pass
        decision = DecisionProxy()
        decision.combined_score = combined_score
        decision.execute = bool(passed_trade_threshold)
        decision.signal = final_slice_res.get("_raw_signal")

"""

lines = lines[:start_idx] + [new_code] + lines[end_idx:]

with open(target, "w") as f:
    f.writelines(lines)

print("Replaced logic with Python successfully")
