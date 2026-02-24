import os

target = "/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_runtime/intrabar_slice.py"
with open(target, "r") as f:
    content = f.read()

target_str = """        result["signal"] = signal.to_dict()

        cand_diag = signal.metadata.get("candidate_diagnostics")"""

replacement_str = """        result["signal"] = signal.to_dict()
        result["_raw_signal"] = signal

        cand_diag = signal.metadata.get("candidate_diagnostics")"""

if target_str in content:
    content = content.replace(target_str, replacement_str)
    with open(target, "w") as f:
        f.write(content)
    print("Patched intrabar_slice.py successfully.")
else:
    print("Could not find target_str in intrabar_slice.py")

