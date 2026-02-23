import json
import re

file_path = '/Users/hotovo/.gemini/antigravity/scratch/market_regime_detection/src/day_trading_runtime_impl.py'

# let's grep 'sweep_detected' in this file to see where it is inserted
import subprocess
print(subprocess.check_output(['grep', '-n', '-A', '10', '-B', '10', 'runtime_detect_liquidity_sweep', file_path]).decode())

