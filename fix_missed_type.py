import re

filename = 'tests/test_intraday_levels_entry_quality_gate.py'
with open(filename, 'r') as f:
    text = f.read()

text = text.replace('dict | None = None', 'Optional[dict] = None')

with open(filename, 'w') as f:
    f.write(text)
