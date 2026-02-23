filename = 'tests/test_intraday_levels_entry_quality_gate.py'
with open(filename, 'r') as f:
    text = f.read()

text = text.replace('from typing import Any, Dict, List, Tuple', 'from typing import Any, Dict, List, Tuple, Optional')

with open(filename, 'w') as f:
    f.write(text)
