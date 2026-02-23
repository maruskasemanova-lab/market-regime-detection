filename = 'tests/test_intraday_levels_entry_quality_gate.py'
with open(filename, 'r') as f:
    text = f.read()

# Just ensure Optional is imported from typing at the top
if 'from typing import' in text and 'Optional' not in text:
    text = text.replace('from typing import', 'from typing import Optional,')
elif 'import typing' not in text and 'from typing import Optional' not in text:
    text = 'from typing import Optional\n' + text

with open(filename, 'w') as f:
    f.write(text)
