import glob

for filename in glob.glob('tests/test_day_trading_manager*.py'):
    with open(filename, 'r') as f:
        text = f.read()
    
    # Add import typing if missing
    if 'from typing import ' not in text and 'import typing' not in text:
        text = 'from typing import Any, Dict, List, Optional, Tuple\n' + text
    elif 'Optional' not in text and 'from typing import' in text:
        text = text.replace('from typing import ', 'from typing import Optional, ')
    
    with open(filename, 'w') as f:
        f.write(text)
