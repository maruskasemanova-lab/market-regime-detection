import re
import glob

for filename in glob.glob('tests/test_day_trading_manager*.py'):
    with open(filename, 'r') as f:
        text = f.read()
    
    # Remove the `from typing import ...` that was accidentally added at the absolute top of file
    text = re.sub(r'^from typing import Any, Dict, List, Optional, Tuple\n', '', text)
    
    # Insert safely after from __future__ import annotations if it exists
    if 'from __future__ import annotations' in text:
        text = text.replace('from __future__ import annotations', 'from __future__ import annotations\nfrom typing import Any, Dict, List, Optional, Tuple', 1)
    else:
        text = 'from typing import Any, Dict, List, Optional, Tuple\n' + text
        
    with open(filename, 'w') as f:
        f.write(text)

