import glob

for filename in glob.glob('tests/test_day_trading_manager*.py'):
    with open(filename, 'r') as f:
        text = f.read()
    
    # Fix python 3.10 syntax back to python 3.9
    text = text.replace('dict | None', 'Optional[dict]')
    
    with open(filename, 'w') as f:
        f.write(text)
