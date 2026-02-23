import pytest
from datetime import datetime, timezone
from src.day_trading_manager import DayTradingManager
from src.day_trading_models import BarData

def test_evaluate_intrabar_slice():
    manager = DayTradingManager()
    run_id = "test_run"
    ticker = "TEST"
    date = "2026-02-22"
    
    # Pre-populate session
    session = manager.get_or_create_session(run_id, ticker, date)
    
    # 5 dummy bars to initialize indicators somewhat
    for i in range(5):
        ts = datetime(2026, 2, 22, 9, 30 + i, tzinfo=timezone.utc)
        bar = BarData(
            timestamp=ts,
            open=100.0, high=101.0, low=99.0, close=100.0, volume=1000
        )
        session.bars.append(bar)
        
    ts_new = datetime(2026, 2, 22, 9, 35, tzinfo=timezone.utc)
    bar_data = {
        'open': 100.0, 'high': 102.0, 'low': 99.0, 'close': 101.5, 'volume': 2000
    }
    
    # Action
    result = manager.evaluate_intrabar_slice(run_id, ticker, ts_new, bar_data)
    
    assert result is not None
    assert "layer_scores" in result
    assert result["layer_scores"]["schema_version"] == 2
    assert "timestamp" in result
    
    # Verify no side-effects
    assert len(session.bars) == 5
