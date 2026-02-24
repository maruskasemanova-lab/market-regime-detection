from datetime import datetime, timezone
from src.day_trading_models import TradingSession, BarData, Signal, SignalType
from src.day_trading_runtime_intrabar import intrabar_confirmation_snapshot

ts = datetime(2026, 2, 12, 15, 32, tzinfo=timezone.utc)
session = TradingSession(run_id="r3", ticker="MU", date="2026-02-12")
session.bars = [
    BarData(
        timestamp=ts,
        open=100.0,
        high=100.0,
        low=100.0,
        close=100.0,
        volume=100,
        intrabar_quotes_1s=[
            {"s": 1, "bid": 100.00, "ask": 100.02},
            {"s": 2, "bid": 100.01, "ask": 100.03},
            {"s": 3, "bid": 100.00, "ask": 100.02},
            {"s": 4, "bid": 100.02, "ask": 100.04},
        ],
    )
]
signal = Signal(
    strategy_name="test",
    signal_type=SignalType.BUY,
    price=100.02,
    timestamp=ts,
    confidence=50.0,
    stop_loss=99.0
)

snapshot = intrabar_confirmation_snapshot(
    session=session,
    signal=signal,
    current_bar_index=1,
    signal_bar_index=0,
    window_seconds=5,
    min_coverage_points=2,
    min_move_pct=0.005,
    min_push_ratio=0.8,
    max_spread_bps=20.0,
)

print(snapshot)
