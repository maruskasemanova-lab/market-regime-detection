"""Tests for runtime intrabar indicator snapshot generation."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.day_trading_manager import BarData, DayTradingManager


def _bar(ts: datetime, close: float, intrabar_quotes_1s=None) -> BarData:
    return BarData(
        timestamp=ts,
        open=close - 0.05,
        high=close + 0.08,
        low=close - 0.08,
        close=close,
        volume=22_000.0,
        vwap=close - 0.01,
        l2_delta=900.0,
        l2_buy_volume=6_500.0,
        l2_sell_volume=5_800.0,
        l2_volume=12_300.0,
        l2_imbalance=0.09,
        l2_bid_depth_total=4_200.0,
        l2_ask_depth_total=3_900.0,
        l2_book_pressure=0.07,
        l2_book_pressure_change=0.01,
        l2_iceberg_buy_count=1.0,
        l2_iceberg_sell_count=0.0,
        l2_iceberg_bias=0.12,
        intrabar_quotes_1s=intrabar_quotes_1s,
    )


def test_intrabar_snapshot_is_exposed_in_runtime_indicators() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    t0 = datetime(2026, 2, 3, 14, 30, tzinfo=timezone.utc)
    bars = [
        _bar(t0 + timedelta(minutes=idx), 100.0 + idx * 0.1)
        for idx in range(4)
    ]
    bars.append(
        _bar(
            t0 + timedelta(minutes=4),
            100.6,
            intrabar_quotes_1s=[
                {"s": 5, "bid": 100.40, "ask": 100.44},
                {"s": 17, "bid": 100.46, "ask": 100.50},
                {"s": 31, "bid": 100.52, "ask": 100.55},
                {"s": 47, "bid": 100.56, "ask": 100.60},
            ],
        )
    )

    indicators = manager._calculate_indicators(bars)
    intrabar = indicators.get("intrabar_1s") or {}

    assert intrabar.get("has_intrabar_coverage") is True
    assert intrabar.get("coverage_points", 0) >= 4
    assert intrabar.get("mid_move_pct", 0.0) > 0.0
    assert intrabar.get("push_ratio", 0.0) > 0.0
    assert intrabar.get("window_eval_seconds") == 5
    assert intrabar.get("window_long_move_pct", 0.0) >= 0.0
    assert intrabar.get("window_short_move_pct", 0.0) <= 0.0
    assert "order_flow" in indicators


def test_intrabar_snapshot_uses_current_bar_only() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    t0 = datetime(2026, 2, 3, 14, 30, tzinfo=timezone.utc)
    bars = [
        _bar(t0 + timedelta(minutes=idx), 100.0 + idx * 0.1)
        for idx in range(3)
    ]
    bars.append(
        _bar(
            t0 + timedelta(minutes=3),
            100.3,
            intrabar_quotes_1s=[
                {"s": 10, "bid": 100.18, "ask": 100.21},
                {"s": 30, "bid": 100.24, "ask": 100.28},
            ],
        )
    )
    bars.append(_bar(t0 + timedelta(minutes=4), 100.4, intrabar_quotes_1s=None))

    indicators = manager._calculate_indicators(bars)
    intrabar = indicators.get("intrabar_1s") or {}

    assert intrabar.get("has_intrabar_coverage") is False
    assert intrabar.get("coverage_points") == 0


def test_intrabar_snapshot_exists_even_during_short_warmup() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    t0 = datetime(2026, 2, 3, 14, 30, tzinfo=timezone.utc)
    bars = [
        _bar(t0, 100.0),
        _bar(
            t0 + timedelta(minutes=1),
            100.1,
            intrabar_quotes_1s=[
                {"s": 15, "bid": 100.08, "ask": 100.10},
                {"s": 40, "bid": 100.12, "ask": 100.14},
            ],
        ),
    ]

    indicators = manager._calculate_indicators(bars)
    assert "intrabar_1s" in indicators
    assert "order_flow" in indicators
