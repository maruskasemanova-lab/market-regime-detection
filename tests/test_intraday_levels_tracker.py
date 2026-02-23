from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.day_trading_manager import DayTradingManager


def _bar(
    *,
    open_px: float,
    high_px: float,
    low_px: float,
    close_px: float,
    volume: float,
) -> dict:
    return {
        "open": open_px,
        "high": high_px,
        "low": low_px,
        "close": close_px,
        "volume": volume,
        "vwap": (high_px + low_px + close_px) / 3.0,
    }


def test_intraday_level_tracker_detects_bounce_and_break() -> None:
    manager = DayTradingManager(regime_detection_minutes=180)
    run_id = "run-levels"
    ticker = "MU"
    start = datetime(2026, 2, 3, 15, 30, tzinfo=timezone.utc)  # 10:30 ET

    bars = [
        _bar(open_px=100.0, high_px=100.4, low_px=99.8, close_px=100.2, volume=1_000.0),
        _bar(open_px=100.2, high_px=101.4, low_px=100.0, close_px=101.0, volume=1_050.0),
        _bar(open_px=101.0, high_px=103.5, low_px=100.9, close_px=103.0, volume=1_100.0),
        _bar(open_px=103.0, high_px=103.1, low_px=101.2, close_px=101.6, volume=950.0),
        _bar(open_px=101.6, high_px=101.8, low_px=100.3, close_px=100.6, volume=980.0),
        _bar(open_px=102.7, high_px=103.45, low_px=102.3, close_px=102.5, volume=1_020.0),
        _bar(open_px=102.8, high_px=104.1, low_px=102.7, close_px=103.9, volume=3_000.0),
    ]

    result = {}
    for idx, payload in enumerate(bars):
        result = manager.process_bar(
            run_id,
            ticker,
            start + timedelta(minutes=idx),
            payload,
        )

    intraday = result.get("intraday_levels") or {}
    events = intraday.get("recent_events") or []
    assert any(
        evt.get("event_type") == "bounce" and evt.get("level_kind") == "resistance"
        for evt in events
    )
    assert any(
        evt.get("event_type") == "break"
        and evt.get("level_kind") == "resistance"
        and evt.get("volume_confirmed") is True
        for evt in events
    )
    assert (intraday.get("stats") or {}).get("tested_levels", 0) >= 1


def test_intraday_level_tracker_exposes_volume_profile() -> None:
    manager = DayTradingManager(regime_detection_minutes=180)
    run_id = "run-levels-profile"
    ticker = "MU"
    start = datetime(2026, 2, 3, 15, 30, tzinfo=timezone.utc)

    for idx, close_px in enumerate([100.0, 100.2, 100.4, 100.3, 100.1, 100.25]):
        manager.process_bar(
            run_id,
            ticker,
            start + timedelta(minutes=idx),
            _bar(
                open_px=close_px - 0.05,
                high_px=close_px + 0.15,
                low_px=close_px - 0.12,
                close_px=close_px,
                volume=900.0 + idx * 25.0,
            ),
        )

    result = manager.process_bar(
        run_id,
        ticker,
        start + timedelta(minutes=6),
        _bar(open_px=100.2, high_px=100.5, low_px=100.0, close_px=100.3, volume=1_200.0),
    )

    intraday = result.get("intraday_levels") or {}
    profile = intraday.get("volume_profile") or {}
    poc = profile.get("poc_price")
    va_low = profile.get("value_area_low")
    va_high = profile.get("value_area_high")

    assert profile.get("total_volume", 0.0) > 0.0
    assert poc is not None
    assert va_low is not None
    assert va_high is not None
    assert va_low <= poc <= va_high


def test_intraday_level_tracker_isolation_per_day_session() -> None:
    manager = DayTradingManager(regime_detection_minutes=180)
    run_id = "run-levels-reset"
    ticker = "MU"
    day1 = datetime(2026, 2, 3, 15, 30, tzinfo=timezone.utc)

    for idx, payload in enumerate(
        [
            _bar(open_px=100.0, high_px=100.5, low_px=99.8, close_px=100.3, volume=1_000.0),
            _bar(open_px=100.3, high_px=101.7, low_px=100.2, close_px=101.4, volume=1_020.0),
            _bar(open_px=101.4, high_px=102.6, low_px=101.2, close_px=102.1, volume=1_040.0),
            _bar(open_px=102.1, high_px=102.2, low_px=100.8, close_px=101.0, volume=990.0),
            _bar(open_px=101.0, high_px=101.1, low_px=100.4, close_px=100.6, volume=980.0),
        ]
    ):
        manager.process_bar(run_id, ticker, day1 + timedelta(minutes=idx), payload)

    session_day1 = manager.get_session(run_id, ticker, "2026-02-03")
    assert session_day1 is not None
    day1_snapshot = (session_day1.intraday_levels_state or {}).get("snapshot") or {}
    assert (day1_snapshot.get("stats") or {}).get("total_levels", 0) >= 1

    day2_result = manager.process_bar(
        run_id,
        ticker,
        datetime(2026, 2, 4, 15, 30, tzinfo=timezone.utc),
        _bar(open_px=101.0, high_px=101.3, low_px=100.9, close_px=101.2, volume=1_100.0),
    )
    day2_snapshot = day2_result.get("intraday_levels") or {}

    assert day2_snapshot.get("bars_processed") == 1
    assert (day2_snapshot.get("stats") or {}).get("total_levels", -1) == 0
    assert (day2_snapshot.get("recent_events") or []) == []
