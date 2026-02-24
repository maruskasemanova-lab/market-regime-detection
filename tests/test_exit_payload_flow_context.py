from datetime import datetime, timezone
from pathlib import Path
import sys

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.day_trading_manager import DayTradingManager
from src.day_trading_models import DayTrade


def test_build_position_closed_payload_includes_level_context_and_flow_diagnostics() -> None:
    manager = DayTradingManager(regime_detection_minutes=180)
    trade = DayTrade(
        id=1,
        strategy="momentum_flow",
        side="long",
        entry_price=100.0,
        entry_time=datetime(2026, 2, 3, 14, 30, tzinfo=timezone.utc),
        exit_price=101.0,
        exit_time=datetime(2026, 2, 3, 14, 36, tzinfo=timezone.utc),
        size=10.0,
        pnl_pct=1.0,
        pnl_dollars=10.0,
        exit_reason="take_profit",
        flow_snapshot={
            "signed_aggression": 0.12,
            "book_pressure_avg": 0.05,
            "book_pressure_trend": 0.01,
        },
        signal_metadata={
            "level_context": {
                "gate": "intraday_levels_entry_quality",
                "passed": True,
                "reason": "passed",
            }
        },
    )

    payload = manager.exit_engine.build_position_closed_payload(
        trade=trade,
        exit_reason="take_profit",
        bars_held=6,
    )

    assert payload["flow_strategy"] is True
    assert payload["book_pressure_confirmed"] is True
    assert payload["book_pressure_avg"] == 0.05
    assert payload["book_pressure_trend"] == 0.01
    assert payload["signed_aggression"] == 0.12
    assert payload["level_context"]["passed"] is True
    assert payload["signal_metadata"]["level_context"]["gate"] == "intraday_levels_entry_quality"


def test_build_position_closed_payload_marks_first_bar_stop_loss_diagnostics() -> None:
    manager = DayTradingManager(regime_detection_minutes=180)
    trade = DayTrade(
        id=2,
        strategy="vwap_magnet",
        side="long",
        entry_price=100.0,
        entry_time=datetime(2026, 2, 3, 14, 30, tzinfo=timezone.utc),
        exit_price=99.8,
        exit_time=datetime(2026, 2, 3, 14, 31, tzinfo=timezone.utc),
        size=10.0,
        pnl_pct=-0.2,
        pnl_dollars=-2.0,
        exit_reason="stop_loss",
        flow_snapshot={
            "signed_aggression": -0.08,
            "book_pressure_avg": -0.03,
        },
        signal_metadata={
            "vwap_distance_pct": 0.42,
            "bars_since_vwap": 7,
            "risk_controls": {
                "stop_loss_mode": "strategy",
                "strategy_stop_loss": 99.85,
                "effective_stop_loss": 99.85,
            },
            "level_context": {
                "passed": True,
                "reason": "passed",
                "stats": {
                    "near_tested_levels_count": 0,
                    "near_confluence_score": 1,
                    "near_memory_levels_count": 0,
                },
                "volume_profile": {
                    "value_area_position": "inside",
                    "poc_on_trade_side": False,
                },
                "poc_migration": {"regime_bias": "downtrend"},
            },
        },
    )

    payload = manager.exit_engine.build_position_closed_payload(
        trade=trade,
        exit_reason="stop_loss",
        bars_held=1,
    )

    diag = payload.get("entry_quality_diagnostics") or {}
    assert diag["is_first_bar_stop_loss"] is True
    assert diag["is_stop_exit"] is True
    assert diag["stop_distance_pct"] == 0.15
    assert "tight_stop_distance" in diag["first_bar_stop_tags"]
    assert "missing_tested_level_confluence" in diag["first_bar_stop_tags"]
    assert "poc_not_on_trade_side" in diag["first_bar_stop_tags"]
