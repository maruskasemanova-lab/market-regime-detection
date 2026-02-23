from typing import Optional
from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.day_trading_manager import DayTradingManager
from src.intraday_levels import ensure_intraday_levels_state
from src.strategies.base_strategy import Signal, SignalType
from src.trading_config import TradingConfig


def _make_manager_and_session():
    manager = DayTradingManager(regime_detection_minutes=180)
    session = manager.get_or_create_session("run-gate", "MU", "2026-02-03")
    session.config = TradingConfig.from_dict(
        {
            "intraday_levels_entry_quality_enabled": True,
            "intraday_levels_min_levels_for_context": 2,
            "intraday_levels_entry_tolerance_pct": 0.10,
            "intraday_levels_break_cooldown_bars": 6,
            "intraday_levels_rotation_max_tests": 2,
            "intraday_levels_rotation_volume_max_ratio": 0.95,
            "intraday_levels_recent_bounce_lookback_bars": 6,
            "intraday_levels_require_recent_bounce_for_mean_reversion": True,
            "intraday_levels_momentum_break_max_age_bars": 3,
            "intraday_levels_momentum_min_room_pct": 0.30,
            "intraday_levels_momentum_min_broken_ratio": 0.30,
            # Keep base gate tests focused on level logic unless a case
            # explicitly validates RVOL/adaptive-window behavior.
            "intraday_levels_rvol_filter_enabled": False,
            "intraday_levels_adaptive_window_enabled": False,
        }
    )
    return manager, session


def _signal(strategy_name: str, *, side: str = "buy", metadata: Optional[dict] = None) -> Signal:
    signal_type = SignalType.BUY if side.lower() == "buy" else SignalType.SELL
    return Signal(
        strategy_name=strategy_name,
        signal_type=signal_type,
        price=100.0,
        timestamp=datetime(2026, 2, 3, 16, 0, tzinfo=timezone.utc),
        confidence=70.0,
        metadata=dict(metadata or {}),
    )


def _state(*, levels, recent_events, volume_profile):
    return ensure_intraday_levels_state(
        {
            "config": {"enabled": True},
            "levels": list(levels),
            "recent_events": list(recent_events),
            "volume_profile": dict(volume_profile),
        }
    )


def test_entry_quality_gate_blocks_vwap_without_tested_level() -> None:
    manager, session = _make_manager_and_session()
    session.intraday_levels_state = _state(
        levels=[
            {"id": 1, "kind": "support", "price": 99.95, "tests": 0, "broken": False},
            {"id": 2, "kind": "resistance", "price": 101.0, "tests": 1, "broken": False},
        ],
        recent_events=[],
        volume_profile={
            "poc_price": 100.6,
            "value_area_low": 100.1,
            "value_area_high": 100.4,
        },
    )

    context = manager._evaluate_intraday_levels_entry_quality(
        session=session,
        signal=_signal("vwap_magnet", side="buy"),
        current_price=100.0,
        current_bar_index=120,
    )

    assert context["passed"] is False
    assert "vwap_requires_tested_level_near_entry" in context["reasons"]
    assert context["checks"]["near_tested_level"] is False
    assert context["volume_profile"]["price_outside_value_area"] is True


def test_entry_quality_gate_sets_mean_reversion_target_to_poc() -> None:
    manager, session = _make_manager_and_session()
    session.config = TradingConfig.from_dict(
        {
            **session.config.to_session_params(),
            "intraday_levels_min_confluence_score": 1,
        }
    )
    session.intraday_levels_state = _state(
        levels=[
            {"id": 1, "kind": "support", "price": 99.94, "tests": 1, "broken": False},
            {"id": 2, "kind": "resistance", "price": 101.2, "tests": 1, "broken": False},
        ],
        recent_events=[
            {
                "event_type": "bounce",
                "direction": "bullish",
                "bar_index": 118,
                "price": 99.94,
            }
        ],
        volume_profile={
            "poc_price": 100.8,
            "value_area_low": 100.2,
            "value_area_high": 100.5,
        },
    )

    context = manager._evaluate_intraday_levels_entry_quality(
        session=session,
        signal=_signal("mean_reversion", side="buy"),
        current_price=99.95,
        current_bar_index=120,
    )

    assert context["passed"] is True
    assert context["reason"] == "passed"
    assert context["checks"]["recent_aligned_bounce"] is True
    assert context["target_price_override"] == pytest.approx(100.8, abs=1e-4)


def test_entry_quality_gate_blocks_rotation_on_overtested_level() -> None:
    manager, session = _make_manager_and_session()
    session.intraday_levels_state = _state(
        levels=[
            {"id": 1, "kind": "resistance", "price": 100.04, "tests": 3, "broken": False},
            {"id": 2, "kind": "support", "price": 99.1, "tests": 1, "broken": False},
        ],
        recent_events=[],
        volume_profile={
            "poc_price": 100.1,
            "value_area_low": 99.8,
            "value_area_high": 100.2,
        },
    )

    context = manager._evaluate_intraday_levels_entry_quality(
        session=session,
        signal=_signal("rotation", side="buy", metadata={"volume_ratio": 0.8}),
        current_price=100.0,
        current_bar_index=120,
    )

    assert context["passed"] is False
    assert "rotation_level_overtested" in context["reasons"]
    assert context["checks"]["rotation_level_tests_range"] is False
    assert context["checks"]["rotation_volume_exhaustion"] is True


def test_entry_quality_gate_blocks_momentum_without_volume_confirmed_break() -> None:
    manager, session = _make_manager_and_session()
    session.intraday_levels_state = _state(
        levels=[
            {"id": 1, "kind": "resistance", "price": 100.2, "tests": 2, "broken": True},
            {"id": 2, "kind": "resistance", "price": 101.0, "tests": 1, "broken": False},
            {"id": 3, "kind": "support", "price": 99.5, "tests": 1, "broken": False},
        ],
        recent_events=[
            {
                "event_type": "break",
                "direction": "bullish",
                "volume_confirmed": False,
                "bar_index": 119,
                "price": 100.2,
            }
        ],
        volume_profile={
            "poc_price": 99.9,
            "value_area_low": 99.7,
            "value_area_high": 100.3,
        },
    )

    context = manager._evaluate_intraday_levels_entry_quality(
        session=session,
        signal=_signal("momentum", side="buy"),
        current_price=100.5,
        current_bar_index=120,
    )

    assert context["passed"] is False
    assert "momentum_requires_latest_event_break" in context["reasons"]
    assert context["checks"]["latest_event_break"] is False


def test_entry_quality_gate_uses_memory_level_confluence_score() -> None:
    manager, session = _make_manager_and_session()
    session.config = TradingConfig.from_dict(
        {
            **session.config.to_session_params(),
            "intraday_levels_entry_quality_enabled": True,
            "intraday_levels_min_confluence_score": 3,
        }
    )
    session.intraday_levels_state = _state(
        levels=[
            {
                "id": 1,
                "kind": "support",
                "price": 99.94,
                "tests": 1,
                "broken": False,
                "memory_level": True,
                "memory_age_days": 2,
                "memory_weight": 0.5,
                "source": "memory_level",
            },
            {"id": 2, "kind": "resistance", "price": 101.2, "tests": 1, "broken": False},
        ],
        recent_events=[],
        volume_profile={
            "poc_price": 100.8,
            "value_area_low": 100.1,
            "value_area_high": 100.5,
        },
    )

    context = manager._evaluate_intraday_levels_entry_quality(
        session=session,
        signal=_signal("vwap_magnet", side="buy"),
        current_price=99.95,
        current_bar_index=120,
    )

    assert context["passed"] is True
    assert context["stats"]["near_tested_levels_count"] >= 1
    assert context["stats"]["near_confluence_score"] >= 3
    assert context["checks"]["minimum_confluence_score"] is True
