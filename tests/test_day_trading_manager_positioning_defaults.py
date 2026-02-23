from datetime import datetime, timezone

from src.day_trading_manager import DayTradingManager


def test_run_defaults_apply_break_even_settings() -> None:
    manager = DayTradingManager(regime_detection_minutes=5)
    manager.set_run_defaults(
        "run-pos",
        "MU",
        trailing_activation_pct=0.22,
        break_even_buffer_pct=0.01,
        break_even_min_hold_bars=4,
        trailing_enabled_in_choppy=True,
    )

    manager.process_bar(
        "run-pos",
        "MU",
        datetime(2026, 2, 3, 15, 0, tzinfo=timezone.utc),
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.5,
            "close": 100.5,
            "volume": 1000.0,
        },
    )

    session = manager.get_session("run-pos", "MU", "2026-02-03")
    assert session is not None
    assert session.trailing_activation_pct == 0.22
    assert session.break_even_buffer_pct == 0.01
    assert session.break_even_min_hold_bars == 4
    assert session.trailing_enabled_in_choppy is True


def test_run_defaults_apply_momentum_diversification_override() -> None:
    manager = DayTradingManager(regime_detection_minutes=5)
    manager.set_run_defaults(
        "run-momentum",
        "MU",
        momentum_diversification={
            "enabled": True,
            "apply_to_strategies": ["momentum_flow"],
            "min_flow_score": 66.0,
            "fail_fast_exit_enabled": True,
            "fail_fast_max_bars": 5,
        },
    )

    manager.process_bar(
        "run-momentum",
        "MU",
        datetime(2026, 2, 3, 15, 0, tzinfo=timezone.utc),
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.5,
            "close": 100.5,
            "volume": 1000.0,
        },
    )

    session = manager.get_session("run-momentum", "MU", "2026-02-03")
    assert session is not None
    assert session.momentum_diversification_override is True
    assert session.momentum_diversification["enabled"] is True
    assert session.momentum_diversification["min_flow_score"] == 66.0
    assert session.momentum_diversification["fail_fast_exit_enabled"] is True
    assert session.momentum_diversification["fail_fast_max_bars"] == 5


def test_run_defaults_apply_momentum_diversification_multi_sleeves() -> None:
    manager = DayTradingManager(regime_detection_minutes=5)
    manager.set_run_defaults(
        "run-momentum-sleeves",
        "MU",
        momentum_diversification={
            "enabled": True,
            "sleeves": [
                {
                    "sleeve_id": "impulse",
                    "enabled": True,
                    "apply_to_strategies": ["momentum_flow"],
                    "min_flow_score": 65.0,
                    "route_enabled": True,
                    "allocation_weight": 0.7,
                },
                {
                    "sleeve_id": "defensive",
                    "enabled": True,
                    "apply_to_strategies": ["absorption_reversal"],
                    "min_flow_score": 48.0,
                    "route_enabled": True,
                    "allocation_weight": 0.3,
                },
            ],
        },
    )

    manager.process_bar(
        "run-momentum-sleeves",
        "MU",
        datetime(2026, 2, 3, 15, 0, tzinfo=timezone.utc),
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.5,
            "close": 100.5,
            "volume": 1000.0,
        },
    )

    session = manager.get_session("run-momentum-sleeves", "MU", "2026-02-03")
    assert session is not None
    assert session.momentum_diversification_override is True
    sleeves = session.momentum_diversification.get("sleeves", [])
    assert isinstance(sleeves, list)
    assert len(sleeves) == 2
    assert sleeves[0]["sleeve_id"] == "impulse"
    assert sleeves[0]["allocation_weight"] == 0.7
    assert sleeves[1]["sleeve_id"] == "defensive"


def test_existing_session_recovers_missing_orchestrator() -> None:
    manager = DayTradingManager(regime_detection_minutes=5)
    run_id = "run-recover-orchestrator"
    ticker = "MU"
    date = "2026-02-03"

    created = manager.get_or_create_session(run_id, ticker, date)
    assert created.orchestrator is manager.orchestrator

    # Simulate partially initialized session after an earlier config failure.
    created.orchestrator = None
    recovered = manager.get_or_create_session(run_id, ticker, date)

    assert recovered is created
    assert recovered.orchestrator is manager.orchestrator
