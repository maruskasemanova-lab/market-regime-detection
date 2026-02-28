from src.api_server_helpers.session_config_apply import (
    SessionConfigRequestResolution,
    apply_session_config_resolution,
    resolve_session_config_request,
)
from src.day_trading_manager import DayTradingManager


def test_resolve_session_config_request_merges_body_overrides():
    resolution = resolve_session_config_request(
        local_values={
            "strategy_selection_mode": "adaptive_top_n",
            "max_active_strategies": 3,
        },
        body_payload={
            "strategy_selection_mode": "all_enabled",
            "momentum_diversification": {"route_name": "defensive"},
            "regime_filter": ["choppy", " mixed "],
            "max_daily_trades": 7,
            "mu_choppy_hard_block_enabled": False,
        },
        query_param_keys=set(),
        momentum_diversification_json="",
        regime_filter_json="",
        max_daily_trades=None,
        mu_choppy_hard_block_enabled=None,
    )

    assert resolution.config_payload["strategy_selection_mode"] == "all_enabled"
    assert resolution.config_payload["regime_filter"] == ["CHOPPY", "MIXED"]
    assert resolution.config_payload["momentum_diversification"] == {"route_name": "defensive"}
    assert resolution.momentum_diversification_payload == {"route_name": "defensive"}
    assert resolution.max_daily_trades == 7
    assert resolution.mu_choppy_hard_block_enabled is False


def test_resolve_session_config_request_keeps_query_param_priority():
    resolution = resolve_session_config_request(
        local_values={
            "strategy_selection_mode": "adaptive_top_n",
        },
        body_payload={
            "strategy_selection_mode": "all_enabled",
            "max_daily_trades": 12,
        },
        query_param_keys={"strategy_selection_mode", "max_daily_trades"},
        momentum_diversification_json="",
        regime_filter_json="",
        max_daily_trades=5,
        mu_choppy_hard_block_enabled=None,
    )

    assert resolution.config_payload["strategy_selection_mode"] == "adaptive_top_n"
    assert resolution.max_daily_trades == 5


def test_apply_session_config_resolution_sets_manager_and_overrides():
    manager = DayTradingManager()
    session = manager.get_or_create_session("run_cfg_apply", "MU", "2026-02-20")
    payload = dict(session.config.to_session_params())
    payload["regime_refresh_bars"] = 11

    resolution = SessionConfigRequestResolution(
        config_payload=payload,
        momentum_diversification_payload=None,
        max_daily_trades=4,
        mu_choppy_hard_block_enabled=True,
    )

    response = apply_session_config_resolution(
        manager=manager,
        session=session,
        run_id="run_cfg_apply",
        ticker="MU",
        resolution=resolution,
    )

    assert manager.regime_refresh_bars == 11
    assert session.max_daily_trades_override == 4
    assert session.mu_choppy_hard_block_enabled_override is True
    assert response["message"] == "Session configured"
    assert response["overrides"]["max_daily_trades"] == 4
