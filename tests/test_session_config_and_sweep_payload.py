from __future__ import annotations

import os
from datetime import datetime, timezone

from fastapi.testclient import TestClient

import api_server
from src.day_trading_manager import DayTradingManager
from src.day_trading_models import SessionPhase
from src.strategies.base_strategy import Regime
from src.trading_config import TradingConfig


def _internal_headers() -> dict[str, str]:
    token = str(os.getenv("STRATEGY_INTERNAL_API_TOKEN") or "").strip()
    if not token:
        return {}
    return {"x-internal-token": token}


def test_configure_session_accepts_json_body_payload() -> None:
    client = TestClient(api_server.app)
    run_id = "test_session_config_json_body"
    ticker = "MU"
    date = "2026-02-20"

    response = client.post(
        "/api/session/config",
        params={"run_id": run_id, "ticker": ticker, "date": date},
        json={
            "liquidity_sweep_detection_enabled": True,
            "strategy_selection_mode": "all_enabled",
            "regime_filter": [],
        },
        headers=_internal_headers(),
    )

    assert response.status_code == 200, response.text
    session = api_server.day_trading_manager.get_session(run_id, ticker, date)
    assert session is not None
    assert bool(session.config.liquidity_sweep_detection_enabled) is True
    assert str(session.strategy_selection_mode) == "all_enabled"
    assert list(getattr(session, "regime_filter_override", [])) == []

    api_server.day_trading_manager.clear_session(run_id, ticker, date)


def test_configure_session_accepts_tcbbo_body_payload() -> None:
    client = TestClient(api_server.app)
    run_id = "test_session_config_tcbbo_body"
    ticker = "MU"
    date = "2026-02-20"

    response = client.post(
        "/api/session/config",
        params={"run_id": run_id, "ticker": ticker, "date": date},
        json={
            "tcbbo_gate_enabled": True,
            "tcbbo_min_net_premium": 250000.0,
            "tcbbo_sweep_boost": 5.0,
            "tcbbo_lookback_bars": 7,
        },
        headers=_internal_headers(),
    )

    assert response.status_code == 200, response.text
    session = api_server.day_trading_manager.get_session(run_id, ticker, date)
    assert session is not None
    assert bool(session.tcbbo_gate_enabled) is True
    assert float(session.tcbbo_min_net_premium) == 250000.0
    assert float(session.tcbbo_sweep_boost) == 5.0
    assert int(session.tcbbo_lookback_bars) == 7

    key = api_server.day_trading_manager._get_session_key(run_id, ticker, date)
    api_server.day_trading_manager.sessions.pop(key, None)


def test_trading_result_always_contains_liquidity_sweep_payload() -> None:
    manager = DayTradingManager()
    run_id = "test_liquidity_sweep_payload_visibility"
    ticker = "MU"
    date = "2026-02-20"
    timestamp = datetime(2026, 2, 20, 15, 0, tzinfo=timezone.utc)

    session = manager.get_or_create_session(run_id, ticker, date)
    session.phase = SessionPhase.TRADING
    session.detected_regime = Regime.CHOPPY
    session.selected_strategy = None
    session.selection_warnings = ["regime CHOPPY blocked by regime_filter=['TRENDING']"]

    cfg_payload = dict(session.config.to_session_params())
    cfg_payload["liquidity_sweep_detection_enabled"] = True
    manager._apply_trading_config_to_session(
        session,
        TradingConfig.from_dict(cfg_payload),
        normalize_momentum=False,
    )

    result = manager.process_bar(
        run_id=run_id,
        ticker=ticker,
        timestamp=timestamp,
        bar_data={
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000.0,
        },
    )

    sweep_payload = result.get("liquidity_sweep")
    assert isinstance(sweep_payload, dict)
    assert sweep_payload.get("enabled") is True
    assert sweep_payload.get("sweep_detected") is False
    assert sweep_payload.get("reason") == "strategy_not_selected"
    assert isinstance(sweep_payload.get("selection_warnings"), list)
