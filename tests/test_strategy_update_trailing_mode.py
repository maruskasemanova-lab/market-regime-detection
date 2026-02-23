from __future__ import annotations

from fastapi.testclient import TestClient

import api_server


def test_strategy_update_supports_global_and_custom_trailing_modes() -> None:
    client = TestClient(api_server.app)

    update_global = client.post(
        "/api/strategies/update",
        json={
            "strategy_name": "momentum_flow",
            "params": {
                "global_trailing_stop_pct": 0.57,
                "global_rr_ratio": 1.9,
                "global_atr_stop_multiplier": 0.72,
                "exit_mode": "global",
                "risk_mode": "global",
            },
        },
    )
    assert update_global.status_code == 200
    payload_global = update_global.json()
    assert payload_global["current"]["exit_mode"] == "global"
    assert payload_global["current"]["trailing_stop_mode"] == "global"
    assert payload_global["current"]["risk_mode"] == "global"
    assert payload_global["current"]["global_trailing_stop_pct"] == 0.57
    assert payload_global["current"]["global_rr_ratio"] == 1.9
    assert payload_global["current"]["global_atr_stop_multiplier"] == 0.72
    assert payload_global["current"]["effective_trailing_stop_pct"] == 0.57
    assert payload_global["current"]["effective_rr_ratio"] == 1.9
    assert payload_global["current"]["effective_atr_stop_multiplier"] == 0.72

    update_custom = client.post(
        "/api/strategies/update",
        json={
            "strategy_name": "momentum_flow",
            "params": {
                "trailing_stop_pct": 1.22,
                "rr_ratio": 2.35,
                "atr_stop_multiplier": 1.1,
                "trailing_stop_mode": "custom",
                "risk_mode": "custom",
            },
        },
    )
    assert update_custom.status_code == 200
    payload_custom = update_custom.json()
    assert payload_custom["current"]["exit_mode"] == "custom"
    assert payload_custom["current"]["trailing_stop_mode"] == "custom"
    assert payload_custom["current"]["risk_mode"] == "custom"
    assert payload_custom["current"]["effective_trailing_stop_pct"] == 1.22
    assert payload_custom["current"]["effective_rr_ratio"] == 2.35
    assert payload_custom["current"]["effective_atr_stop_multiplier"] == 1.1


def test_strategy_update_supports_custom_entry_exit_formulas() -> None:
    client = TestClient(api_server.app)

    update_formula = client.post(
        "/api/strategies/update",
        json={
            "strategy_name": "mean_reversion",
            "params": {
                "custom_entry_formula_enabled": True,
                "custom_entry_formula": "regime in ('TRENDING','MIXED') and flow_score >= 45",
                "custom_exit_formula_enabled": True,
                "custom_exit_formula": "position_side == 'long' and position_pnl_pct < -0.35",
            },
        },
    )
    assert update_formula.status_code == 200
    payload = update_formula.json()
    assert payload["current"]["custom_entry_formula_enabled"] is True
    assert payload["current"]["custom_exit_formula_enabled"] is True
    assert "flow_score" in payload["current"]["custom_entry_formula"]
    assert "position_pnl_pct" in payload["current"]["custom_exit_formula"]


def test_strategy_update_rejects_invalid_custom_formula() -> None:
    client = TestClient(api_server.app)

    invalid_formula = client.post(
        "/api/strategies/update",
        json={
            "strategy_name": "mean_reversion",
            "params": {
                "custom_entry_formula_enabled": True,
                "custom_entry_formula": "__import__('os').system('echo x')",
            },
        },
    )
    assert invalid_formula.status_code == 400
    detail = str(invalid_formula.json().get("detail", ""))
    assert "Invalid custom_entry_formula" in detail
