from __future__ import annotations

import pytest

from src.api_server_helpers.strategy_update import apply_strategy_param_updates
from src.strategy_formula_engine import StrategyFormulaValidationError


class _Regime:
    def __init__(self, value: str) -> None:
        self.value = value


class _DummyStrategy:
    def __init__(self) -> None:
        self.custom_entry_formula = ""
        self.custom_entry_formula_enabled = False
        self.custom_exit_formula = ""
        self.custom_exit_formula_enabled = False
        self.allowed_regimes = []
        self.exit_mode = "custom"
        self.trailing_stop_mode = "custom"
        self.risk_mode = "custom"
        self.trailing_stop_pct = 1.0
        self.global_trailing_stop_pct = None
        self.global_rr_ratio = None
        self.global_atr_stop_multiplier = None
        self.global_volume_stop_pct = None
        self.global_min_stop_loss_pct = None


def _validate_formula(formula: str):
    if "invalid" in formula:
        raise StrategyFormulaValidationError("invalid expression")
    return {"normalized": formula.strip()}


def _coerce_regimes(raw):
    return [_Regime(str(item).upper()) for item in raw]


def test_apply_strategy_param_updates_syncs_primary_and_dtm_strategy():
    strat = _DummyStrategy()
    dtm_strat = _DummyStrategy()

    updated = apply_strategy_param_updates(
        strat=strat,
        dtm_strat=dtm_strat,
        params={
            "custom_entry_formula_enabled": True,
            "custom_entry_formula": " flow_score > 50 ",
            "exit_mode": "global",
            "risk_mode": "global",
            "global_trailing_stop_pct": 0.75,
            "trailing_stop_pct": 1.4,
            "allowed_regimes": ["trending", "mixed"],
        },
        validate_formula=_validate_formula,
        coerce_regimes=_coerce_regimes,
    )

    assert updated["custom_entry_formula_enabled"] is True
    assert updated["custom_entry_formula"] == "flow_score > 50"
    assert updated["exit_mode"] == "global"
    assert updated["trailing_stop_mode"] == "global"
    assert updated["risk_mode"] == "global"
    assert updated["global_trailing_stop_pct"] == 0.75
    assert updated["trailing_stop_pct"] == 1.4
    assert updated["allowed_regimes"] == ["TRENDING", "MIXED"]
    assert dtm_strat.trailing_stop_mode == "global"
    assert dtm_strat.risk_mode == "global"
    assert dtm_strat.global_trailing_stop_pct == 0.75


def test_apply_strategy_param_updates_raises_value_error_for_invalid_formula():
    strat = _DummyStrategy()

    with pytest.raises(ValueError, match="Invalid custom_entry_formula"):
        apply_strategy_param_updates(
            strat=strat,
            dtm_strat=None,
            params={"custom_entry_formula": "invalid()"},
            validate_formula=_validate_formula,
            coerce_regimes=_coerce_regimes,
        )
