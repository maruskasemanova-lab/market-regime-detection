from __future__ import annotations

import pytest

from src.strategy_formula_engine import (
    StrategyFormulaValidationError,
    evaluate_strategy_formula,
    validate_strategy_formula,
)


def test_validate_formula_accepts_supported_expression() -> None:
    info = validate_strategy_formula(
        "regime in ('TRENDING','MIXED') and flow_score >= 55 and signed_aggression > 0.05"
    )
    assert info["valid"] is True
    assert "flow_score" in info["variables"]
    assert "signed_aggression" in info["variables"]


def test_validate_formula_rejects_unknown_variable() -> None:
    with pytest.raises(StrategyFormulaValidationError):
        validate_strategy_formula("unknown_metric > 1")


def test_evaluate_formula_handles_boolean_and_numeric_ops() -> None:
    passed = evaluate_strategy_formula(
        "position_side == 'long' and position_pnl_pct < -0.2 and (flow_score < 45 or signed_aggression < -0.02)",
        {
            "position_side": "long",
            "position_pnl_pct": -0.35,
            "flow_score": 42.0,
            "signed_aggression": -0.01,
        },
    )
    assert passed is True

    not_passed = evaluate_strategy_formula(
        "max(atr, 0.01) > 0.5 and rsi < 30",
        {
            "atr": 0.3,
            "rsi": 48.0,
        },
    )
    assert not_passed is False

