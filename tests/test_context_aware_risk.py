from src.context_aware_risk import ContextRiskConfig, adjust_entry_risk


def test_adjust_entry_risk_enforces_min_sl_floor_for_long() -> None:
    result = adjust_entry_risk(
        entry_price=100.0,
        side="long",
        original_stop_loss=99.95,
        original_take_profit=101.0,
        levels_payload={
            "levels": [
                {
                    "kind": "support",
                    "price": 99.96,
                    "tests": 2,
                    "source": "swing",
                    "broken": False,
                }
            ]
        },
        config=ContextRiskConfig(
            enabled=True,
            sl_buffer_pct=0.0,
            min_sl_pct=0.30,
            min_room_pct=0.0,
            min_effective_rr=0.0,
        ),
    )

    assert result["skip"] is False
    assert result["adjusted_stop_loss"] == 99.7
    assert "min_sl_floor:0.3000" in str(result["sl_reason"])


def test_adjust_entry_risk_enforces_min_sl_floor_for_short() -> None:
    result = adjust_entry_risk(
        entry_price=100.0,
        side="short",
        original_stop_loss=100.05,
        original_take_profit=99.0,
        levels_payload={
            "levels": [
                {
                    "kind": "resistance",
                    "price": 100.02,
                    "tests": 2,
                    "source": "swing",
                    "broken": False,
                }
            ]
        },
        config=ContextRiskConfig(
            enabled=True,
            sl_buffer_pct=0.0,
            min_sl_pct=0.30,
            min_room_pct=0.0,
            min_effective_rr=0.0,
        ),
    )

    assert result["skip"] is False
    assert result["adjusted_stop_loss"] == 100.3
    assert "min_sl_floor:0.3000" in str(result["sl_reason"])


def test_adjust_entry_risk_applies_sweep_atr_buffer_to_anchor() -> None:
    result = adjust_entry_risk(
        entry_price=100.0,
        side="long",
        original_stop_loss=99.6,
        original_take_profit=101.0,
        levels_payload={
            "levels": [
                {
                    "kind": "support",
                    "price": 99.0,
                    "tests": 2,
                    "source": "swing",
                    "broken": False,
                }
            ]
        },
        config=ContextRiskConfig(
            enabled=True,
            sl_buffer_pct=0.0,
            min_sl_pct=0.0,
            min_room_pct=0.0,
            min_effective_rr=0.0,
            sweep_atr_buffer_multiplier=0.5,
        ),
        atr=2.0,
        is_sweep_trade=True,
    )

    assert result["skip"] is False
    assert result["adjusted_stop_loss"] == 98.0
    assert "sweep_atr_buffer:1.0000" in str(result["sl_reason"])


def test_adjust_entry_risk_uses_pullback_specific_min_sl_floor() -> None:
    result = adjust_entry_risk(
        entry_price=100.0,
        side="long",
        original_stop_loss=99.85,
        original_take_profit=101.0,
        levels_payload={
            "levels": [
                {
                    "kind": "support",
                    "price": 99.9,
                    "tests": 2,
                    "source": "swing",
                    "broken": False,
                }
            ]
        },
        config=ContextRiskConfig(
            enabled=True,
            sl_buffer_pct=0.0,
            min_sl_pct=0.30,
            pullback_min_sl_pct=0.50,
            min_room_pct=0.0,
            min_effective_rr=0.0,
        ),
        strategy_key="pullback",
    )

    assert result["skip"] is False
    assert result["adjusted_stop_loss"] == 99.5
    assert "min_sl_floor:0.5000" in str(result["sl_reason"])
    assert result["configured_min_sl_pct"] == 0.5
