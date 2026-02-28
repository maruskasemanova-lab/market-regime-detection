from src.trading_config import TradingConfig


def test_from_dict_validates_and_clamps_fields() -> None:
    cfg = TradingConfig.from_dict(
        {
            "regime_detection_minutes": 0,
            "regime_refresh_bars": 1,
            "account_size_usd": -100,
            "risk_per_trade_pct": -5,
            "max_position_notional_pct": 0,
            "max_fill_participation_rate": 2.5,
            "min_fill_ratio": -1.0,
            "partial_take_profit_rr": 0.1,
            "partial_take_profit_fraction": 9.0,
            "break_even_min_hold_bars": 0,
            "time_exit_bars": 0,
            "adverse_flow_threshold": 0.0,
            "adverse_flow_min_hold_bars": 0,
            "adverse_flow_consistency_threshold": -1,
            "adverse_book_pressure_threshold": -1,
            "stop_loss_mode": "INVALID",
            "fixed_stop_loss_pct": -1,
            "context_risk_min_sl_pct": -1,
            "intraday_levels_micro_confirmation_disable_for_sweep": "1",
            "intraday_levels_micro_confirmation_sweep_bars": -2,
            "intraday_levels_micro_confirmation_require_intrabar": "yes",
            "intraday_levels_micro_confirmation_intrabar_window_seconds": 0,
            "intraday_levels_micro_confirmation_intrabar_min_coverage_points": -1,
            "intraday_levels_micro_confirmation_intrabar_min_move_pct": -1,
            "intraday_levels_micro_confirmation_intrabar_min_push_ratio": 2.5,
            "intraday_levels_micro_confirmation_intrabar_max_spread_bps": -1,
            "intrabar_eval_step_seconds": 0,
            "liquidity_sweep_detection_enabled": "1",
            "sweep_min_aggression_z": "-3.5",
            "sweep_min_book_pressure_z": "2.2",
            "sweep_max_price_change_pct": -1,
            "sweep_atr_buffer_multiplier": -1,
            "l2_lookback_bars": 0,
            "strategy_selection_mode": "NOPE",
            "max_active_strategies": 999,
            "cold_start_each_day": "yes",
            "enable_partial_take_profit": "0",
            "momentum_diversification": ["bad"],
        }
    )

    assert cfg.regime_detection_minutes == 1
    assert cfg.regime_refresh_bars == 1
    assert cfg.account_size_usd == 0.0
    assert cfg.risk_per_trade_pct == 0.1
    assert cfg.max_position_notional_pct == 1.0
    assert cfg.max_fill_participation_rate == 1.0
    assert cfg.min_fill_ratio == 0.01
    assert cfg.partial_take_profit_rr == 0.25
    assert cfg.partial_take_profit_fraction == 0.95
    assert cfg.break_even_min_hold_bars == 1
    assert cfg.time_exit_bars == 1
    assert cfg.adverse_flow_threshold == 0.02
    assert cfg.adverse_flow_min_hold_bars == 1
    assert cfg.adverse_flow_consistency_threshold == 0.02
    assert cfg.adverse_book_pressure_threshold == 0.02
    assert cfg.stop_loss_mode == "strategy"
    assert cfg.fixed_stop_loss_pct == 0.0
    assert cfg.context_risk_min_sl_pct == 0.0
    assert cfg.intraday_levels_micro_confirmation_disable_for_sweep is True
    assert cfg.intraday_levels_micro_confirmation_sweep_bars == 0
    assert cfg.intraday_levels_micro_confirmation_require_intrabar is True
    assert cfg.intraday_levels_micro_confirmation_intrabar_window_seconds == 1
    assert cfg.intraday_levels_micro_confirmation_intrabar_min_coverage_points == 0
    assert cfg.intraday_levels_micro_confirmation_intrabar_min_move_pct == 0.0
    assert cfg.intraday_levels_micro_confirmation_intrabar_min_push_ratio == 1.0
    assert cfg.intraday_levels_micro_confirmation_intrabar_max_spread_bps == 0.0
    assert cfg.intrabar_eval_step_seconds == 1
    assert cfg.liquidity_sweep_detection_enabled is True
    assert cfg.sweep_min_aggression_z == -3.5
    assert cfg.sweep_min_book_pressure_z == 2.2
    assert cfg.sweep_max_price_change_pct == 0.0
    assert cfg.sweep_atr_buffer_multiplier == 0.0
    assert cfg.l2_lookback_bars == 1
    assert cfg.strategy_selection_mode == "adaptive_top_n"
    assert cfg.max_active_strategies == 20
    assert cfg.cold_start_each_day is True
    assert cfg.enable_partial_take_profit is False
    assert cfg.momentum_diversification == {}


def test_merge_overlays_selected_fields_only() -> None:
    base = TradingConfig.from_dict(
        {
            "risk_per_trade_pct": 1.25,
            "time_exit_bars": 40,
            "stop_loss_mode": "strategy",
        }
    )

    merged = base.merge(
        {
            "risk_per_trade_pct": 2.5,
            "time_exit_bars": 0,
            "stop_loss_mode": "fixed",
            "fixed_stop_loss_pct": 0.35,
        }
    )

    assert base.risk_per_trade_pct == 1.25
    assert base.time_exit_bars == 40
    assert merged.risk_per_trade_pct == 2.5
    assert merged.time_exit_bars == 1
    assert merged.stop_loss_mode == "fixed"
    assert merged.fixed_stop_loss_pct == 0.35


def test_round_trip_dict_conversion_is_stable() -> None:
    original = TradingConfig.from_dict(
        {
            "regime_detection_minutes": 17,
            "risk_per_trade_pct": 1.7,
            "strategy_selection_mode": "all_enabled",
            "max_active_strategies": 7,
            "momentum_diversification": {
                "enabled": True,
                "min_flow_score": 63.0,
            },
            "intraday_levels_micro_confirmation_disable_for_sweep": True,
            "intraday_levels_micro_confirmation_sweep_bars": 0,
            "intraday_levels_micro_confirmation_require_intrabar": True,
            "intraday_levels_micro_confirmation_intrabar_window_seconds": 10,
            "intraday_levels_micro_confirmation_intrabar_min_coverage_points": 4,
            "intraday_levels_micro_confirmation_intrabar_min_move_pct": 0.03,
            "intraday_levels_micro_confirmation_intrabar_min_push_ratio": 0.2,
            "intraday_levels_micro_confirmation_intrabar_max_spread_bps": 8.0,
            "intrabar_eval_step_seconds": 7,
            "pullback_time_exit_bars": 8,
            "pullback_entry_start_time": "09:55",
            "pullback_entry_end_time": "11:25",
            "pullback_blocked_micro_regimes": ["CHOPPY", "TRANSITION"],
            "pullback_break_even_activation_min_r": 0.45,
        }
    )

    dumped = original.to_session_params()
    reconstructed = TradingConfig.from_dict(dumped)

    assert reconstructed == original
    dumped["momentum_diversification"]["min_flow_score"] = 99.0
    assert original.momentum_diversification["min_flow_score"] == 63.0


def test_runtime_exit_formula_fields_round_trip_and_validation() -> None:
    cfg = TradingConfig.from_dict(
        {
            "break_even_activation_formula_enabled": True,
            "break_even_activation_formula": "mfe_r >= 1 and not no_go_blocked",
            "break_even_trailing_handoff_formula_enabled": True,
            "break_even_trailing_handoff_formula": "break_even_stop_active and trailing_handoff_base",
            "time_exit_formula_enabled": True,
            "time_exit_formula": "time_exit_base and not quality_favorable",
            "adverse_flow_exit_formula_enabled": True,
            "adverse_flow_exit_formula": "adverse_flow_base and not absorption_override",
        }
    )

    dumped = cfg.to_session_params()
    reconstructed = TradingConfig.from_dict(dumped)

    assert reconstructed == cfg
    assert reconstructed.break_even_activation_formula_enabled is True
    assert reconstructed.time_exit_formula_enabled is True

    try:
        TradingConfig.from_dict(
            {
                "break_even_activation_formula_enabled": True,
                "break_even_activation_formula": "unknown_var > 0",
            }
        )
    except ValueError as exc:
        assert "break_even_activation_formula" in str(exc)
    else:
        raise AssertionError("Expected invalid runtime formula to raise ValueError")
