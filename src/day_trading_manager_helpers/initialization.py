"""Initialization helpers for DayTradingManager runtime defaults."""

from __future__ import annotations

from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo


def apply_core_runtime_defaults(
    *,
    manager: Any,
    init_values: Mapping[str, Any],
    trading_costs_factory: Callable[[], Any],
    intraday_memory_factory: Callable[[], Any],
) -> None:
    """Apply normalized scalar defaults to a manager instance.

    This keeps DayTradingManager.__init__ focused on wiring services/components,
    while this helper owns value normalization and baseline runtime state.
    """

    manager.sessions = {}
    manager.regime_detection_minutes = init_values["regime_detection_minutes"]

    trading_costs = init_values.get("trading_costs")
    manager.trading_costs = trading_costs if trading_costs is not None else trading_costs_factory()

    manager.max_daily_loss = init_values["max_daily_loss"]
    manager.max_trades_per_day = init_values["max_trades_per_day"]
    manager.trade_cooldown_bars = init_values["trade_cooldown_bars"]
    manager.pending_signal_ttl_bars = max(1, int(init_values["pending_signal_ttl_bars"]))
    manager.consecutive_loss_limit = max(1, int(init_values["consecutive_loss_limit"]))
    manager.consecutive_loss_cooldown_bars = max(
        0, int(init_values["consecutive_loss_cooldown_bars"])
    )
    manager.regime_refresh_bars = max(1, int(init_values["regime_refresh_bars"]))
    manager.risk_per_trade_pct = max(0.1, float(init_values["risk_per_trade_pct"]))
    manager.max_position_notional_pct = max(1.0, float(init_values["max_position_notional_pct"]))
    manager.max_fill_participation_rate = 0.20
    manager.min_fill_ratio = 0.35
    manager.time_exit_bars = max(1, int(init_values["time_exit_bars"]))
    manager.enable_partial_take_profit = bool(init_values["enable_partial_take_profit"])
    manager.partial_take_profit_rr = max(0.25, float(init_values["partial_take_profit_rr"]))
    manager.partial_take_profit_fraction = min(
        0.95,
        max(0.05, float(init_values["partial_take_profit_fraction"])),
    )
    manager.trailing_activation_pct = max(0.0, float(init_values["trailing_activation_pct"]))
    manager.break_even_buffer_pct = max(0.0, float(init_values["break_even_buffer_pct"]))
    manager.break_even_min_hold_bars = max(1, int(init_values["break_even_min_hold_bars"]))
    manager.break_even_activation_min_mfe_pct = max(
        0.0,
        float(init_values["break_even_activation_min_mfe_pct"]),
    )
    manager.break_even_activation_min_r = max(0.0, float(init_values["break_even_activation_min_r"]))
    manager.break_even_activation_min_r_trending_5m = max(
        0.0,
        float(init_values["break_even_activation_min_r_trending_5m"]),
    )
    manager.break_even_activation_min_r_choppy_5m = max(
        0.0,
        float(init_values["break_even_activation_min_r_choppy_5m"]),
    )
    manager.break_even_activation_use_levels = bool(init_values["break_even_activation_use_levels"])
    manager.break_even_activation_use_l2 = bool(init_values["break_even_activation_use_l2"])
    manager.break_even_level_buffer_pct = max(
        0.0,
        float(init_values["break_even_level_buffer_pct"]),
    )
    manager.break_even_level_max_distance_pct = max(
        0.01,
        float(init_values["break_even_level_max_distance_pct"]),
    )
    manager.break_even_level_min_confluence = max(
        0,
        int(init_values["break_even_level_min_confluence"]),
    )
    manager.break_even_level_min_tests = max(0, int(init_values["break_even_level_min_tests"]))
    manager.break_even_l2_signed_aggression_min = max(
        0.0,
        float(init_values["break_even_l2_signed_aggression_min"]),
    )
    manager.break_even_l2_imbalance_min = max(0.0, float(init_values["break_even_l2_imbalance_min"]))
    manager.break_even_l2_book_pressure_min = max(
        0.0,
        float(init_values["break_even_l2_book_pressure_min"]),
    )
    manager.break_even_l2_spread_bps_max = max(
        0.0,
        float(init_values["break_even_l2_spread_bps_max"]),
    )
    manager.break_even_costs_pct = max(0.0, float(init_values["break_even_costs_pct"]))
    manager.break_even_min_buffer_pct = max(0.0, float(init_values["break_even_min_buffer_pct"]))
    manager.break_even_atr_buffer_k = max(0.0, float(init_values["break_even_atr_buffer_k"]))
    manager.break_even_5m_atr_buffer_k = max(0.0, float(init_values["break_even_5m_atr_buffer_k"]))
    manager.break_even_tick_size = max(0.0, float(init_values["break_even_tick_size"]))
    manager.break_even_min_tick_buffer = max(0, int(init_values["break_even_min_tick_buffer"]))
    manager.break_even_anti_spike_bars = max(0, int(init_values["break_even_anti_spike_bars"]))
    manager.break_even_anti_spike_hits_required = max(
        1,
        int(init_values["break_even_anti_spike_hits_required"]),
    )
    manager.break_even_anti_spike_require_close_beyond = bool(
        init_values["break_even_anti_spike_require_close_beyond"]
    )
    manager.break_even_5m_no_go_proximity_pct = max(
        0.0,
        float(init_values["break_even_5m_no_go_proximity_pct"]),
    )
    manager.break_even_5m_mfe_atr_factor = max(
        0.0,
        float(init_values["break_even_5m_mfe_atr_factor"]),
    )
    manager.break_even_5m_l2_bias_threshold = max(
        0.0,
        float(init_values["break_even_5m_l2_bias_threshold"]),
    )
    manager.break_even_5m_l2_bias_tighten_factor = max(
        0.1,
        min(2.0, float(init_values["break_even_5m_l2_bias_tighten_factor"])),
    )
    manager.trailing_enabled_in_choppy = bool(init_values["trailing_enabled_in_choppy"])
    manager.adverse_flow_exit_enabled = bool(init_values["adverse_flow_exit_enabled"])
    manager.adverse_flow_exit_threshold = max(0.02, float(init_values["adverse_flow_exit_threshold"]))
    manager.adverse_flow_min_hold_bars = max(1, int(init_values["adverse_flow_min_hold_bars"]))
    manager.adverse_flow_consistency_threshold = max(
        0.02,
        float(init_values["adverse_flow_consistency_threshold"]),
    )
    manager.adverse_book_pressure_threshold = max(
        0.02,
        float(init_values["adverse_book_pressure_threshold"]),
    )
    manager.portfolio_drawdown_halt_pct = max(
        0.0,
        float(init_values["portfolio_drawdown_halt_pct"]),
    )
    manager.headwind_activation_score = min(
        0.95,
        max(0.0, float(init_values["headwind_activation_score"])),
    )
    manager.market_tz = ZoneInfo("America/New_York")
    manager.run_defaults = {}
    manager.run_equity_state = {}
    manager.intraday_memory = intraday_memory_factory()
    manager.last_trade_bar_index = {}


__all__ = ["apply_core_runtime_defaults"]
