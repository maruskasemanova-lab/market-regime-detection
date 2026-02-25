"""Helpers for building DayTradingManager run default overrides."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple


RUN_DEFAULT_EXPLICIT_OVERRIDE_FIELDS: Tuple[str, ...] = (
    "regime_detection_minutes",
    "regime_refresh_bars",
    "account_size_usd",
    "risk_per_trade_pct",
    "max_position_notional_pct",
    "max_fill_participation_rate",
    "min_fill_ratio",
    "enable_partial_take_profit",
    "partial_take_profit_rr",
    "partial_take_profit_fraction",
    "trailing_activation_pct",
    "break_even_buffer_pct",
    "break_even_min_hold_bars",
    "break_even_activation_min_mfe_pct",
    "break_even_activation_min_r",
    "break_even_activation_min_r_trending_5m",
    "break_even_activation_min_r_choppy_5m",
    "break_even_activation_use_levels",
    "break_even_activation_use_l2",
    "break_even_level_buffer_pct",
    "break_even_level_max_distance_pct",
    "break_even_level_min_confluence",
    "break_even_level_min_tests",
    "break_even_l2_signed_aggression_min",
    "break_even_l2_imbalance_min",
    "break_even_l2_book_pressure_min",
    "break_even_l2_spread_bps_max",
    "break_even_costs_pct",
    "break_even_min_buffer_pct",
    "break_even_atr_buffer_k",
    "break_even_5m_atr_buffer_k",
    "break_even_tick_size",
    "break_even_min_tick_buffer",
    "break_even_anti_spike_bars",
    "break_even_anti_spike_hits_required",
    "break_even_anti_spike_require_close_beyond",
    "break_even_5m_no_go_proximity_pct",
    "break_even_5m_mfe_atr_factor",
    "break_even_5m_l2_bias_threshold",
    "break_even_5m_l2_bias_tighten_factor",
    "trailing_enabled_in_choppy",
    "time_exit_bars",
    "adverse_flow_exit_enabled",
    "adverse_flow_threshold",
    "adverse_flow_min_hold_bars",
    "adverse_flow_consistency_threshold",
    "adverse_book_pressure_threshold",
    "stop_loss_mode",
    "fixed_stop_loss_pct",
    "l2_confirm_enabled",
    "l2_min_delta",
    "l2_min_imbalance",
    "l2_min_iceberg_bias",
    "l2_lookback_bars",
    "l2_min_participation_ratio",
    "l2_min_directional_consistency",
    "l2_min_signed_aggression",
    "cold_start_each_day",
    "strategy_selection_mode",
    "max_active_strategies",
)


def collect_non_none_run_default_overrides(
    values: Mapping[str, Any],
) -> Dict[str, Any]:
    """Pick supported explicit overrides from a local variable mapping."""

    return {
        field_name: values[field_name]
        for field_name in RUN_DEFAULT_EXPLICIT_OVERRIDE_FIELDS
        if values.get(field_name) is not None
    }
