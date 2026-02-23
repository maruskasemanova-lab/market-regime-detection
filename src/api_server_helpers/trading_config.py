"""Helpers for reading/applying global trading config on the manager."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from ..trading_config import TradingConfig


_MANAGER_ATTR_ALIAS = {
    "adverse_flow_threshold": "adverse_flow_exit_threshold",
}

_GLOBAL_TRADING_CONFIG_KEYS = (
    "regime_detection_minutes",
    "regime_refresh_bars",
    "risk_per_trade_pct",
    "max_position_notional_pct",
    "max_fill_participation_rate",
    "min_fill_ratio",
    "time_exit_bars",
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
    "adverse_flow_exit_enabled",
    "adverse_flow_threshold",
    "adverse_flow_min_hold_bars",
    "adverse_flow_consistency_threshold",
    "adverse_book_pressure_threshold",
    "stop_loss_mode",
    "fixed_stop_loss_pct",
)


def _manager_attr_name(config_key: str) -> str:
    return _MANAGER_ATTR_ALIAS.get(config_key, config_key)


def read_global_trading_config(manager: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key in _GLOBAL_TRADING_CONFIG_KEYS:
        payload[key] = getattr(manager, _manager_attr_name(key))
    return payload


def apply_global_trading_config(manager: Any, canonical: TradingConfig) -> None:
    session_params = canonical.to_session_params()
    for key in _GLOBAL_TRADING_CONFIG_KEYS:
        if key not in session_params:
            continue
        setattr(manager, _manager_attr_name(key), session_params[key])


def read_trading_limits(manager: Any) -> Dict[str, Any]:
    return {
        "max_daily_loss": manager.max_daily_loss,
        "max_trades_per_day": manager.max_trades_per_day,
        "trade_cooldown_bars": manager.trade_cooldown_bars,
        "pending_signal_ttl_bars": manager.pending_signal_ttl_bars,
        "consecutive_loss_limit": manager.consecutive_loss_limit,
        "consecutive_loss_cooldown_bars": manager.consecutive_loss_cooldown_bars,
        "portfolio_drawdown_halt_pct": manager.portfolio_drawdown_halt_pct,
        "headwind_activation_score": manager.headwind_activation_score,
    }


def apply_trading_limits(manager: Any, config_model: Mapping[str, Any]) -> None:
    manager.max_daily_loss = config_model["max_daily_loss"]
    manager.max_trades_per_day = config_model["max_trades_per_day"]
    manager.trade_cooldown_bars = config_model["trade_cooldown_bars"]
    manager.pending_signal_ttl_bars = max(1, int(config_model["pending_signal_ttl_bars"]))
    manager.consecutive_loss_limit = max(1, int(config_model["consecutive_loss_limit"]))
    manager.consecutive_loss_cooldown_bars = max(
        0,
        int(config_model["consecutive_loss_cooldown_bars"]),
    )
    manager.portfolio_drawdown_halt_pct = max(
        0.0,
        float(config_model["portfolio_drawdown_halt_pct"]),
    )
    manager.headwind_activation_score = max(
        0.0,
        min(0.95, float(config_model["headwind_activation_score"])),
    )


__all__ = [
    "apply_global_trading_config",
    "apply_trading_limits",
    "read_global_trading_config",
    "read_trading_limits",
]
