"""Session config payload builders for DayTradingManager."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from typing import Any, Dict, Tuple


_SESSION_BASE_CONFIG_ATTR_MAP: Tuple[Tuple[str, str], ...] = (
    ("regime_refresh_bars", "regime_refresh_bars"),
    ("risk_per_trade_pct", "risk_per_trade_pct"),
    ("max_position_notional_pct", "max_position_notional_pct"),
    ("max_fill_participation_rate", "max_fill_participation_rate"),
    ("min_fill_ratio", "min_fill_ratio"),
    ("enable_partial_take_profit", "enable_partial_take_profit"),
    ("partial_take_profit_rr", "partial_take_profit_rr"),
    ("partial_take_profit_fraction", "partial_take_profit_fraction"),
    ("trailing_activation_pct", "trailing_activation_pct"),
    ("break_even_buffer_pct", "break_even_buffer_pct"),
    ("break_even_min_hold_bars", "break_even_min_hold_bars"),
    ("break_even_activation_min_mfe_pct", "break_even_activation_min_mfe_pct"),
    ("break_even_activation_min_r", "break_even_activation_min_r"),
    ("break_even_activation_min_r_trending_5m", "break_even_activation_min_r_trending_5m"),
    ("break_even_activation_min_r_choppy_5m", "break_even_activation_min_r_choppy_5m"),
    ("break_even_activation_use_levels", "break_even_activation_use_levels"),
    ("break_even_activation_use_l2", "break_even_activation_use_l2"),
    ("break_even_level_buffer_pct", "break_even_level_buffer_pct"),
    ("break_even_level_max_distance_pct", "break_even_level_max_distance_pct"),
    ("break_even_level_min_confluence", "break_even_level_min_confluence"),
    ("break_even_level_min_tests", "break_even_level_min_tests"),
    ("break_even_l2_signed_aggression_min", "break_even_l2_signed_aggression_min"),
    ("break_even_l2_imbalance_min", "break_even_l2_imbalance_min"),
    ("break_even_l2_book_pressure_min", "break_even_l2_book_pressure_min"),
    ("break_even_l2_spread_bps_max", "break_even_l2_spread_bps_max"),
    ("break_even_costs_pct", "break_even_costs_pct"),
    ("break_even_min_buffer_pct", "break_even_min_buffer_pct"),
    ("break_even_atr_buffer_k", "break_even_atr_buffer_k"),
    ("break_even_5m_atr_buffer_k", "break_even_5m_atr_buffer_k"),
    ("break_even_tick_size", "break_even_tick_size"),
    ("break_even_min_tick_buffer", "break_even_min_tick_buffer"),
    ("break_even_anti_spike_bars", "break_even_anti_spike_bars"),
    ("break_even_anti_spike_hits_required", "break_even_anti_spike_hits_required"),
    (
        "break_even_anti_spike_require_close_beyond",
        "break_even_anti_spike_require_close_beyond",
    ),
    ("break_even_5m_no_go_proximity_pct", "break_even_5m_no_go_proximity_pct"),
    ("break_even_5m_mfe_atr_factor", "break_even_5m_mfe_atr_factor"),
    ("break_even_5m_l2_bias_threshold", "break_even_5m_l2_bias_threshold"),
    ("break_even_5m_l2_bias_tighten_factor", "break_even_5m_l2_bias_tighten_factor"),
    ("trailing_enabled_in_choppy", "trailing_enabled_in_choppy"),
    ("time_exit_bars", "time_exit_bars"),
    ("adverse_flow_exit_enabled", "adverse_flow_exit_enabled"),
    ("adverse_flow_threshold", "adverse_flow_exit_threshold"),
    ("adverse_flow_min_hold_bars", "adverse_flow_min_hold_bars"),
    ("adverse_flow_consistency_threshold", "adverse_flow_consistency_threshold"),
    ("adverse_book_pressure_threshold", "adverse_book_pressure_threshold"),
    ("stop_loss_mode", "stop_loss_mode"),
    ("fixed_stop_loss_pct", "fixed_stop_loss_pct"),
)


def build_session_base_config_payload(
    manager: Any,
    *,
    regime_detection_minutes: int,
) -> Dict[str, Any]:
    """Build the canonical per-session config payload from manager defaults."""

    payload: Dict[str, Any] = {
        "regime_detection_minutes": regime_detection_minutes,
    }
    payload.update(
        {
            config_key: getattr(manager, attr_name)
            for config_key, attr_name in _SESSION_BASE_CONFIG_ATTR_MAP
        }
    )
    return payload


def apply_ticker_adverse_flow_overrides(
    config_payload: MutableMapping[str, Any],
    ticker_cfg: Mapping[str, Any],
    *,
    safe_float: Callable[[Any, float], float],
) -> None:
    """Apply ticker-level adverse-flow threshold overrides."""

    if not ticker_cfg:
        return

    for key in (
        "adverse_flow_consistency_threshold",
        "adverse_book_pressure_threshold",
    ):
        config_payload[key] = safe_float(ticker_cfg.get(key), config_payload[key])


def apply_ticker_l2_overrides(
    config_payload: MutableMapping[str, Any],
    ticker_cfg: Mapping[str, Any],
    *,
    safe_float: Callable[[Any, float], float],
) -> None:
    """Apply ticker-level L2 thresholds and flags."""

    l2_cfg = ticker_cfg.get("l2", {})
    if not isinstance(l2_cfg, Mapping):
        return

    config_payload.update(
        {
            "l2_confirm_enabled": bool(l2_cfg.get("confirm_enabled", False)),
            "l2_min_delta": safe_float(l2_cfg.get("min_delta"), 0.0) or 0.0,
            "l2_min_imbalance": safe_float(l2_cfg.get("min_imbalance"), 0.0) or 0.0,
            "l2_min_iceberg_bias": safe_float(l2_cfg.get("min_iceberg_bias"), 0.0) or 0.0,
            "l2_lookback_bars": max(1, int(safe_float(l2_cfg.get("lookback_bars"), 3) or 3)),
            "l2_min_participation_ratio": safe_float(
                l2_cfg.get("min_participation_ratio"), 0.0
            )
            or 0.0,
            "l2_min_directional_consistency": safe_float(
                l2_cfg.get("min_directional_consistency"), 0.0
            )
            or 0.0,
            "l2_min_signed_aggression": safe_float(
                l2_cfg.get("min_signed_aggression"), 0.0
            )
            or 0.0,
        }
    )


def build_ticker_session_config_payload(
    manager: Any,
    *,
    ticker_cfg: Mapping[str, Any],
    regime_detection_minutes: int,
) -> Dict[str, Any]:
    """Build per-session config payload with ticker-specific overrides applied."""

    config_payload = build_session_base_config_payload(
        manager,
        regime_detection_minutes=regime_detection_minutes,
    )

    apply_ticker_adverse_flow_overrides(
        config_payload,
        ticker_cfg,
        safe_float=manager._safe_float,
    )
    apply_ticker_l2_overrides(
        config_payload,
        ticker_cfg,
        safe_float=manager._safe_float,
    )
    config_payload["strategy_selection_mode"] = manager._normalize_strategy_selection_mode(
        ticker_cfg.get("strategy_selection_mode")
    )
    config_payload["max_active_strategies"] = manager._normalize_max_active_strategies(
        ticker_cfg.get("max_active_strategies"),
        default=3,
    )
    return config_payload
