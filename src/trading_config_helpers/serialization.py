"""Serialization and normalization helpers for TradingConfig."""

from __future__ import annotations

import copy
from typing import Any, Dict, TYPE_CHECKING

from ..runtime_exit_formulas import normalize_runtime_exit_formula_fields

if TYPE_CHECKING:
    from ..trading_config import TradingConfig


def build_trading_config_from_dict(cls: type["TradingConfig"], d: Dict[str, Any]) -> "TradingConfig":
    raw = d if isinstance(d, dict) else {}
    defaults = cls()
    runtime_formula_fields = normalize_runtime_exit_formula_fields(raw)

    stop_mode = str(raw.get("stop_loss_mode", defaults.stop_loss_mode) or defaults.stop_loss_mode).strip().lower()
    if stop_mode not in cls.VALID_STOP_LOSS_MODES:
        stop_mode = defaults.stop_loss_mode

    selection_mode = str(
        raw.get("strategy_selection_mode", defaults.strategy_selection_mode)
        or defaults.strategy_selection_mode
    ).strip().lower()
    if selection_mode not in cls.VALID_STRATEGY_SELECTION_MODES:
        selection_mode = defaults.strategy_selection_mode

    momentum = raw.get("momentum_diversification", defaults.momentum_diversification)
    if not isinstance(momentum, dict):
        momentum = {}

    micro_mode = str(
        raw.get("micro_confirmation_mode", defaults.micro_confirmation_mode)
        or defaults.micro_confirmation_mode
    ).strip().lower()
    if micro_mode not in cls.VALID_MICRO_CONFIRMATION_MODES:
        micro_mode = defaults.micro_confirmation_mode

    return cls(
        regime_detection_minutes=max(
            1,
            cls._to_int(raw.get("regime_detection_minutes"), defaults.regime_detection_minutes),
        ),
        regime_refresh_bars=max(
            1,
            cls._to_int(raw.get("regime_refresh_bars"), defaults.regime_refresh_bars),
        ),
        account_size_usd=max(
            0.0,
            cls._to_float(raw.get("account_size_usd"), defaults.account_size_usd),
        ),
        risk_per_trade_pct=max(
            0.1,
            cls._to_float(raw.get("risk_per_trade_pct"), defaults.risk_per_trade_pct),
        ),
        max_position_notional_pct=max(
            1.0,
            cls._to_float(
                raw.get("max_position_notional_pct"),
                defaults.max_position_notional_pct,
            ),
        ),
        max_fill_participation_rate=min(
            1.0,
            max(
                0.01,
                cls._to_float(
                    raw.get("max_fill_participation_rate"),
                    defaults.max_fill_participation_rate,
                ),
            ),
        ),
        min_fill_ratio=min(
            1.0,
            max(
                0.01,
                cls._to_float(raw.get("min_fill_ratio"), defaults.min_fill_ratio),
            ),
        ),
        enable_partial_take_profit=cls._to_bool(
            raw.get("enable_partial_take_profit"),
            defaults.enable_partial_take_profit,
        ),
        partial_take_profit_rr=max(
            0.25,
            cls._to_float(raw.get("partial_take_profit_rr"), defaults.partial_take_profit_rr),
        ),
        partial_take_profit_fraction=min(
            0.95,
            max(
                0.05,
                cls._to_float(
                    raw.get("partial_take_profit_fraction"),
                    defaults.partial_take_profit_fraction,
                ),
            ),
        ),
        partial_flow_deterioration_min_r=max(
            0.0,
            cls._to_float(
                raw.get("partial_flow_deterioration_min_r"),
                defaults.partial_flow_deterioration_min_r,
            ),
        ),
        partial_flow_deterioration_skip_be=cls._to_bool(
            raw.get("partial_flow_deterioration_skip_be"),
            defaults.partial_flow_deterioration_skip_be,
        ),
        trailing_activation_pct=max(
            0.0,
            cls._to_float(raw.get("trailing_activation_pct"), defaults.trailing_activation_pct),
        ),
        break_even_buffer_pct=max(
            0.0,
            cls._to_float(raw.get("break_even_buffer_pct"), defaults.break_even_buffer_pct),
        ),
        break_even_min_hold_bars=max(
            1,
            cls._to_int(
                raw.get("break_even_min_hold_bars"),
                defaults.break_even_min_hold_bars,
            ),
        ),
        break_even_activation_min_mfe_pct=max(
            0.0,
            cls._to_float(
                raw.get("break_even_activation_min_mfe_pct"),
                defaults.break_even_activation_min_mfe_pct,
            ),
        ),
        break_even_activation_min_r=max(
            0.0,
            cls._to_float(
                raw.get("break_even_activation_min_r"),
                defaults.break_even_activation_min_r,
            ),
        ),
        break_even_activation_min_r_trending_5m=max(
            0.0,
            cls._to_float(
                raw.get("break_even_activation_min_r_trending_5m"),
                defaults.break_even_activation_min_r_trending_5m,
            ),
        ),
        break_even_activation_min_r_choppy_5m=max(
            0.0,
            cls._to_float(
                raw.get("break_even_activation_min_r_choppy_5m"),
                defaults.break_even_activation_min_r_choppy_5m,
            ),
        ),
        break_even_activation_use_levels=cls._to_bool(
            raw.get("break_even_activation_use_levels"),
            defaults.break_even_activation_use_levels,
        ),
        break_even_activation_use_l2=cls._to_bool(
            raw.get("break_even_activation_use_l2"),
            defaults.break_even_activation_use_l2,
        ),
        break_even_level_buffer_pct=max(
            0.0,
            cls._to_float(
                raw.get("break_even_level_buffer_pct"),
                defaults.break_even_level_buffer_pct,
            ),
        ),
        break_even_level_max_distance_pct=max(
            0.01,
            cls._to_float(
                raw.get("break_even_level_max_distance_pct"),
                defaults.break_even_level_max_distance_pct,
            ),
        ),
        break_even_level_min_confluence=max(
            0,
            cls._to_int(
                raw.get("break_even_level_min_confluence"),
                defaults.break_even_level_min_confluence,
            ),
        ),
        break_even_level_min_tests=max(
            0,
            cls._to_int(
                raw.get("break_even_level_min_tests"),
                defaults.break_even_level_min_tests,
            ),
        ),
        break_even_l2_signed_aggression_min=max(
            0.0,
            cls._to_float(
                raw.get("break_even_l2_signed_aggression_min"),
                defaults.break_even_l2_signed_aggression_min,
            ),
        ),
        break_even_l2_imbalance_min=max(
            0.0,
            cls._to_float(
                raw.get("break_even_l2_imbalance_min"),
                defaults.break_even_l2_imbalance_min,
            ),
        ),
        break_even_l2_book_pressure_min=max(
            0.0,
            cls._to_float(
                raw.get("break_even_l2_book_pressure_min"),
                defaults.break_even_l2_book_pressure_min,
            ),
        ),
        break_even_l2_spread_bps_max=max(
            0.0,
            cls._to_float(
                raw.get("break_even_l2_spread_bps_max"),
                defaults.break_even_l2_spread_bps_max,
            ),
        ),
        break_even_costs_pct=max(
            0.0,
            cls._to_float(
                raw.get("break_even_costs_pct"),
                defaults.break_even_costs_pct,
            ),
        ),
        break_even_min_buffer_pct=max(
            0.0,
            cls._to_float(
                raw.get("break_even_min_buffer_pct"),
                defaults.break_even_min_buffer_pct,
            ),
        ),
        break_even_atr_buffer_k=max(
            0.0,
            cls._to_float(
                raw.get("break_even_atr_buffer_k"),
                defaults.break_even_atr_buffer_k,
            ),
        ),
        break_even_5m_atr_buffer_k=max(
            0.0,
            cls._to_float(
                raw.get("break_even_5m_atr_buffer_k"),
                defaults.break_even_5m_atr_buffer_k,
            ),
        ),
        break_even_tick_size=max(
            0.0,
            cls._to_float(
                raw.get("break_even_tick_size"),
                defaults.break_even_tick_size,
            ),
        ),
        break_even_min_tick_buffer=max(
            0,
            cls._to_int(
                raw.get("break_even_min_tick_buffer"),
                defaults.break_even_min_tick_buffer,
            ),
        ),
        break_even_anti_spike_bars=max(
            0,
            cls._to_int(
                raw.get("break_even_anti_spike_bars"),
                defaults.break_even_anti_spike_bars,
            ),
        ),
        break_even_anti_spike_hits_required=max(
            1,
            cls._to_int(
                raw.get("break_even_anti_spike_hits_required"),
                defaults.break_even_anti_spike_hits_required,
            ),
        ),
        break_even_anti_spike_require_close_beyond=cls._to_bool(
            raw.get("break_even_anti_spike_require_close_beyond"),
            defaults.break_even_anti_spike_require_close_beyond,
        ),
        break_even_5m_no_go_proximity_pct=max(
            0.0,
            cls._to_float(
                raw.get("break_even_5m_no_go_proximity_pct"),
                defaults.break_even_5m_no_go_proximity_pct,
            ),
        ),
        break_even_5m_mfe_atr_factor=max(
            0.0,
            cls._to_float(
                raw.get("break_even_5m_mfe_atr_factor"),
                defaults.break_even_5m_mfe_atr_factor,
            ),
        ),
        break_even_5m_l2_bias_threshold=max(
            0.0,
            cls._to_float(
                raw.get("break_even_5m_l2_bias_threshold"),
                defaults.break_even_5m_l2_bias_threshold,
            ),
        ),
        break_even_5m_l2_bias_tighten_factor=max(
            0.1,
            min(
                2.0,
                cls._to_float(
                    raw.get("break_even_5m_l2_bias_tighten_factor"),
                    defaults.break_even_5m_l2_bias_tighten_factor,
                ),
            ),
        ),
        break_even_movement_formula_enabled=cls._to_bool(
            runtime_formula_fields.get(
                "break_even_movement_formula_enabled",
                raw.get("break_even_movement_formula_enabled"),
            ),
            defaults.break_even_movement_formula_enabled,
        ),
        break_even_movement_formula=cls._to_text(
            runtime_formula_fields.get(
                "break_even_movement_formula",
                raw.get("break_even_movement_formula"),
            ),
            defaults.break_even_movement_formula,
        ),
        break_even_proof_formula_enabled=cls._to_bool(
            runtime_formula_fields.get(
                "break_even_proof_formula_enabled",
                raw.get("break_even_proof_formula_enabled"),
            ),
            defaults.break_even_proof_formula_enabled,
        ),
        break_even_proof_formula=cls._to_text(
            runtime_formula_fields.get(
                "break_even_proof_formula",
                raw.get("break_even_proof_formula"),
            ),
            defaults.break_even_proof_formula,
        ),
        break_even_activation_formula_enabled=cls._to_bool(
            runtime_formula_fields.get(
                "break_even_activation_formula_enabled",
                raw.get("break_even_activation_formula_enabled"),
            ),
            defaults.break_even_activation_formula_enabled,
        ),
        break_even_activation_formula=cls._to_text(
            runtime_formula_fields.get(
                "break_even_activation_formula",
                raw.get("break_even_activation_formula"),
            ),
            defaults.break_even_activation_formula,
        ),
        break_even_trailing_handoff_formula_enabled=cls._to_bool(
            runtime_formula_fields.get(
                "break_even_trailing_handoff_formula_enabled",
                raw.get("break_even_trailing_handoff_formula_enabled"),
            ),
            defaults.break_even_trailing_handoff_formula_enabled,
        ),
        break_even_trailing_handoff_formula=cls._to_text(
            runtime_formula_fields.get(
                "break_even_trailing_handoff_formula",
                raw.get("break_even_trailing_handoff_formula"),
            ),
            defaults.break_even_trailing_handoff_formula,
        ),
        trailing_enabled_in_choppy=cls._to_bool(
            raw.get("trailing_enabled_in_choppy"),
            defaults.trailing_enabled_in_choppy,
        ),
        time_exit_bars=max(
            1,
            cls._to_int(raw.get("time_exit_bars"), defaults.time_exit_bars),
        ),
        time_exit_formula_enabled=cls._to_bool(
            runtime_formula_fields.get(
                "time_exit_formula_enabled",
                raw.get("time_exit_formula_enabled"),
            ),
            defaults.time_exit_formula_enabled,
        ),
        time_exit_formula=cls._to_text(
            runtime_formula_fields.get(
                "time_exit_formula",
                raw.get("time_exit_formula"),
            ),
            defaults.time_exit_formula,
        ),
        adverse_flow_exit_enabled=cls._to_bool(
            raw.get("adverse_flow_exit_enabled"),
            defaults.adverse_flow_exit_enabled,
        ),
        adverse_flow_threshold=max(
            0.02,
            cls._to_float(raw.get("adverse_flow_threshold"), defaults.adverse_flow_threshold),
        ),
        adverse_flow_min_hold_bars=max(
            1,
            cls._to_int(raw.get("adverse_flow_min_hold_bars"), defaults.adverse_flow_min_hold_bars),
        ),
        adverse_flow_consistency_threshold=max(
            0.02,
            cls._to_float(
                raw.get("adverse_flow_consistency_threshold"),
                defaults.adverse_flow_consistency_threshold,
            ),
        ),
        adverse_book_pressure_threshold=max(
            0.02,
            cls._to_float(
                raw.get("adverse_book_pressure_threshold"),
                defaults.adverse_book_pressure_threshold,
            ),
        ),
        adverse_flow_exit_formula_enabled=cls._to_bool(
            runtime_formula_fields.get(
                "adverse_flow_exit_formula_enabled",
                raw.get("adverse_flow_exit_formula_enabled"),
            ),
            defaults.adverse_flow_exit_formula_enabled,
        ),
        adverse_flow_exit_formula=cls._to_text(
            runtime_formula_fields.get(
                "adverse_flow_exit_formula",
                raw.get("adverse_flow_exit_formula"),
            ),
            defaults.adverse_flow_exit_formula,
        ),
        stop_loss_mode=stop_mode,
        fixed_stop_loss_pct=max(
            0.0,
            cls._to_float(raw.get("fixed_stop_loss_pct"), defaults.fixed_stop_loss_pct),
        ),
        l2_confirm_enabled=cls._to_bool(raw.get("l2_confirm_enabled"), defaults.l2_confirm_enabled),
        l2_min_delta=cls._to_float(raw.get("l2_min_delta"), defaults.l2_min_delta),
        l2_min_imbalance=cls._to_float(raw.get("l2_min_imbalance"), defaults.l2_min_imbalance),
        l2_min_iceberg_bias=cls._to_float(raw.get("l2_min_iceberg_bias"), defaults.l2_min_iceberg_bias),
        l2_lookback_bars=max(
            1,
            cls._to_int(raw.get("l2_lookback_bars"), defaults.l2_lookback_bars),
        ),
        l2_min_participation_ratio=cls._to_float(
            raw.get("l2_min_participation_ratio"),
            defaults.l2_min_participation_ratio,
        ),
        l2_min_directional_consistency=cls._to_float(
            raw.get("l2_min_directional_consistency"),
            defaults.l2_min_directional_consistency,
        ),
        l2_min_signed_aggression=cls._to_float(
            raw.get("l2_min_signed_aggression"),
            defaults.l2_min_signed_aggression,
        ),
        tcbbo_gate_enabled=cls._to_bool(
            raw.get("tcbbo_gate_enabled"), defaults.tcbbo_gate_enabled
        ),
        tcbbo_min_net_premium=cls._to_float(
            raw.get("tcbbo_min_net_premium"), defaults.tcbbo_min_net_premium
        ),
        tcbbo_sweep_boost=max(
            0.0,
            min(20.0, cls._to_float(raw.get("tcbbo_sweep_boost"), defaults.tcbbo_sweep_boost)),
        ),
        tcbbo_lookback_bars=max(
            1, cls._to_int(raw.get("tcbbo_lookback_bars"), defaults.tcbbo_lookback_bars)
        ),
        intraday_levels_enabled=cls._to_bool(
            raw.get("intraday_levels_enabled"), defaults.intraday_levels_enabled
        ),
        intraday_levels_swing_left_bars=max(
            1,
            cls._to_int(
                raw.get("intraday_levels_swing_left_bars"),
                defaults.intraday_levels_swing_left_bars,
            ),
        ),
        intraday_levels_swing_right_bars=max(
            1,
            cls._to_int(
                raw.get("intraday_levels_swing_right_bars"),
                defaults.intraday_levels_swing_right_bars,
            ),
        ),
        intraday_levels_test_tolerance_pct=max(
            0.0,
            cls._to_float(
                raw.get("intraday_levels_test_tolerance_pct"),
                defaults.intraday_levels_test_tolerance_pct,
            ),
        ),
        intraday_levels_break_tolerance_pct=max(
            0.0,
            cls._to_float(
                raw.get("intraday_levels_break_tolerance_pct"),
                defaults.intraday_levels_break_tolerance_pct,
            ),
        ),
        intraday_levels_breakout_volume_lookback=max(
            2,
            cls._to_int(
                raw.get("intraday_levels_breakout_volume_lookback"),
                defaults.intraday_levels_breakout_volume_lookback,
            ),
        ),
        intraday_levels_breakout_volume_multiplier=max(
            1.0,
            cls._to_float(
                raw.get("intraday_levels_breakout_volume_multiplier"),
                defaults.intraday_levels_breakout_volume_multiplier,
            ),
        ),
        intraday_levels_volume_profile_bin_size_pct=max(
            0.01,
            cls._to_float(
                raw.get("intraday_levels_volume_profile_bin_size_pct"),
                defaults.intraday_levels_volume_profile_bin_size_pct,
            ),
        ),
        intraday_levels_value_area_pct=min(
            0.95,
            max(
                0.5,
                cls._to_float(
                    raw.get("intraday_levels_value_area_pct"),
                    defaults.intraday_levels_value_area_pct,
                ),
            ),
        ),
        intraday_levels_entry_quality_enabled=cls._to_bool(
            raw.get("intraday_levels_entry_quality_enabled"),
            defaults.intraday_levels_entry_quality_enabled,
        ),
        intraday_levels_min_levels_for_context=max(
            1,
            cls._to_int(
                raw.get("intraday_levels_min_levels_for_context"),
                defaults.intraday_levels_min_levels_for_context,
            ),
        ),
        intraday_levels_entry_tolerance_pct=max(
            0.01,
            cls._to_float(
                raw.get("intraday_levels_entry_tolerance_pct"),
                defaults.intraday_levels_entry_tolerance_pct,
            ),
        ),
        intraday_levels_break_cooldown_bars=max(
            1,
            cls._to_int(
                raw.get("intraday_levels_break_cooldown_bars"),
                defaults.intraday_levels_break_cooldown_bars,
            ),
        ),
        intraday_levels_rotation_max_tests=max(
            1,
            cls._to_int(
                raw.get("intraday_levels_rotation_max_tests"),
                defaults.intraday_levels_rotation_max_tests,
            ),
        ),
        intraday_levels_rotation_volume_max_ratio=min(
            2.0,
            max(
                0.1,
                cls._to_float(
                    raw.get("intraday_levels_rotation_volume_max_ratio"),
                    defaults.intraday_levels_rotation_volume_max_ratio,
                ),
            ),
        ),
        intraday_levels_recent_bounce_lookback_bars=max(
            1,
            cls._to_int(
                raw.get("intraday_levels_recent_bounce_lookback_bars"),
                defaults.intraday_levels_recent_bounce_lookback_bars,
            ),
        ),
        intraday_levels_require_recent_bounce_for_mean_reversion=cls._to_bool(
            raw.get("intraday_levels_require_recent_bounce_for_mean_reversion"),
            defaults.intraday_levels_require_recent_bounce_for_mean_reversion,
        ),
        intraday_levels_momentum_break_max_age_bars=max(
            1,
            cls._to_int(
                raw.get("intraday_levels_momentum_break_max_age_bars"),
                defaults.intraday_levels_momentum_break_max_age_bars,
            ),
        ),
        intraday_levels_momentum_min_room_pct=max(
            0.01,
            cls._to_float(
                raw.get("intraday_levels_momentum_min_room_pct"),
                defaults.intraday_levels_momentum_min_room_pct,
            ),
        ),
        intraday_levels_momentum_min_broken_ratio=min(
            1.0,
            max(
                0.0,
                cls._to_float(
                    raw.get("intraday_levels_momentum_min_broken_ratio"),
                    defaults.intraday_levels_momentum_min_broken_ratio,
                ),
            ),
        ),
        intraday_levels_min_confluence_score=max(
            1,
            cls._to_int(
                raw.get("intraday_levels_min_confluence_score"),
                defaults.intraday_levels_min_confluence_score,
            ),
        ),
        intraday_levels_memory_enabled=cls._to_bool(
            raw.get("intraday_levels_memory_enabled"),
            defaults.intraday_levels_memory_enabled,
        ),
        intraday_levels_memory_min_tests=max(
            1,
            cls._to_int(
                raw.get("intraday_levels_memory_min_tests"),
                defaults.intraday_levels_memory_min_tests,
            ),
        ),
        intraday_levels_memory_max_age_days=max(
            1,
            cls._to_int(
                raw.get("intraday_levels_memory_max_age_days"),
                defaults.intraday_levels_memory_max_age_days,
            ),
        ),
        intraday_levels_memory_decay_after_days=max(
            1,
            cls._to_int(
                raw.get("intraday_levels_memory_decay_after_days"),
                defaults.intraday_levels_memory_decay_after_days,
            ),
        ),
        intraday_levels_memory_decay_weight=min(
            1.0,
            max(
                0.1,
                cls._to_float(
                    raw.get("intraday_levels_memory_decay_weight"),
                    defaults.intraday_levels_memory_decay_weight,
                ),
            ),
        ),
        intraday_levels_memory_max_levels=max(
            1,
            cls._to_int(
                raw.get("intraday_levels_memory_max_levels"),
                defaults.intraday_levels_memory_max_levels,
            ),
        ),
        intraday_levels_opening_range_enabled=cls._to_bool(
            raw.get("intraday_levels_opening_range_enabled"),
            defaults.intraday_levels_opening_range_enabled,
        ),
        intraday_levels_opening_range_minutes=max(
            5,
            cls._to_int(
                raw.get("intraday_levels_opening_range_minutes"),
                defaults.intraday_levels_opening_range_minutes,
            ),
        ),
        intraday_levels_opening_range_break_tolerance_pct=max(
            0.01,
            cls._to_float(
                raw.get("intraday_levels_opening_range_break_tolerance_pct"),
                defaults.intraday_levels_opening_range_break_tolerance_pct,
            ),
        ),
        intraday_levels_poc_migration_enabled=cls._to_bool(
            raw.get("intraday_levels_poc_migration_enabled"),
            defaults.intraday_levels_poc_migration_enabled,
        ),
        intraday_levels_poc_migration_interval_bars=max(
            1,
            cls._to_int(
                raw.get("intraday_levels_poc_migration_interval_bars"),
                defaults.intraday_levels_poc_migration_interval_bars,
            ),
        ),
        intraday_levels_poc_migration_trend_threshold_pct=max(
            0.01,
            cls._to_float(
                raw.get("intraday_levels_poc_migration_trend_threshold_pct"),
                defaults.intraday_levels_poc_migration_trend_threshold_pct,
            ),
        ),
        intraday_levels_poc_migration_range_threshold_pct=max(
            0.01,
            cls._to_float(
                raw.get("intraday_levels_poc_migration_range_threshold_pct"),
                defaults.intraday_levels_poc_migration_range_threshold_pct,
            ),
        ),
        intraday_levels_composite_profile_enabled=cls._to_bool(
            raw.get("intraday_levels_composite_profile_enabled"),
            defaults.intraday_levels_composite_profile_enabled,
        ),
        intraday_levels_composite_profile_days=max(
            1,
            cls._to_int(
                raw.get("intraday_levels_composite_profile_days"),
                defaults.intraday_levels_composite_profile_days,
            ),
        ),
        intraday_levels_composite_profile_current_day_weight=max(
            0.1,
            cls._to_float(
                raw.get("intraday_levels_composite_profile_current_day_weight"),
                defaults.intraday_levels_composite_profile_current_day_weight,
            ),
        ),
        intraday_levels_spike_detection_enabled=cls._to_bool(
            raw.get("intraday_levels_spike_detection_enabled"),
            defaults.intraday_levels_spike_detection_enabled,
        ),
        intraday_levels_spike_min_wick_ratio=min(
            0.95,
            max(
                0.4,
                cls._to_float(
                    raw.get("intraday_levels_spike_min_wick_ratio"),
                    defaults.intraday_levels_spike_min_wick_ratio,
                ),
            ),
        ),
        intraday_levels_prior_day_anchors_enabled=cls._to_bool(
            raw.get("intraday_levels_prior_day_anchors_enabled"),
            defaults.intraday_levels_prior_day_anchors_enabled,
        ),
        intraday_levels_gap_analysis_enabled=cls._to_bool(
            raw.get("intraday_levels_gap_analysis_enabled"),
            defaults.intraday_levels_gap_analysis_enabled,
        ),
        intraday_levels_gap_min_pct=max(
            0.0,
            cls._to_float(
                raw.get("intraday_levels_gap_min_pct"),
                defaults.intraday_levels_gap_min_pct,
            ),
        ),
        intraday_levels_gap_momentum_threshold_pct=max(
            0.1,
            cls._to_float(
                raw.get("intraday_levels_gap_momentum_threshold_pct"),
                defaults.intraday_levels_gap_momentum_threshold_pct,
            ),
        ),
        intraday_levels_rvol_filter_enabled=cls._to_bool(
            raw.get("intraday_levels_rvol_filter_enabled"),
            defaults.intraday_levels_rvol_filter_enabled,
        ),
        intraday_levels_rvol_lookback_bars=max(
            5,
            cls._to_int(
                raw.get("intraday_levels_rvol_lookback_bars"),
                defaults.intraday_levels_rvol_lookback_bars,
            ),
        ),
        intraday_levels_rvol_min_threshold=max(
            0.0,
            cls._to_float(
                raw.get("intraday_levels_rvol_min_threshold"),
                defaults.intraday_levels_rvol_min_threshold,
            ),
        ),
        intraday_levels_rvol_strong_threshold=max(
            0.1,
            cls._to_float(
                raw.get("intraday_levels_rvol_strong_threshold"),
                defaults.intraday_levels_rvol_strong_threshold,
            ),
        ),
        intraday_levels_adaptive_window_enabled=cls._to_bool(
            raw.get("intraday_levels_adaptive_window_enabled"),
            defaults.intraday_levels_adaptive_window_enabled,
        ),
        intraday_levels_adaptive_window_min_bars=max(
            1,
            cls._to_int(
                raw.get("intraday_levels_adaptive_window_min_bars"),
                defaults.intraday_levels_adaptive_window_min_bars,
            ),
        ),
        intraday_levels_adaptive_window_rvol_threshold=max(
            0.0,
            cls._to_float(
                raw.get("intraday_levels_adaptive_window_rvol_threshold"),
                defaults.intraday_levels_adaptive_window_rvol_threshold,
            ),
        ),
        intraday_levels_adaptive_window_atr_ratio_max=max(
            0.1,
            cls._to_float(
                raw.get("intraday_levels_adaptive_window_atr_ratio_max"),
                defaults.intraday_levels_adaptive_window_atr_ratio_max,
            ),
        ),
        intraday_levels_micro_confirmation_enabled=cls._to_bool(
            raw.get("intraday_levels_micro_confirmation_enabled"),
            defaults.intraday_levels_micro_confirmation_enabled,
        ),
        intraday_levels_micro_confirmation_bars=max(
            1,
            cls._to_int(
                raw.get("intraday_levels_micro_confirmation_bars"),
                defaults.intraday_levels_micro_confirmation_bars,
            ),
        ),
        intraday_levels_micro_confirmation_disable_for_sweep=cls._to_bool(
            raw.get("intraday_levels_micro_confirmation_disable_for_sweep"),
            defaults.intraday_levels_micro_confirmation_disable_for_sweep,
        ),
        intraday_levels_micro_confirmation_sweep_bars=max(
            0,
            cls._to_int(
                raw.get("intraday_levels_micro_confirmation_sweep_bars"),
                defaults.intraday_levels_micro_confirmation_sweep_bars,
            ),
        ),
        intraday_levels_micro_confirmation_require_intrabar=cls._to_bool(
            raw.get("intraday_levels_micro_confirmation_require_intrabar"),
            defaults.intraday_levels_micro_confirmation_require_intrabar,
        ),
        intraday_levels_micro_confirmation_intrabar_window_seconds=max(
            1,
            min(
                60,
                cls._to_int(
                    raw.get("intraday_levels_micro_confirmation_intrabar_window_seconds"),
                    defaults.intraday_levels_micro_confirmation_intrabar_window_seconds,
                ),
            ),
        ),
        intraday_levels_micro_confirmation_intrabar_min_coverage_points=max(
            0,
            cls._to_int(
                raw.get("intraday_levels_micro_confirmation_intrabar_min_coverage_points"),
                defaults.intraday_levels_micro_confirmation_intrabar_min_coverage_points,
            ),
        ),
        intraday_levels_micro_confirmation_intrabar_min_move_pct=max(
            0.0,
            cls._to_float(
                raw.get("intraday_levels_micro_confirmation_intrabar_min_move_pct"),
                defaults.intraday_levels_micro_confirmation_intrabar_min_move_pct,
            ),
        ),
        intraday_levels_micro_confirmation_intrabar_min_push_ratio=min(
            1.0,
            max(
                0.0,
                cls._to_float(
                    raw.get("intraday_levels_micro_confirmation_intrabar_min_push_ratio"),
                    defaults.intraday_levels_micro_confirmation_intrabar_min_push_ratio,
                ),
            ),
        ),
        intraday_levels_micro_confirmation_intrabar_max_spread_bps=max(
            0.0,
            cls._to_float(
                raw.get("intraday_levels_micro_confirmation_intrabar_max_spread_bps"),
                defaults.intraday_levels_micro_confirmation_intrabar_max_spread_bps,
            ),
        ),
        intraday_levels_confluence_sizing_enabled=cls._to_bool(
            raw.get("intraday_levels_confluence_sizing_enabled"),
            defaults.intraday_levels_confluence_sizing_enabled,
        ),
        liquidity_sweep_detection_enabled=cls._to_bool(
            raw.get("liquidity_sweep_detection_enabled"),
            defaults.liquidity_sweep_detection_enabled,
        ),
        sweep_min_aggression_z=cls._to_float(
            raw.get("sweep_min_aggression_z"),
            defaults.sweep_min_aggression_z,
        ),
        sweep_min_book_pressure_z=cls._to_float(
            raw.get("sweep_min_book_pressure_z"),
            defaults.sweep_min_book_pressure_z,
        ),
        sweep_max_price_change_pct=max(
            0.0,
            cls._to_float(
                raw.get("sweep_max_price_change_pct"),
                defaults.sweep_max_price_change_pct,
            ),
        ),
        sweep_atr_buffer_multiplier=max(
            0.0,
            cls._to_float(
                raw.get("sweep_atr_buffer_multiplier"),
                defaults.sweep_atr_buffer_multiplier,
            ),
        ),
        context_aware_risk_enabled=cls._to_bool(
            raw.get("context_aware_risk_enabled"),
            defaults.context_aware_risk_enabled,
        ),
        context_risk_sl_buffer_pct=max(
            0.0,
            cls._to_float(
                raw.get("context_risk_sl_buffer_pct"),
                defaults.context_risk_sl_buffer_pct,
            ),
        ),
        context_risk_min_sl_pct=max(
            0.0,
            cls._to_float(
                raw.get("context_risk_min_sl_pct"),
                defaults.context_risk_min_sl_pct,
            ),
        ),
        context_risk_min_room_pct=max(
            0.0,
            cls._to_float(
                raw.get("context_risk_min_room_pct"),
                defaults.context_risk_min_room_pct,
            ),
        ),
        context_risk_min_effective_rr=max(
            0.0,
            cls._to_float(
                raw.get("context_risk_min_effective_rr"),
                defaults.context_risk_min_effective_rr,
            ),
        ),
        context_risk_trailing_tighten_zone=min(
            1.0,
            max(
                0.0,
                cls._to_float(
                    raw.get("context_risk_trailing_tighten_zone"),
                    defaults.context_risk_trailing_tighten_zone,
                ),
            ),
        ),
        context_risk_trailing_tighten_factor=min(
            1.0,
            max(
                0.0,
                cls._to_float(
                    raw.get("context_risk_trailing_tighten_factor"),
                    defaults.context_risk_trailing_tighten_factor,
                ),
            ),
        ),
        context_risk_level_trail_enabled=cls._to_bool(
            raw.get("context_risk_level_trail_enabled"),
            defaults.context_risk_level_trail_enabled,
        ),
        context_risk_max_anchor_search_pct=max(
            0.1,
            cls._to_float(
                raw.get("context_risk_max_anchor_search_pct"),
                defaults.context_risk_max_anchor_search_pct,
            ),
        ),
        context_risk_min_level_tests_for_sl=max(
            0,
            cls._to_int(
                raw.get("context_risk_min_level_tests_for_sl"),
                defaults.context_risk_min_level_tests_for_sl,
            ),
        ),
        l2_gate_mode=str(raw.get("l2_gate_mode", defaults.l2_gate_mode) or defaults.l2_gate_mode).strip().lower(),
        l2_gate_threshold=max(
            0.0,
            min(1.0, cls._to_float(raw.get("l2_gate_threshold"), defaults.l2_gate_threshold)),
        ),
        cold_start_each_day=cls._to_bool(raw.get("cold_start_each_day"), defaults.cold_start_each_day),
        premarket_trading_enabled=cls._to_bool(raw.get("premarket_trading_enabled"), defaults.premarket_trading_enabled),
        strategy_selection_mode=selection_mode,
        max_active_strategies=max(
            1,
            min(
                20,
                cls._to_int(raw.get("max_active_strategies"), defaults.max_active_strategies),
            ),
        ),
        momentum_diversification=copy.deepcopy(momentum),
        regime_filter=cls._parse_regime_filter(raw.get("regime_filter")),
        micro_confirmation_mode=micro_mode,
        micro_confirmation_volume_delta_min_pct=min(
            1.0,
            max(
                0.0,
                cls._to_float(
                    raw.get("micro_confirmation_volume_delta_min_pct"),
                    defaults.micro_confirmation_volume_delta_min_pct,
                ),
            ),
        ),
        weak_l2_fast_break_even_enabled=cls._to_bool(
            raw.get("weak_l2_fast_break_even_enabled"),
            defaults.weak_l2_fast_break_even_enabled,
        ),
        weak_l2_aggression_threshold=max(
            0.0,
            cls._to_float(
                raw.get("weak_l2_aggression_threshold"),
                defaults.weak_l2_aggression_threshold,
            ),
        ),
        weak_l2_break_even_min_hold_bars=max(
            1,
            cls._to_int(
                raw.get("weak_l2_break_even_min_hold_bars"),
                defaults.weak_l2_break_even_min_hold_bars,
            ),
        ),
        ev_relaxation_enabled=cls._to_bool(
            raw.get("ev_relaxation_enabled"),
            defaults.ev_relaxation_enabled,
        ),
        ev_relaxation_threshold=max(
            0.0,
            cls._to_float(
                raw.get("ev_relaxation_threshold"),
                defaults.ev_relaxation_threshold,
            ),
        ),
        ev_relaxation_factor=min(
            1.0,
            max(
                0.01,
                cls._to_float(
                    raw.get("ev_relaxation_factor"),
                    defaults.ev_relaxation_factor,
                ),
            ),
        ),
        intraday_levels_bounce_conflict_buffer_bars=max(
            0,
            cls._to_int(
                raw.get("intraday_levels_bounce_conflict_buffer_bars"),
                defaults.intraday_levels_bounce_conflict_buffer_bars,
            ),
        ),
        orchestrator_strategy_weight=min(
            1.0,
            max(
                0.0,
                cls._to_float(
                    raw.get("orchestrator_strategy_weight"),
                    defaults.orchestrator_strategy_weight,
                ),
            ),
        ),
        orchestrator_strategy_only_threshold=max(
            0.0,
            cls._to_float(
                raw.get("orchestrator_strategy_only_threshold"),
                defaults.orchestrator_strategy_only_threshold,
            ),
        ),
    )



def trading_config_to_session_params(config: "TradingConfig") -> Dict[str, Any]:
    return {
        "regime_detection_minutes": int(config.regime_detection_minutes),
        "regime_refresh_bars": int(config.regime_refresh_bars),
        "account_size_usd": float(config.account_size_usd),
        "risk_per_trade_pct": float(config.risk_per_trade_pct),
        "max_position_notional_pct": float(config.max_position_notional_pct),
        "max_fill_participation_rate": float(config.max_fill_participation_rate),
        "min_fill_ratio": float(config.min_fill_ratio),
        "enable_partial_take_profit": bool(config.enable_partial_take_profit),
        "partial_take_profit_rr": float(config.partial_take_profit_rr),
        "partial_take_profit_fraction": float(config.partial_take_profit_fraction),
        "trailing_activation_pct": float(config.trailing_activation_pct),
        "break_even_buffer_pct": float(config.break_even_buffer_pct),
        "break_even_min_hold_bars": int(config.break_even_min_hold_bars),
        "break_even_activation_min_mfe_pct": float(config.break_even_activation_min_mfe_pct),
        "break_even_activation_min_r": float(config.break_even_activation_min_r),
        "break_even_activation_min_r_trending_5m": float(
            config.break_even_activation_min_r_trending_5m
        ),
        "break_even_activation_min_r_choppy_5m": float(
            config.break_even_activation_min_r_choppy_5m
        ),
        "break_even_activation_use_levels": bool(config.break_even_activation_use_levels),
        "break_even_activation_use_l2": bool(config.break_even_activation_use_l2),
        "break_even_level_buffer_pct": float(config.break_even_level_buffer_pct),
        "break_even_level_max_distance_pct": float(config.break_even_level_max_distance_pct),
        "break_even_level_min_confluence": int(config.break_even_level_min_confluence),
        "break_even_level_min_tests": int(config.break_even_level_min_tests),
        "break_even_l2_signed_aggression_min": float(
            config.break_even_l2_signed_aggression_min
        ),
        "break_even_l2_imbalance_min": float(config.break_even_l2_imbalance_min),
        "break_even_l2_book_pressure_min": float(config.break_even_l2_book_pressure_min),
        "break_even_l2_spread_bps_max": float(config.break_even_l2_spread_bps_max),
        "break_even_costs_pct": float(config.break_even_costs_pct),
        "break_even_min_buffer_pct": float(config.break_even_min_buffer_pct),
        "break_even_atr_buffer_k": float(config.break_even_atr_buffer_k),
        "break_even_5m_atr_buffer_k": float(config.break_even_5m_atr_buffer_k),
        "break_even_tick_size": float(config.break_even_tick_size),
        "break_even_min_tick_buffer": int(config.break_even_min_tick_buffer),
        "break_even_anti_spike_bars": int(config.break_even_anti_spike_bars),
        "break_even_anti_spike_hits_required": int(
            config.break_even_anti_spike_hits_required
        ),
        "break_even_anti_spike_require_close_beyond": bool(
            config.break_even_anti_spike_require_close_beyond
        ),
        "break_even_5m_no_go_proximity_pct": float(config.break_even_5m_no_go_proximity_pct),
        "break_even_5m_mfe_atr_factor": float(config.break_even_5m_mfe_atr_factor),
        "break_even_5m_l2_bias_threshold": float(config.break_even_5m_l2_bias_threshold),
        "break_even_5m_l2_bias_tighten_factor": float(
            config.break_even_5m_l2_bias_tighten_factor
        ),
        "break_even_movement_formula_enabled": bool(config.break_even_movement_formula_enabled),
        "break_even_movement_formula": str(config.break_even_movement_formula or ""),
        "break_even_proof_formula_enabled": bool(config.break_even_proof_formula_enabled),
        "break_even_proof_formula": str(config.break_even_proof_formula or ""),
        "break_even_activation_formula_enabled": bool(
            config.break_even_activation_formula_enabled
        ),
        "break_even_activation_formula": str(config.break_even_activation_formula or ""),
        "break_even_trailing_handoff_formula_enabled": bool(
            config.break_even_trailing_handoff_formula_enabled
        ),
        "break_even_trailing_handoff_formula": str(
            config.break_even_trailing_handoff_formula or ""
        ),
        "trailing_enabled_in_choppy": bool(config.trailing_enabled_in_choppy),
        "time_exit_bars": int(config.time_exit_bars),
        "time_exit_formula_enabled": bool(config.time_exit_formula_enabled),
        "time_exit_formula": str(config.time_exit_formula or ""),
        "adverse_flow_exit_enabled": bool(config.adverse_flow_exit_enabled),
        "adverse_flow_threshold": float(config.adverse_flow_threshold),
        "adverse_flow_min_hold_bars": int(config.adverse_flow_min_hold_bars),
        "adverse_flow_consistency_threshold": float(config.adverse_flow_consistency_threshold),
        "adverse_book_pressure_threshold": float(config.adverse_book_pressure_threshold),
        "adverse_flow_exit_formula_enabled": bool(config.adverse_flow_exit_formula_enabled),
        "adverse_flow_exit_formula": str(config.adverse_flow_exit_formula or ""),
        "stop_loss_mode": str(config.stop_loss_mode),
        "fixed_stop_loss_pct": float(config.fixed_stop_loss_pct),
        "l2_confirm_enabled": bool(config.l2_confirm_enabled),
        "l2_min_delta": float(config.l2_min_delta),
        "l2_min_imbalance": float(config.l2_min_imbalance),
        "l2_min_iceberg_bias": float(config.l2_min_iceberg_bias),
        "l2_lookback_bars": int(config.l2_lookback_bars),
        "l2_min_participation_ratio": float(config.l2_min_participation_ratio),
        "l2_min_directional_consistency": float(config.l2_min_directional_consistency),
        "l2_min_signed_aggression": float(config.l2_min_signed_aggression),
        "tcbbo_gate_enabled": bool(config.tcbbo_gate_enabled),
        "tcbbo_min_net_premium": float(config.tcbbo_min_net_premium),
        "tcbbo_sweep_boost": float(config.tcbbo_sweep_boost),
        "tcbbo_lookback_bars": int(config.tcbbo_lookback_bars),
        "intraday_levels_enabled": bool(config.intraday_levels_enabled),
        "intraday_levels_swing_left_bars": int(config.intraday_levels_swing_left_bars),
        "intraday_levels_swing_right_bars": int(config.intraday_levels_swing_right_bars),
        "intraday_levels_test_tolerance_pct": float(config.intraday_levels_test_tolerance_pct),
        "intraday_levels_break_tolerance_pct": float(config.intraday_levels_break_tolerance_pct),
        "intraday_levels_breakout_volume_lookback": int(
            config.intraday_levels_breakout_volume_lookback
        ),
        "intraday_levels_breakout_volume_multiplier": float(
            config.intraday_levels_breakout_volume_multiplier
        ),
        "intraday_levels_volume_profile_bin_size_pct": float(
            config.intraday_levels_volume_profile_bin_size_pct
        ),
        "intraday_levels_value_area_pct": float(config.intraday_levels_value_area_pct),
        "intraday_levels_entry_quality_enabled": bool(
            config.intraday_levels_entry_quality_enabled
        ),
        "intraday_levels_min_levels_for_context": int(
            config.intraday_levels_min_levels_for_context
        ),
        "intraday_levels_entry_tolerance_pct": float(
            config.intraday_levels_entry_tolerance_pct
        ),
        "intraday_levels_break_cooldown_bars": int(
            config.intraday_levels_break_cooldown_bars
        ),
        "intraday_levels_rotation_max_tests": int(
            config.intraday_levels_rotation_max_tests
        ),
        "intraday_levels_rotation_volume_max_ratio": float(
            config.intraday_levels_rotation_volume_max_ratio
        ),
        "intraday_levels_recent_bounce_lookback_bars": int(
            config.intraday_levels_recent_bounce_lookback_bars
        ),
        "intraday_levels_require_recent_bounce_for_mean_reversion": bool(
            config.intraday_levels_require_recent_bounce_for_mean_reversion
        ),
        "intraday_levels_momentum_break_max_age_bars": int(
            config.intraday_levels_momentum_break_max_age_bars
        ),
        "intraday_levels_momentum_min_room_pct": float(
            config.intraday_levels_momentum_min_room_pct
        ),
        "intraday_levels_momentum_min_broken_ratio": float(
            config.intraday_levels_momentum_min_broken_ratio
        ),
        "intraday_levels_min_confluence_score": int(
            config.intraday_levels_min_confluence_score
        ),
        "intraday_levels_memory_enabled": bool(config.intraday_levels_memory_enabled),
        "intraday_levels_memory_min_tests": int(config.intraday_levels_memory_min_tests),
        "intraday_levels_memory_max_age_days": int(config.intraday_levels_memory_max_age_days),
        "intraday_levels_memory_decay_after_days": int(
            config.intraday_levels_memory_decay_after_days
        ),
        "intraday_levels_memory_decay_weight": float(
            config.intraday_levels_memory_decay_weight
        ),
        "intraday_levels_memory_max_levels": int(
            config.intraday_levels_memory_max_levels
        ),
        "intraday_levels_opening_range_enabled": bool(
            config.intraday_levels_opening_range_enabled
        ),
        "intraday_levels_opening_range_minutes": int(
            config.intraday_levels_opening_range_minutes
        ),
        "intraday_levels_opening_range_break_tolerance_pct": float(
            config.intraday_levels_opening_range_break_tolerance_pct
        ),
        "intraday_levels_poc_migration_enabled": bool(
            config.intraday_levels_poc_migration_enabled
        ),
        "intraday_levels_poc_migration_interval_bars": int(
            config.intraday_levels_poc_migration_interval_bars
        ),
        "intraday_levels_poc_migration_trend_threshold_pct": float(
            config.intraday_levels_poc_migration_trend_threshold_pct
        ),
        "intraday_levels_poc_migration_range_threshold_pct": float(
            config.intraday_levels_poc_migration_range_threshold_pct
        ),
        "intraday_levels_composite_profile_enabled": bool(
            config.intraday_levels_composite_profile_enabled
        ),
        "intraday_levels_composite_profile_days": int(
            config.intraday_levels_composite_profile_days
        ),
        "intraday_levels_composite_profile_current_day_weight": float(
            config.intraday_levels_composite_profile_current_day_weight
        ),
        "intraday_levels_spike_detection_enabled": bool(
            config.intraday_levels_spike_detection_enabled
        ),
        "intraday_levels_spike_min_wick_ratio": float(
            config.intraday_levels_spike_min_wick_ratio
        ),
        "intraday_levels_prior_day_anchors_enabled": bool(
            config.intraday_levels_prior_day_anchors_enabled
        ),
        "intraday_levels_gap_analysis_enabled": bool(
            config.intraday_levels_gap_analysis_enabled
        ),
        "intraday_levels_gap_min_pct": float(config.intraday_levels_gap_min_pct),
        "intraday_levels_gap_momentum_threshold_pct": float(
            config.intraday_levels_gap_momentum_threshold_pct
        ),
        "intraday_levels_rvol_filter_enabled": bool(
            config.intraday_levels_rvol_filter_enabled
        ),
        "intraday_levels_rvol_lookback_bars": int(
            config.intraday_levels_rvol_lookback_bars
        ),
        "intraday_levels_rvol_min_threshold": float(
            config.intraday_levels_rvol_min_threshold
        ),
        "intraday_levels_rvol_strong_threshold": float(
            config.intraday_levels_rvol_strong_threshold
        ),
        "intraday_levels_adaptive_window_enabled": bool(
            config.intraday_levels_adaptive_window_enabled
        ),
        "intraday_levels_adaptive_window_min_bars": int(
            config.intraday_levels_adaptive_window_min_bars
        ),
        "intraday_levels_adaptive_window_rvol_threshold": float(
            config.intraday_levels_adaptive_window_rvol_threshold
        ),
        "intraday_levels_adaptive_window_atr_ratio_max": float(
            config.intraday_levels_adaptive_window_atr_ratio_max
        ),
        "intraday_levels_micro_confirmation_enabled": bool(
            config.intraday_levels_micro_confirmation_enabled
        ),
        "intraday_levels_micro_confirmation_bars": int(
            config.intraday_levels_micro_confirmation_bars
        ),
        "intraday_levels_micro_confirmation_disable_for_sweep": bool(
            config.intraday_levels_micro_confirmation_disable_for_sweep
        ),
        "intraday_levels_micro_confirmation_sweep_bars": int(
            config.intraday_levels_micro_confirmation_sweep_bars
        ),
        "intraday_levels_micro_confirmation_require_intrabar": bool(
            config.intraday_levels_micro_confirmation_require_intrabar
        ),
        "intraday_levels_micro_confirmation_intrabar_window_seconds": int(
            config.intraday_levels_micro_confirmation_intrabar_window_seconds
        ),
        "intraday_levels_micro_confirmation_intrabar_min_coverage_points": int(
            config.intraday_levels_micro_confirmation_intrabar_min_coverage_points
        ),
        "intraday_levels_micro_confirmation_intrabar_min_move_pct": float(
            config.intraday_levels_micro_confirmation_intrabar_min_move_pct
        ),
        "intraday_levels_micro_confirmation_intrabar_min_push_ratio": float(
            config.intraday_levels_micro_confirmation_intrabar_min_push_ratio
        ),
        "intraday_levels_micro_confirmation_intrabar_max_spread_bps": float(
            config.intraday_levels_micro_confirmation_intrabar_max_spread_bps
        ),
        "intraday_levels_confluence_sizing_enabled": bool(
            config.intraday_levels_confluence_sizing_enabled
        ),
        "liquidity_sweep_detection_enabled": bool(
            config.liquidity_sweep_detection_enabled
        ),
        "sweep_min_aggression_z": float(config.sweep_min_aggression_z),
        "sweep_min_book_pressure_z": float(config.sweep_min_book_pressure_z),
        "sweep_max_price_change_pct": float(config.sweep_max_price_change_pct),
        "sweep_atr_buffer_multiplier": float(config.sweep_atr_buffer_multiplier),
        "context_aware_risk_enabled": bool(config.context_aware_risk_enabled),
        "context_risk_sl_buffer_pct": float(config.context_risk_sl_buffer_pct),
        "context_risk_min_sl_pct": float(config.context_risk_min_sl_pct),
        "context_risk_min_room_pct": float(config.context_risk_min_room_pct),
        "context_risk_min_effective_rr": float(config.context_risk_min_effective_rr),
        "context_risk_trailing_tighten_zone": float(
            config.context_risk_trailing_tighten_zone
        ),
        "context_risk_trailing_tighten_factor": float(
            config.context_risk_trailing_tighten_factor
        ),
        "context_risk_level_trail_enabled": bool(
            config.context_risk_level_trail_enabled
        ),
        "context_risk_max_anchor_search_pct": float(
            config.context_risk_max_anchor_search_pct
        ),
        "context_risk_min_level_tests_for_sl": int(
            config.context_risk_min_level_tests_for_sl
        ),
        "l2_gate_mode": str(config.l2_gate_mode),
        "l2_gate_threshold": float(config.l2_gate_threshold),
        "cold_start_each_day": bool(config.cold_start_each_day),
        "premarket_trading_enabled": bool(config.premarket_trading_enabled),
        "strategy_selection_mode": str(config.strategy_selection_mode),
        "max_active_strategies": int(config.max_active_strategies),
        "momentum_diversification": copy.deepcopy(config.momentum_diversification),
        "regime_filter": list(config.regime_filter) if config.regime_filter is not None else None,
        "micro_confirmation_mode": str(config.micro_confirmation_mode),
        "micro_confirmation_volume_delta_min_pct": float(
            config.micro_confirmation_volume_delta_min_pct
        ),
        "weak_l2_fast_break_even_enabled": bool(config.weak_l2_fast_break_even_enabled),
        "weak_l2_aggression_threshold": float(config.weak_l2_aggression_threshold),
        "weak_l2_break_even_min_hold_bars": int(config.weak_l2_break_even_min_hold_bars),
        "ev_relaxation_enabled": bool(config.ev_relaxation_enabled),
        "ev_relaxation_threshold": float(config.ev_relaxation_threshold),
        "ev_relaxation_factor": float(config.ev_relaxation_factor),
        "intraday_levels_bounce_conflict_buffer_bars": int(
            config.intraday_levels_bounce_conflict_buffer_bars
        ),
        "orchestrator_strategy_weight": float(config.orchestrator_strategy_weight),
        "orchestrator_strategy_only_threshold": float(
            config.orchestrator_strategy_only_threshold
        ),
    }




__all__ = [
    "build_trading_config_from_dict",
    "trading_config_to_session_params",
]
