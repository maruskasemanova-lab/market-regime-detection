"""trading_config_to_session_params implementation."""

from __future__ import annotations

import copy
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ...trading_config import TradingConfig


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
        "partial_flow_deterioration_min_r": float(config.partial_flow_deterioration_min_r),
        "partial_flow_deterioration_skip_be": bool(config.partial_flow_deterioration_skip_be),
        "partial_protect_min_mfe_r": float(config.partial_protect_min_mfe_r),
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
        "tcbbo_adaptive_threshold": bool(config.tcbbo_adaptive_threshold),
        "tcbbo_adaptive_lookback_bars": int(config.tcbbo_adaptive_lookback_bars),
        "tcbbo_adaptive_min_pct": float(config.tcbbo_adaptive_min_pct),
        "tcbbo_flow_fade_filter": bool(config.tcbbo_flow_fade_filter),
        "tcbbo_flow_fade_min_ratio": float(config.tcbbo_flow_fade_min_ratio),
        "tcbbo_exit_tighten_enabled": bool(config.tcbbo_exit_tighten_enabled),
        "tcbbo_exit_lookback_bars": int(config.tcbbo_exit_lookback_bars),
        "tcbbo_exit_contra_threshold": float(config.tcbbo_exit_contra_threshold),
        "tcbbo_exit_tighten_pct": float(config.tcbbo_exit_tighten_pct),
        "options_flow_alpha_enabled": bool(config.options_flow_alpha_enabled),
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
        "intraday_levels_pullback_rvol_min_threshold": float(
            config.intraday_levels_pullback_rvol_min_threshold
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
        "intrabar_eval_step_seconds": int(config.intrabar_eval_step_seconds),
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
        "golden_setups_enabled": bool(config.golden_setups_enabled),
        "golden_setups_max_entries_per_day": int(config.golden_setups_max_entries_per_day),
        "golden_setups_cooldown_bars": int(config.golden_setups_cooldown_bars),
        "golden_setups_min_l2_score": float(config.golden_setups_min_l2_score),
        "golden_setup_absorption_reversal_enabled": bool(
            config.golden_setup_absorption_reversal_enabled
        ),
        "golden_setup_ar_threshold_reduction": float(
            config.golden_setup_ar_threshold_reduction
        ),
        "golden_setup_ar_confidence_boost": float(
            config.golden_setup_ar_confidence_boost
        ),
        "golden_setup_ar_tcbbo_boost": float(config.golden_setup_ar_tcbbo_boost),
        "golden_setup_gamma_squeeze_enabled": bool(
            config.golden_setup_gamma_squeeze_enabled
        ),
        "golden_setup_gsb_threshold_reduction": float(
            config.golden_setup_gsb_threshold_reduction
        ),
        "golden_setup_gsb_confidence_boost": float(
            config.golden_setup_gsb_confidence_boost
        ),
        "golden_setup_gsb_tcbbo_boost": float(config.golden_setup_gsb_tcbbo_boost),
        "golden_setup_liquidity_trap_enabled": bool(
            config.golden_setup_liquidity_trap_enabled
        ),
        "golden_setup_lt_threshold_reduction": float(
            config.golden_setup_lt_threshold_reduction
        ),
        "golden_setup_lt_confidence_boost": float(
            config.golden_setup_lt_confidence_boost
        ),
        "golden_setup_lt_tcbbo_boost": float(config.golden_setup_lt_tcbbo_boost),
        "golden_setup_iceberg_defense_enabled": bool(
            config.golden_setup_iceberg_defense_enabled
        ),
        "golden_setup_id_threshold_reduction": float(
            config.golden_setup_id_threshold_reduction
        ),
        "golden_setup_id_confidence_boost": float(
            config.golden_setup_id_confidence_boost
        ),
        "golden_setup_id_tcbbo_boost": float(config.golden_setup_id_tcbbo_boost),
        "golden_setup_fuel_injection_enabled": bool(
            config.golden_setup_fuel_injection_enabled
        ),
        "golden_setup_fi_threshold_reduction": float(
            config.golden_setup_fi_threshold_reduction
        ),
        "golden_setup_fi_confidence_boost": float(
            config.golden_setup_fi_confidence_boost
        ),
        "golden_setup_fi_tcbbo_boost": float(config.golden_setup_fi_tcbbo_boost),
        "orchestrator_strategy_weight": float(config.orchestrator_strategy_weight),
        "orchestrator_strategy_only_threshold": float(
            config.orchestrator_strategy_only_threshold
        ),
    }
