"""Canonical runtime trading configuration schema."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
from typing import Any, Dict, Optional, Tuple

from .trading_config_helpers.serialization import (
    build_trading_config_from_dict,
    trading_config_to_session_params,
)


@dataclass(frozen=True)
class TradingConfig:
    """Single source of truth for runtime trading configuration."""

    regime_detection_minutes: int = 15
    regime_refresh_bars: int = 30
    account_size_usd: float = 10_000.0
    risk_per_trade_pct: float = 1.0
    max_position_notional_pct: float = 100.0
    max_fill_participation_rate: float = 0.20
    min_fill_ratio: float = 0.35
    enable_partial_take_profit: bool = True
    partial_take_profit_rr: float = 1.0
    partial_take_profit_fraction: float = 0.5
    partial_flow_deterioration_min_r: float = 0.5
    partial_flow_deterioration_skip_be: bool = True
    partial_protect_min_mfe_r: float = 0.0
    trailing_activation_pct: float = 0.15
    break_even_buffer_pct: float = 0.03
    break_even_min_hold_bars: int = 3
    break_even_activation_min_mfe_pct: float = 0.25
    break_even_activation_min_r: float = 0.60
    break_even_activation_min_r_trending_5m: float = 0.90
    break_even_activation_min_r_choppy_5m: float = 0.60
    break_even_activation_use_levels: bool = True
    break_even_activation_use_l2: bool = True
    break_even_level_buffer_pct: float = 0.02
    break_even_level_max_distance_pct: float = 0.60
    break_even_level_min_confluence: int = 2
    break_even_level_min_tests: int = 1
    break_even_l2_signed_aggression_min: float = 0.12
    break_even_l2_imbalance_min: float = 0.15
    break_even_l2_book_pressure_min: float = 0.10
    break_even_l2_spread_bps_max: float = 12.0
    break_even_costs_pct: float = 0.03
    break_even_min_buffer_pct: float = 0.05
    break_even_atr_buffer_k: float = 0.10
    break_even_5m_atr_buffer_k: float = 0.10
    break_even_tick_size: float = 0.01
    break_even_min_tick_buffer: int = 1
    break_even_anti_spike_bars: int = 1
    break_even_anti_spike_hits_required: int = 2
    break_even_anti_spike_require_close_beyond: bool = True
    break_even_5m_no_go_proximity_pct: float = 0.10
    break_even_5m_mfe_atr_factor: float = 0.15
    break_even_5m_l2_bias_threshold: float = 0.10
    break_even_5m_l2_bias_tighten_factor: float = 0.85
    break_even_movement_formula_enabled: bool = False
    break_even_movement_formula: str = ""
    break_even_proof_formula_enabled: bool = False
    break_even_proof_formula: str = ""
    break_even_activation_formula_enabled: bool = False
    break_even_activation_formula: str = ""
    break_even_trailing_handoff_formula_enabled: bool = False
    break_even_trailing_handoff_formula: str = ""
    trailing_enabled_in_choppy: bool = False
    time_exit_bars: int = 40
    time_exit_formula_enabled: bool = False
    time_exit_formula: str = ""
    adverse_flow_exit_enabled: bool = True
    adverse_flow_threshold: float = 0.20
    adverse_flow_min_hold_bars: int = 6
    adverse_flow_consistency_threshold: float = 0.45
    adverse_book_pressure_threshold: float = 0.15
    adverse_flow_exit_formula_enabled: bool = False
    adverse_flow_exit_formula: str = ""
    stop_loss_mode: str = "strategy"
    fixed_stop_loss_pct: float = 0.0
    l2_confirm_enabled: bool = False
    l2_min_delta: float = 0.0
    l2_min_imbalance: float = 0.0
    l2_min_iceberg_bias: float = 0.0
    l2_lookback_bars: int = 3
    l2_min_participation_ratio: float = 0.0
    l2_min_directional_consistency: float = 0.0
    l2_min_signed_aggression: float = 0.0
    tcbbo_gate_enabled: bool = False
    tcbbo_min_net_premium: float = 0.0
    tcbbo_sweep_boost: float = 5.0
    tcbbo_lookback_bars: int = 5
    # TCBBO adaptive entry threshold
    tcbbo_adaptive_threshold: bool = True
    tcbbo_adaptive_lookback_bars: int = 30
    tcbbo_adaptive_min_pct: float = 0.15
    # TCBBO anti-flow-fade filter
    tcbbo_flow_fade_filter: bool = True
    tcbbo_flow_fade_min_ratio: float = 0.3
    # TCBBO exit management
    tcbbo_exit_tighten_enabled: bool = False
    tcbbo_exit_lookback_bars: int = 5
    tcbbo_exit_contra_threshold: float = 50000.0
    tcbbo_exit_tighten_pct: float = 0.15
    # Options flow alpha strategy
    options_flow_alpha_enabled: bool = False
    intraday_levels_enabled: bool = True
    intraday_levels_swing_left_bars: int = 2
    intraday_levels_swing_right_bars: int = 2
    intraday_levels_test_tolerance_pct: float = 0.08
    intraday_levels_break_tolerance_pct: float = 0.05
    intraday_levels_breakout_volume_lookback: int = 20
    intraday_levels_breakout_volume_multiplier: float = 1.2
    intraday_levels_volume_profile_bin_size_pct: float = 0.05
    intraday_levels_value_area_pct: float = 0.70
    intraday_levels_entry_quality_enabled: bool = True
    intraday_levels_min_levels_for_context: int = 2
    intraday_levels_entry_tolerance_pct: float = 0.10
    intraday_levels_break_cooldown_bars: int = 6
    intraday_levels_rotation_max_tests: int = 2
    intraday_levels_rotation_volume_max_ratio: float = 0.95
    intraday_levels_recent_bounce_lookback_bars: int = 6
    intraday_levels_require_recent_bounce_for_mean_reversion: bool = True
    intraday_levels_momentum_break_max_age_bars: int = 3
    intraday_levels_momentum_min_room_pct: float = 0.30
    intraday_levels_momentum_min_broken_ratio: float = 0.30
    intraday_levels_min_confluence_score: int = 2
    intraday_levels_memory_enabled: bool = True
    intraday_levels_memory_min_tests: int = 2
    intraday_levels_memory_max_age_days: int = 5
    intraday_levels_memory_decay_after_days: int = 2
    intraday_levels_memory_decay_weight: float = 0.50
    intraday_levels_memory_max_levels: int = 12
    intraday_levels_opening_range_enabled: bool = True
    intraday_levels_opening_range_minutes: int = 30
    intraday_levels_opening_range_break_tolerance_pct: float = 0.05
    intraday_levels_poc_migration_enabled: bool = True
    intraday_levels_poc_migration_interval_bars: int = 30
    intraday_levels_poc_migration_trend_threshold_pct: float = 0.20
    intraday_levels_poc_migration_range_threshold_pct: float = 0.10
    intraday_levels_composite_profile_enabled: bool = True
    intraday_levels_composite_profile_days: int = 3
    intraday_levels_composite_profile_current_day_weight: float = 1.0
    intraday_levels_spike_detection_enabled: bool = True
    intraday_levels_spike_min_wick_ratio: float = 0.60
    intraday_levels_prior_day_anchors_enabled: bool = True
    intraday_levels_gap_analysis_enabled: bool = True
    intraday_levels_gap_min_pct: float = 0.30
    intraday_levels_gap_momentum_threshold_pct: float = 2.0
    intraday_levels_rvol_filter_enabled: bool = True
    intraday_levels_rvol_lookback_bars: int = 20
    intraday_levels_rvol_min_threshold: float = 0.80
    intraday_levels_pullback_rvol_min_threshold: float = 0.30
    intraday_levels_rvol_strong_threshold: float = 1.50
    intraday_levels_adaptive_window_enabled: bool = True
    intraday_levels_adaptive_window_min_bars: int = 6
    intraday_levels_adaptive_window_rvol_threshold: float = 1.0
    intraday_levels_adaptive_window_atr_ratio_max: float = 1.5
    intraday_levels_micro_confirmation_enabled: bool = False
    intraday_levels_micro_confirmation_bars: int = 2
    intraday_levels_micro_confirmation_disable_for_sweep: bool = False
    intraday_levels_micro_confirmation_sweep_bars: int = 0
    intraday_levels_micro_confirmation_require_intrabar: bool = False
    intraday_levels_micro_confirmation_intrabar_window_seconds: int = 5
    intraday_levels_micro_confirmation_intrabar_min_coverage_points: int = 3
    intraday_levels_micro_confirmation_intrabar_min_move_pct: float = 0.02
    intraday_levels_micro_confirmation_intrabar_min_push_ratio: float = 0.10
    intraday_levels_micro_confirmation_intrabar_max_spread_bps: float = 12.0
    intrabar_eval_step_seconds: int = 5
    intraday_levels_confluence_sizing_enabled: bool = False
    liquidity_sweep_detection_enabled: bool = False
    sweep_min_aggression_z: float = -2.0
    sweep_min_book_pressure_z: float = 1.5
    sweep_max_price_change_pct: float = 0.05
    sweep_atr_buffer_multiplier: float = 0.5
    context_aware_risk_enabled: bool = False
    context_risk_sl_buffer_pct: float = 0.03
    context_risk_min_sl_pct: float = 0.30
    context_risk_min_room_pct: float = 0.15
    context_risk_min_effective_rr: float = 0.80
    context_risk_trailing_tighten_zone: float = 0.20
    context_risk_trailing_tighten_factor: float = 0.50
    context_risk_level_trail_enabled: bool = True
    context_risk_max_anchor_search_pct: float = 1.5
    context_risk_min_level_tests_for_sl: int = 1
    # Pullback-specific risk/entry/exit hardening.
    pullback_context_min_sl_pct: float = 0.50
    pullback_time_exit_bars: int = 7
    pullback_morning_window_enabled: bool = True
    pullback_entry_start_time: str = "10:00"
    pullback_entry_end_time: str = "11:30"
    pullback_require_poc_on_trade_side: bool = True
    pullback_block_choppy_macro: bool = True
    pullback_blocked_micro_regimes: Tuple[str, ...] = ("CHOPPY", "TRANSITION")
    pullback_min_price_trend_efficiency: float = 0.15
    pullback_break_even_proof_required: bool = False
    pullback_break_even_activation_min_r: float = 0.40
    pullback_break_even_l2_book_pressure_min: float = 0.03
    l2_gate_mode: str = "weighted"  # "weighted" | "all_pass"
    l2_gate_threshold: float = 0.30
    cold_start_each_day: bool = False
    premarket_trading_enabled: bool = False
    strategy_selection_mode: str = "adaptive_top_n"
    max_active_strategies: int = 3
    momentum_diversification: Dict[str, Any] = field(default_factory=dict)
    regime_filter: Optional[Tuple[str, ...]] = None
    # -- Entry filter relaxation --
    micro_confirmation_mode: str = "consecutive_close"  # "consecutive_close" | "volume_delta" | "disabled"
    micro_confirmation_volume_delta_min_pct: float = 0.60
    weak_l2_fast_break_even_enabled: bool = False
    weak_l2_aggression_threshold: float = 0.05
    weak_l2_break_even_min_hold_bars: int = 2
    ev_relaxation_enabled: bool = False
    ev_relaxation_threshold: float = 10.0
    ev_relaxation_factor: float = 0.50
    intraday_levels_bounce_conflict_buffer_bars: int = 0
    # -- Golden setup detection --
    golden_setups_enabled: bool = False
    golden_setups_max_entries_per_day: int = 3
    golden_setups_cooldown_bars: int = 5
    golden_setups_min_l2_score: float = 60.0
    golden_setup_absorption_reversal_enabled: bool = True
    golden_setup_ar_threshold_reduction: float = 12.0
    golden_setup_ar_confidence_boost: float = 15.0
    golden_setup_ar_tcbbo_boost: float = 10.0
    golden_setup_gamma_squeeze_enabled: bool = True
    golden_setup_gsb_threshold_reduction: float = 10.0
    golden_setup_gsb_confidence_boost: float = 12.0
    golden_setup_gsb_tcbbo_boost: float = 8.0
    golden_setup_liquidity_trap_enabled: bool = True
    golden_setup_lt_threshold_reduction: float = 10.0
    golden_setup_lt_confidence_boost: float = 12.0
    golden_setup_lt_tcbbo_boost: float = 8.0
    golden_setup_iceberg_defense_enabled: bool = True
    golden_setup_id_threshold_reduction: float = 8.0
    golden_setup_id_confidence_boost: float = 10.0
    golden_setup_id_tcbbo_boost: float = 8.0
    golden_setup_fuel_injection_enabled: bool = True
    golden_setup_fi_threshold_reduction: float = 8.0
    golden_setup_fi_confidence_boost: float = 10.0
    golden_setup_fi_tcbbo_boost: float = 8.0
    # -- Orchestrator weights --
    orchestrator_strategy_weight: float = 0.6
    orchestrator_strategy_only_threshold: float = 0.0

    VALID_STOP_LOSS_MODES = {"strategy", "fixed", "capped"}
    VALID_STRATEGY_SELECTION_MODES = {"adaptive_top_n", "all_enabled"}
    VALID_MICRO_CONFIRMATION_MODES = {"consecutive_close", "volume_delta", "disabled"}

    @staticmethod
    def _to_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _to_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _to_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off"}:
                return False
        return bool(default)

    @staticmethod
    def _to_text(value: Any, default: str = "") -> str:
        return str(value if value is not None else default).strip()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TradingConfig":
        return build_trading_config_from_dict(cls, d)

    @staticmethod
    def _parse_regime_filter(value: Any) -> Optional[Tuple[str, ...]]:
        """Parse regime_filter from raw input. None means 'use ticker default'."""
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            parsed = tuple(
                str(r).strip().upper() for r in value if str(r).strip()
            )
            return parsed  # empty tuple = allow all
        return None

    @staticmethod
    def _parse_upper_tuple(value: Any, fallback: Tuple[str, ...]) -> Tuple[str, ...]:
        if isinstance(value, (list, tuple)):
            parsed = tuple(
                str(token).strip().upper()
                for token in value
                if str(token).strip()
            )
            if parsed:
                return parsed
        return tuple(str(token).strip().upper() for token in fallback if str(token).strip())

    def merge(self, overrides: Dict[str, Any]) -> "TradingConfig":
        if not isinstance(overrides, dict) or not overrides:
            return self
        merged = self.to_session_params()
        for key, value in overrides.items():
            if value is None:
                continue
            merged[key] = value
        return TradingConfig.from_dict(merged)

    def to_session_params(self) -> Dict[str, Any]:
        return trading_config_to_session_params(self)


__all__ = ["TradingConfig"]
