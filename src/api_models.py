"""
Pydantic API models shared by strategy API endpoints.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, model_validator


class StrategyToggle(BaseModel):
    strategy_name: str
    enabled: bool


class StrategyUpdate(BaseModel):
    strategy_name: str
    params: Dict[str, Any]


class TrailingStopConfig(BaseModel):
    stop_type: str = "PERCENT"
    initial_stop_pct: float = 2.0
    trailing_pct: float = 0.8
    atr_multiplier: float = 2.0


class BarInput(BaseModel):
    """Input model for processing a single bar."""

    run_id: str
    ticker: str
    timestamp: str  # ISO format datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    l2_delta: Optional[float] = None
    l2_buy_volume: Optional[float] = None
    l2_sell_volume: Optional[float] = None
    l2_volume: Optional[float] = None
    l2_imbalance: Optional[float] = None
    l2_bid_depth_total: Optional[float] = None
    l2_ask_depth_total: Optional[float] = None
    l2_book_pressure: Optional[float] = None
    l2_book_pressure_change: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def _compat_book_pressure_delta(cls, data):
        """Accept legacy field name l2_book_pressure_delta as alias."""
        if isinstance(data, dict):
            if data.get("l2_book_pressure_change") is None and "l2_book_pressure_delta" in data:
                data["l2_book_pressure_change"] = data.pop("l2_book_pressure_delta")
        return data
    l2_iceberg_buy_count: Optional[float] = None
    l2_iceberg_sell_count: Optional[float] = None
    l2_iceberg_bias: Optional[float] = None
    l2_quality_flags: Optional[List[str]] = None
    l2_quality: Optional[Dict[str, Any]] = None
    # Optional 1-second top-of-book snapshots for intrabar execution replay.
    # Each frame should include:
    #   s: second-in-minute [0..59], bid: top bid px, ask: top ask px
    intrabar_quotes_1s: Optional[List[Dict[str, float]]] = None
    # Cross-asset reference bar (optional, e.g. QQQ)
    ref_ticker: Optional[str] = None
    ref_open: Optional[float] = None
    ref_high: Optional[float] = None
    ref_low: Optional[float] = None
    ref_close: Optional[float] = None
    ref_volume: Optional[float] = None
    # Optional analyzer/backtest warmup mode: update session/regime state but do not
    # generate or execute trades for this bar.
    warmup_only: bool = False


class SessionQuery(BaseModel):
    """Query model for session operations."""

    run_id: str
    ticker: str
    date: str  # YYYY-MM-DD format


class TradingConfig(BaseModel):
    """Global trading configuration."""

    regime_detection_minutes: int = 60
    regime_refresh_bars: int = 12
    max_daily_loss: float = 300.0
    max_trades_per_day: int = 3
    trade_cooldown_bars: int = 2
    pending_signal_ttl_bars: int = 3
    consecutive_loss_limit: int = 3
    consecutive_loss_cooldown_bars: int = 8
    portfolio_drawdown_halt_pct: float = 5.0
    headwind_activation_score: float = 0.5
    account_size_usd: float = 10000.0
    risk_per_trade_pct: float = 1.0
    max_position_notional_pct: float = 100.0
    max_fill_participation_rate: float = 0.20
    min_fill_ratio: float = 0.35
    enable_partial_take_profit: bool = True
    partial_take_profit_rr: float = 1.0
    partial_take_profit_fraction: float = 0.5
    partial_flow_deterioration_min_r: float = 0.5
    partial_flow_deterioration_skip_be: bool = True
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
    adverse_flow_threshold: float = 0.12
    adverse_flow_min_hold_bars: int = 3
    adverse_flow_consistency_threshold: float = 0.45
    adverse_book_pressure_threshold: float = 0.15
    adverse_flow_exit_formula_enabled: bool = False
    adverse_flow_exit_formula: str = ""
    stop_loss_mode: str = "strategy"
    fixed_stop_loss_pct: float = 0.0
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
    regime_filter: Optional[List[str]] = None
