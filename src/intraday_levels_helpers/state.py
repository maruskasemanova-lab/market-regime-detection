"""State defaults and normalization for intraday levels tracker."""

from __future__ import annotations

from typing import Any, Dict, Optional

def new_state() -> Dict[str, Any]:
    return {
        "config": {
            "enabled": True,
            "swing_left_bars": 2,
            "swing_right_bars": 2,
            "merge_tolerance_pct": 0.12,
            "test_tolerance_pct": 0.08,
            "break_tolerance_pct": 0.05,
            "min_retest_bars": 2,
            "breakout_volume_lookback": 20,
            "breakout_volume_multiplier": 1.2,
            "volume_profile_bin_size_pct": 0.05,
            "value_area_pct": 0.70,
            "max_levels": 24,
            "max_recent_events": 40,
            "max_swing_points": 30,
            "snapshot_levels_limit": 10,
            "snapshot_events_limit": 8,
            "snapshot_swings_limit": 8,
            "min_confluence_score": 2,
            "memory_enabled": True,
            "memory_min_tests": 2,
            "memory_max_age_days": 5,
            "memory_decay_after_days": 2,
            "memory_decay_weight": 0.5,
            "memory_max_levels": 12,
            "opening_range_enabled": True,
            "opening_range_minutes": 30,
            "opening_range_break_tolerance_pct": 0.05,
            "poc_migration_enabled": True,
            "poc_migration_interval_bars": 30,
            "poc_migration_trend_threshold_pct": 0.20,
            "poc_migration_range_threshold_pct": 0.10,
            "composite_profile_enabled": True,
            "composite_profile_days": 3,
            "composite_profile_current_day_weight": 1.0,
            "spike_detection_enabled": True,
            "spike_min_wick_ratio": 0.60,
            "prior_day_anchors_enabled": True,
            "gap_analysis_enabled": True,
            "gap_min_pct": 0.30,
            "gap_momentum_threshold_pct": 2.0,
            "rvol_filter_enabled": True,
            "rvol_lookback_bars": 20,
            "rvol_min_threshold": 0.80,
            "rvol_strong_threshold": 1.50,
            "adaptive_window_enabled": True,
            "adaptive_window_min_bars": 6,
            "adaptive_window_rvol_threshold": 1.0,
            "adaptive_window_atr_ratio_max": 1.5,
        },
        "levels": [],
        "next_level_id": 1,
        "swing_highs": [],
        "swing_lows": [],
        "recent_events": [],
        "memory_profiles": [],
        "opening_range": {
            "enabled": True,
            "bars_target": 30,
            "bars_collected": 0,
            "complete": False,
            "high": None,
            "low": None,
            "mid": None,
            "breakout_direction": None,
            "breakout_bar_index": -1,
            "breakout_price": None,
        },
        "poc_history": [],
        "volume_profile_bins": {},
        "volume_profile_bin_size": 0.0,
        "volume_profile": {
            "bin_size": 0.0,
            "total_volume": 0.0,
            "poc_price": None,
            "poc_volume": 0.0,
            "value_area_low": None,
            "value_area_high": None,
            "value_area_coverage": 0.0,
            "bins_count": 0,
            "composite_profile": {},
        },
        "last_bar_index": -1,
        "gap_context": {
            "enabled": True,
            "initialized": False,
            "open_price": None,
            "prev_close": None,
            "gap_pct": 0.0,
            "gap_direction": "none",
            "gap_fill_target": None,
            "favor_side": "none",
            "momentum_bias_side": "none",
        },
        "market_activity": {
            "rvol": None,
            "avg_volume_lookback": None,
            "rvol_lookback_bars": 20,
        },
        "adaptive_window": {
            "enabled": True,
            "ready": False,
            "ready_bar_index": -1,
            "atr_ratio": None,
            "rvol": None,
        },
        "snapshot": {},
    }



def ensure_intraday_levels_state(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return new_state()

    normalized = new_state()
    normalized.update(state)
    if not isinstance(normalized.get("config"), dict):
        normalized["config"] = new_state()["config"]
    else:
        cfg = dict(new_state()["config"])
        cfg.update(normalized["config"])
        normalized["config"] = cfg
    if not isinstance(normalized.get("levels"), list):
        normalized["levels"] = []
    if not isinstance(normalized.get("swing_highs"), list):
        normalized["swing_highs"] = []
    if not isinstance(normalized.get("swing_lows"), list):
        normalized["swing_lows"] = []
    if not isinstance(normalized.get("recent_events"), list):
        normalized["recent_events"] = []
    if not isinstance(normalized.get("memory_profiles"), list):
        normalized["memory_profiles"] = []
    if not isinstance(normalized.get("opening_range"), dict):
        normalized["opening_range"] = dict(new_state()["opening_range"])
    else:
        opening_defaults = dict(new_state()["opening_range"])
        opening_defaults.update(normalized.get("opening_range", {}))
        normalized["opening_range"] = opening_defaults
    if not isinstance(normalized.get("poc_history"), list):
        normalized["poc_history"] = []
    if not isinstance(normalized.get("volume_profile_bins"), dict):
        normalized["volume_profile_bins"] = {}
    if not isinstance(normalized.get("volume_profile"), dict):
        normalized["volume_profile"] = new_state()["volume_profile"]
    else:
        vp_defaults = dict(new_state()["volume_profile"])
        vp_defaults.update(normalized.get("volume_profile", {}))
        if not isinstance(vp_defaults.get("composite_profile"), dict):
            vp_defaults["composite_profile"] = {}
        normalized["volume_profile"] = vp_defaults
    if not isinstance(normalized.get("gap_context"), dict):
        normalized["gap_context"] = dict(new_state()["gap_context"])
    else:
        gap_defaults = dict(new_state()["gap_context"])
        gap_defaults.update(normalized.get("gap_context", {}))
        normalized["gap_context"] = gap_defaults
    if not isinstance(normalized.get("market_activity"), dict):
        normalized["market_activity"] = dict(new_state()["market_activity"])
    else:
        market_defaults = dict(new_state()["market_activity"])
        market_defaults.update(normalized.get("market_activity", {}))
        normalized["market_activity"] = market_defaults
    if not isinstance(normalized.get("adaptive_window"), dict):
        normalized["adaptive_window"] = dict(new_state()["adaptive_window"])
    else:
        adaptive_defaults = dict(new_state()["adaptive_window"])
        adaptive_defaults.update(normalized.get("adaptive_window", {}))
        normalized["adaptive_window"] = adaptive_defaults
    return normalized




__all__ = ["new_state", "ensure_intraday_levels_state"]
