"""Configuration normalization helpers for DayTradingManager."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .strategies.base_strategy import Regime
from .day_trading_models import TradingConfig, TradingSession
from .intraday_levels import ensure_intraday_levels_state


class DayTradingConfigService:
    """Owns normalization/parsing logic for adaptive day-trading config."""

    def __init__(self, manager: Any):
        self.manager = manager

    def canonical_strategy_key(self, strategy_name: str) -> str:
        """Normalize strategy name to one of manager strategy keys."""
        if not strategy_name:
            return ""

        normalized = strategy_name.strip().replace("-", "_").replace(" ", "_")
        lowered = normalized.lower()
        if lowered in self.manager.strategies:
            return lowered

        snake = re.sub(r'(?<!^)(?=[A-Z])', '_', normalized).lower()
        snake = re.sub(r'__+', '_', snake)
        if snake in self.manager.strategies:
            return snake

        compact = re.sub(r'[^a-z0-9]', '', lowered)
        for key in self.manager.strategies.keys():
            if key.replace('_', '') == compact:
                return key

        return snake

    @staticmethod
    def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def safe_bool(value: Any, default: bool = False) -> bool:
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
        return default

    def normalize_stop_loss_mode(self, mode: Any) -> str:
        normalized = str(mode or "strategy").strip().lower()
        if normalized not in self.manager.VALID_STOP_LOSS_MODES:
            return "strategy"
        return normalized

    def normalize_strategy_selection_mode(self, mode: Any) -> str:
        normalized = str(mode or "adaptive_top_n").strip().lower()
        if normalized not in self.manager.VALID_STRATEGY_SELECTION_MODES:
            return "adaptive_top_n"
        return normalized

    @staticmethod
    def normalize_max_active_strategies(value: Any, default: int = 3) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = int(default)
        return max(1, min(20, parsed))

    @staticmethod
    def normalize_non_negative_int(value: Any, default: int = 0, max_value: int = 10_000) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = int(default)
        return max(0, min(max_value, parsed))

    def default_macro_preference_map(self) -> Dict[str, List[str]]:
        return {
            "TRENDING": list(self.manager.default_preference.get(Regime.TRENDING, [])),
            "CHOPPY": list(self.manager.default_preference.get(Regime.CHOPPY, [])),
            "MIXED": list(self.manager.default_preference.get(Regime.MIXED, [])),
        }

    def normalize_strategy_list(
        self,
        raw_value: Any,
        fallback: Optional[List[str]] = None,
    ) -> List[str]:
        source = raw_value if isinstance(raw_value, list) else list(fallback or [])
        normalized: List[str] = []
        seen = set()
        for candidate in source:
            name = self.canonical_strategy_key(candidate)
            if not name or name in seen:
                continue
            if name not in self.manager.strategies:
                continue
            seen.add(name)
            normalized.append(name)
        return normalized

    def normalize_preference_map(
        self,
        raw_value: Any,
        keys: Tuple[str, ...],
        fallback_map: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        source = raw_value if isinstance(raw_value, dict) else {}
        normalized: Dict[str, List[str]] = {}
        for key in keys:
            normalized[key] = self.normalize_strategy_list(
                source.get(key),
                fallback=fallback_map.get(key, []),
            )
        return normalized

    def normalize_momentum_route_name(self, value: Any, default: str = "continuation") -> str:
        normalized = str(value or default).strip().lower()
        if normalized not in self.manager.MOMENTUM_ROUTE_KEYS:
            return default
        return normalized

    @staticmethod
    def normalize_momentum_sleeve_id(value: Any, fallback: str) -> str:
        raw = str(value or "").strip().lower()
        cleaned = re.sub(r"[^a-z0-9_]+", "_", raw)
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        return cleaned or fallback

    @staticmethod
    def default_momentum_route_map() -> Dict[str, List[str]]:
        return {
            "impulse": ["momentum_flow", "momentum", "pullback", "scalp_l2_intrabar"],
            "continuation": ["pullback", "momentum_flow", "momentum", "scalp_l2_intrabar"],
            "defensive": ["absorption_reversal", "exhaustion_fade", "volume_profile"],
        }

    @staticmethod
    def default_micro_route_map() -> Dict[str, str]:
        return {
            "TRENDING_UP": "impulse",
            "TRENDING_DOWN": "impulse",
            "BREAKOUT": "impulse",
            "MIXED": "continuation",
            "ABSORPTION": "defensive",
            "CHOPPY": "defensive",
            "TRANSITION": "defensive",
            "UNKNOWN": "defensive",
        }

    def normalize_micro_regime_list(self, raw_value: Any) -> List[str]:
        if not isinstance(raw_value, list):
            return []
        normalized: List[str] = []
        seen = set()
        for candidate in raw_value:
            name = str(candidate or "").strip().upper()
            if name not in self.manager.MICRO_REGIME_KEYS or name in seen:
                continue
            seen.add(name)
            normalized.append(name)
        return normalized

    def normalize_momentum_diversification_config(
        self,
        raw_value: Any,
        *,
        include_sleeves: bool = True,
    ) -> Dict[str, Any]:
        raw = raw_value if isinstance(raw_value, dict) else {}
        defaults = self.default_momentum_route_map()
        micro_route_defaults = self.default_micro_route_map()

        raw_route_map = raw.get("route_strategy_map")
        route_map_source = raw_route_map if isinstance(raw_route_map, dict) else {}
        route_strategy_map: Dict[str, List[str]] = {}
        for route in self.manager.MOMENTUM_ROUTE_KEYS:
            route_strategy_map[route] = self.normalize_strategy_list(
                route_map_source.get(route),
                fallback=defaults.get(route, []),
            )

        raw_micro_routes = raw.get("micro_regime_routes")
        micro_route_source = raw_micro_routes if isinstance(raw_micro_routes, dict) else {}
        micro_regime_routes: Dict[str, str] = {}
        for micro_key in self.manager.MICRO_REGIME_KEYS:
            micro_regime_routes[micro_key] = self.normalize_momentum_route_name(
                micro_route_source.get(micro_key),
                default=micro_route_defaults.get(micro_key, "continuation"),
            )

        signed_floor = self.safe_float(raw.get("min_signed_aggression"), 0.03)
        imbalance_floor = self.safe_float(raw.get("min_imbalance"), 0.02)
        cvd_floor = self.safe_float(raw.get("min_cvd"), 0.0)
        directional_price_change_floor = self.safe_float(
            raw.get("min_directional_price_change_pct"),
            0.0,
        )
        price_trend_efficiency_floor = self.safe_float(raw.get("min_price_trend_efficiency"), 0.0)
        last_bar_body_floor = self.safe_float(raw.get("min_last_bar_body_ratio"), 0.0)
        last_bar_close_location_floor = self.safe_float(raw.get("min_last_bar_close_location"), 0.0)
        consistency_floor = self.safe_float(raw.get("min_directional_consistency"), 0.35)
        flow_floor = self.safe_float(raw.get("min_flow_score"), 55.0)
        delta_accel_floor = self.safe_float(raw.get("min_delta_acceleration"), 0.0)
        divergence_floor = self.safe_float(raw.get("min_delta_price_divergence"), -0.45)
        route_impulse_floor = self.safe_float(raw.get("route_flow_score_impulse"), 62.0)
        fail_fast_signed_max = self.safe_float(raw.get("fail_fast_signed_aggression_max"), -0.05)
        fail_fast_book_max = self.safe_float(raw.get("fail_fast_book_pressure_max"), -0.08)
        fail_fast_consistency_max = self.safe_float(raw.get("fail_fast_directional_consistency_max"), 0.35)

        normalized = {
            "enabled": self.safe_bool(raw.get("enabled"), False),
            "apply_to_strategies": self.normalize_strategy_list(
                raw.get("apply_to_strategies"),
                fallback=list(self.manager.DEFAULT_MOMENTUM_STRATEGIES),
            ),
            "allowed_micro_regimes": self.normalize_micro_regime_list(raw.get("allowed_micro_regimes")),
            "blocked_micro_regimes": self.normalize_micro_regime_list(raw.get("blocked_micro_regimes")),
            "require_l2_coverage": self.safe_bool(raw.get("require_l2_coverage"), True),
            "min_flow_score": max(0.0, min(100.0, float(flow_floor or 0.0))),
            "min_directional_consistency": max(
                0.0, min(1.0, float(consistency_floor if consistency_floor is not None else 0.35))
            ),
            "min_signed_aggression": max(
                0.0, min(1.0, float(signed_floor if signed_floor is not None else 0.03))
            ),
            "min_imbalance": max(
                0.0, min(1.0, float(imbalance_floor if imbalance_floor is not None else 0.02))
            ),
            "min_cvd": float(cvd_floor or 0.0),
            "min_directional_price_change_pct": float(directional_price_change_floor or 0.0),
            "min_price_trend_efficiency": max(
                0.0,
                min(
                    1.0,
                    float(
                        price_trend_efficiency_floor
                        if price_trend_efficiency_floor is not None
                        else 0.0
                    ),
                ),
            ),
            "min_last_bar_body_ratio": max(
                0.0,
                min(
                    1.0,
                    float(last_bar_body_floor if last_bar_body_floor is not None else 0.0),
                ),
            ),
            "min_last_bar_close_location": max(
                0.0,
                min(
                    1.0,
                    float(
                        last_bar_close_location_floor
                        if last_bar_close_location_floor is not None
                        else 0.0
                    ),
                ),
            ),
            "min_delta_acceleration": float(delta_accel_floor or 0.0),
            "min_delta_price_divergence": float(divergence_floor if divergence_floor is not None else -0.45),
            "gate_mode": str(raw.get("gate_mode", "weighted")).lower().strip(),
            "gate_threshold": max(0.0, min(1.0, self.safe_float(raw.get("gate_threshold"), 0.55) or 0.55)),
            "gate_flow_floor": max(0.0, self.safe_float(raw.get("gate_flow_floor"), 40.0) or 40.0),
            "route_enabled": self.safe_bool(raw.get("route_enabled"), False),
            "route_require_l2_coverage": self.safe_bool(raw.get("route_require_l2_coverage"), True),
            "route_flow_score_impulse": max(0.0, min(100.0, float(route_impulse_floor or 0.0))),
            "route_strategy_map": route_strategy_map,
            "micro_regime_routes": micro_regime_routes,
            "fail_fast_exit_enabled": self.safe_bool(raw.get("fail_fast_exit_enabled"), False),
            "fail_fast_max_bars": max(
                1,
                min(30, int(self.safe_float(raw.get("fail_fast_max_bars"), 3) or 3)),
            ),
            "fail_fast_signed_aggression_max": min(
                0.0,
                float(fail_fast_signed_max if fail_fast_signed_max is not None else -0.05),
            ),
            "fail_fast_book_pressure_max": min(
                0.0,
                float(fail_fast_book_max if fail_fast_book_max is not None else -0.08),
            ),
            "fail_fast_directional_consistency_max": max(
                0.0,
                min(
                    1.0,
                    float(
                        fail_fast_consistency_max
                        if fail_fast_consistency_max is not None
                        else 0.35
                    ),
                ),
            ),
        }

        if include_sleeves and isinstance(raw.get("sleeves"), list):
            sleeves: List[Dict[str, Any]] = []
            seen_ids = set()
            for idx, item in enumerate(raw.get("sleeves", [])):
                if not isinstance(item, dict):
                    continue
                sleeve_cfg = self.normalize_momentum_diversification_config(
                    item,
                    include_sleeves=False,
                )
                sleeve_id = self.normalize_momentum_sleeve_id(
                    item.get("sleeve_id") or item.get("name"),
                    fallback=f"sleeve_{idx + 1}",
                )
                if sleeve_id in seen_ids:
                    continue
                seen_ids.add(sleeve_id)
                sleeve_cfg["sleeve_id"] = sleeve_id

                if "allocation_weight" in item:
                    weight = self.safe_float(item.get("allocation_weight"), 0.0)
                    sleeve_cfg["allocation_weight"] = max(
                        0.0,
                        min(1.0, float(weight if weight is not None else 0.0)),
                    )
                else:
                    sleeve_cfg["allocation_weight"] = 0.0

                sleeves.append(sleeve_cfg)
                if len(sleeves) >= self.manager.MAX_MOMENTUM_SLEEVES:
                    break
            if sleeves:
                normalized["sleeves"] = sleeves

        return normalized

    def normalize_adaptive_config(self, raw_value: Any) -> Dict[str, Any]:
        raw = raw_value if isinstance(raw_value, dict) else {}
        macro_defaults = self.default_macro_preference_map()
        micro_defaults = {
            key: list(values)
            for key, values in self.manager.micro_regime_preference.items()
            if key in self.manager.MICRO_REGIME_KEYS
        }
        return {
            "flow_bias_enabled": self.safe_bool(raw.get("flow_bias_enabled"), True),
            "use_ohlcv_fallbacks": self.safe_bool(raw.get("use_ohlcv_fallbacks"), True),
            "strict_preference_selection": self.safe_bool(
                raw.get("strict_preference_selection"),
                False,
            ),
            "min_active_bars_before_switch": self.normalize_non_negative_int(
                raw.get("min_active_bars_before_switch"),
                default=0,
            ),
            "switch_cooldown_bars": self.normalize_non_negative_int(
                raw.get("switch_cooldown_bars"),
                default=0,
            ),
            "flow_bias_strategies": self.normalize_strategy_list(
                raw.get("flow_bias_strategies"),
                fallback=list(self.manager.DEFAULT_FLOW_BIAS_STRATEGIES),
            ),
            "regime_preferences": self.normalize_preference_map(
                raw.get("regime_preferences"),
                keys=self.manager.MACRO_REGIME_KEYS,
                fallback_map=macro_defaults,
            ),
            "micro_regime_preferences": self.normalize_preference_map(
                raw.get("micro_regime_preferences"),
                keys=self.manager.MICRO_REGIME_KEYS,
                fallback_map=micro_defaults,
            ),
            "momentum_diversification": self.normalize_momentum_diversification_config(
                raw.get("momentum_diversification")
            ),
            "context_exit_response": raw.get("context_exit_response", {})
            if isinstance(raw.get("context_exit_response"), dict) else {},
        }

    def canonical_trading_config(self, raw: Dict[str, Any]) -> TradingConfig:
        """Validate/clamp runtime config through canonical schema."""
        return TradingConfig.from_dict(raw if isinstance(raw, dict) else {})

    def apply_trading_config_to_session(
        self,
        session: TradingSession,
        config: TradingConfig,
        *,
        normalize_momentum: bool = True,
    ) -> None:
        """Apply canonical config to session and keep compat fields aligned."""
        session.apply_trading_config(config)
        intraday_state = ensure_intraday_levels_state(
            getattr(session, "intraday_levels_state", {})
        )
        intraday_cfg = dict(intraday_state.get("config", {}))
        intraday_cfg.update(
            {
                "enabled": bool(config.intraday_levels_enabled),
                "swing_left_bars": int(config.intraday_levels_swing_left_bars),
                "swing_right_bars": int(config.intraday_levels_swing_right_bars),
                "test_tolerance_pct": float(config.intraday_levels_test_tolerance_pct),
                "break_tolerance_pct": float(config.intraday_levels_break_tolerance_pct),
                "breakout_volume_lookback": int(config.intraday_levels_breakout_volume_lookback),
                "breakout_volume_multiplier": float(
                    config.intraday_levels_breakout_volume_multiplier
                ),
                "volume_profile_bin_size_pct": float(
                    config.intraday_levels_volume_profile_bin_size_pct
                ),
                "value_area_pct": float(config.intraday_levels_value_area_pct),
                "entry_quality_enabled": bool(config.intraday_levels_entry_quality_enabled),
                "min_levels_for_context": int(config.intraday_levels_min_levels_for_context),
                "entry_tolerance_pct": float(config.intraday_levels_entry_tolerance_pct),
                "break_cooldown_bars": int(config.intraday_levels_break_cooldown_bars),
                "rotation_max_tests": int(config.intraday_levels_rotation_max_tests),
                "rotation_volume_max_ratio": float(
                    config.intraday_levels_rotation_volume_max_ratio
                ),
                "recent_bounce_lookback_bars": int(
                    config.intraday_levels_recent_bounce_lookback_bars
                ),
                "require_recent_bounce_for_mean_reversion": bool(
                    config.intraday_levels_require_recent_bounce_for_mean_reversion
                ),
                "momentum_break_max_age_bars": int(
                    config.intraday_levels_momentum_break_max_age_bars
                ),
                "momentum_min_room_pct": float(config.intraday_levels_momentum_min_room_pct),
                "momentum_min_broken_ratio": float(
                    config.intraday_levels_momentum_min_broken_ratio
                ),
                "min_confluence_score": int(config.intraday_levels_min_confluence_score),
                "memory_enabled": bool(config.intraday_levels_memory_enabled),
                "memory_min_tests": int(config.intraday_levels_memory_min_tests),
                "memory_max_age_days": int(config.intraday_levels_memory_max_age_days),
                "memory_decay_after_days": int(
                    config.intraday_levels_memory_decay_after_days
                ),
                "memory_decay_weight": float(config.intraday_levels_memory_decay_weight),
                "memory_max_levels": int(config.intraday_levels_memory_max_levels),
                "opening_range_enabled": bool(config.intraday_levels_opening_range_enabled),
                "opening_range_minutes": int(config.intraday_levels_opening_range_minutes),
                "opening_range_break_tolerance_pct": float(
                    config.intraday_levels_opening_range_break_tolerance_pct
                ),
                "poc_migration_enabled": bool(config.intraday_levels_poc_migration_enabled),
                "poc_migration_interval_bars": int(
                    config.intraday_levels_poc_migration_interval_bars
                ),
                "poc_migration_trend_threshold_pct": float(
                    config.intraday_levels_poc_migration_trend_threshold_pct
                ),
                "poc_migration_range_threshold_pct": float(
                    config.intraday_levels_poc_migration_range_threshold_pct
                ),
                "composite_profile_enabled": bool(
                    config.intraday_levels_composite_profile_enabled
                ),
                "composite_profile_days": int(config.intraday_levels_composite_profile_days),
                "composite_profile_current_day_weight": float(
                    config.intraday_levels_composite_profile_current_day_weight
                ),
                "spike_detection_enabled": bool(
                    config.intraday_levels_spike_detection_enabled
                ),
                "spike_min_wick_ratio": float(config.intraday_levels_spike_min_wick_ratio),
                "prior_day_anchors_enabled": bool(
                    config.intraday_levels_prior_day_anchors_enabled
                ),
                "gap_analysis_enabled": bool(
                    config.intraday_levels_gap_analysis_enabled
                ),
                "gap_min_pct": float(config.intraday_levels_gap_min_pct),
                "gap_momentum_threshold_pct": float(
                    config.intraday_levels_gap_momentum_threshold_pct
                ),
                "rvol_filter_enabled": bool(config.intraday_levels_rvol_filter_enabled),
                "rvol_lookback_bars": int(config.intraday_levels_rvol_lookback_bars),
                "rvol_min_threshold": float(config.intraday_levels_rvol_min_threshold),
                "rvol_strong_threshold": float(
                    config.intraday_levels_rvol_strong_threshold
                ),
                "adaptive_window_enabled": bool(
                    config.intraday_levels_adaptive_window_enabled
                ),
                "adaptive_window_min_bars": int(
                    config.intraday_levels_adaptive_window_min_bars
                ),
                "adaptive_window_rvol_threshold": float(
                    config.intraday_levels_adaptive_window_rvol_threshold
                ),
                "adaptive_window_atr_ratio_max": float(
                    config.intraday_levels_adaptive_window_atr_ratio_max
                ),
            }
        )
        intraday_state["config"] = intraday_cfg
        session.intraday_levels_state = intraday_state
        momentum_payload = config.momentum_diversification
        if normalize_momentum and isinstance(momentum_payload, dict) and momentum_payload:
            session.momentum_diversification = self.normalize_momentum_diversification_config(
                momentum_payload
            )
            session.momentum_diversification_override = True
        elif not momentum_payload:
            session.momentum_diversification = {}
            session.momentum_diversification_override = False

        if config.regime_filter is not None:
             session.regime_filter_override = [
                 str(r).strip().upper() for r in config.regime_filter if str(r).strip()
             ]
        else:
             session.regime_filter_override = None

        # Propagate orchestrator weights from TradingConfig to orchestrator
        orch = getattr(self.manager, "orchestrator", None)
        if orch is not None:
            orch.config.strategy_weight = float(config.orchestrator_strategy_weight)
            orch.config.strategy_only_threshold = float(
                config.orchestrator_strategy_only_threshold
            )
            if hasattr(orch, "evidence_engine") and orch.evidence_engine is not None:
                orch.evidence_engine._strategy_only_threshold = float(
                    config.orchestrator_strategy_only_threshold
                )

    def is_day_allowed(self, date_str: str, ticker: str) -> bool:
        """Check if trading day is allowed by ticker-specific avoid_days."""
        ticker_cfg = self.manager.ticker_params.get(str(ticker).upper(), {})
        avoid_days = {str(day).strip().lower() for day in ticker_cfg.get("avoid_days", [])}
        if not avoid_days:
            return True

        try:
            weekday = datetime.strptime(date_str, "%Y-%m-%d").strftime("%A").lower()
            return weekday not in avoid_days
        except Exception:
            return True

    def resolve_max_daily_trades(
        self,
        ticker: str,
        session: Optional[TradingSession] = None,
    ) -> Optional[int]:
        """
        Resolve per-ticker daily trade cap.
        Returns:
            - positive int: enforced trade cap
            - None: unlimited (config value <= 0)
        """
        if isinstance(session, TradingSession):
            raw_override = getattr(session, "max_daily_trades_override", None)
            if raw_override is not None:
                try:
                    resolved_override = int(raw_override)
                except (TypeError, ValueError):
                    resolved_override = 0
                if resolved_override <= 0:
                    return None
                return resolved_override

        ticker_cfg = self.manager.ticker_params.get(str(ticker).upper(), {})
        raw_value = ticker_cfg.get("max_daily_trades", self.manager.max_trades_per_day)
        try:
            resolved = int(raw_value)
        except (TypeError, ValueError):
            try:
                resolved = int(self.manager.max_trades_per_day)
            except (TypeError, ValueError):
                resolved = 0
        if resolved <= 0:
            return None
        return resolved

    def is_mu_choppy_filter_enabled(
        self,
        ticker: str,
        session: Optional[TradingSession] = None,
    ) -> bool:
        """
        Resolve MU CHOPPY hard-block guard from ticker config.
        Defaults to True for backward compatibility.
        """
        if isinstance(session, TradingSession):
            raw_override = getattr(session, "mu_choppy_hard_block_enabled_override", None)
            if raw_override is not None:
                return self.safe_bool(raw_override, True)

        ticker_cfg = self.manager.ticker_params.get(str(ticker).upper(), {})
        if "mu_choppy_hard_block_enabled" not in ticker_cfg:
            return True
        return self.safe_bool(ticker_cfg.get("mu_choppy_hard_block_enabled"), True)

    def get_ticker_strategy_overrides(self, ticker: str, strategy_key: str) -> Dict[str, Any]:
        """Get per-ticker parameter overrides for a strategy."""
        ticker_cfg = self.manager.ticker_params.get(str(ticker).upper(), {})
        if ticker_cfg.get("strategy") == strategy_key:
            return ticker_cfg.get("params", {}) or {}
        return {}
