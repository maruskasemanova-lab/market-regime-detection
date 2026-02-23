"""Strategy-selection helpers for regime-aware routing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import logging

from ..day_trading_models import TradingSession
from ..strategies.base_strategy import Regime

logger = logging.getLogger(__name__)

def regime_resolve_momentum_diversification(
    self,
    session: TradingSession,
    adaptive_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    if (
        bool(session.momentum_diversification_override)
        and isinstance(session.momentum_diversification, dict)
        and session.momentum_diversification
    ):
        return self._normalize_momentum_diversification_config(session.momentum_diversification)
    if isinstance(adaptive_cfg, dict):
        return self._normalize_momentum_diversification_config(
            adaptive_cfg.get("momentum_diversification")
        )
    return self._normalize_momentum_diversification_config({})

def regime_select_momentum_sleeve(
    self,
    momentum_cfg: Dict[str, Any],
    *,
    strategy_key: str = "",
    micro_regime: str = "MIXED",
    has_l2_coverage: Optional[bool] = None,
    preferred_sleeve_id: str = "",
) -> Tuple[Dict[str, Any], Optional[str], str]:
    if not isinstance(momentum_cfg, dict):
        return self._normalize_momentum_diversification_config({}, include_sleeves=False), None, "single"

    sleeves_raw = momentum_cfg.get("sleeves")
    sleeves = [item for item in sleeves_raw if isinstance(item, dict)] if isinstance(sleeves_raw, list) else []
    if not sleeves:
        return self._normalize_momentum_diversification_config(momentum_cfg, include_sleeves=False), None, "single"

    candidates: List[Dict[str, Any]] = []
    for idx, sleeve in enumerate(sleeves):
        if not bool(sleeve.get("enabled", False)):
            continue
        normalized = self._normalize_momentum_diversification_config(sleeve, include_sleeves=False)
        sleeve_id = self._normalize_momentum_sleeve_id(
            sleeve.get("sleeve_id"),
            fallback=f"sleeve_{idx + 1}",
        )
        normalized["sleeve_id"] = sleeve_id
        normalized["allocation_weight"] = max(
            0.0,
            min(1.0, float(self._safe_float(sleeve.get("allocation_weight"), 0.0) or 0.0)),
        )
        candidates.append(normalized)
    if not candidates:
        return self._normalize_momentum_diversification_config(momentum_cfg, include_sleeves=False), None, "single"

    preferred_id = self._normalize_momentum_sleeve_id(preferred_sleeve_id, fallback="")
    if preferred_id:
        for sleeve in candidates:
            if sleeve.get("sleeve_id") == preferred_id:
                return sleeve, preferred_id, "preferred"

    filtered = list(candidates)
    strategy_norm = self._canonical_strategy_key(strategy_key)
    if strategy_norm:
        by_strategy = []
        for sleeve in filtered:
            apply_to = self._normalize_strategy_list(
                sleeve.get("apply_to_strategies"),
                fallback=list(self.DEFAULT_MOMENTUM_STRATEGIES),
            )
            if strategy_norm in apply_to:
                by_strategy.append(sleeve)
        if by_strategy:
            filtered = by_strategy

    micro = str(micro_regime or "MIXED").strip().upper()
    by_micro = []
    for sleeve in filtered:
        blocked = set(sleeve.get("blocked_micro_regimes", []))
        allowed = set(sleeve.get("allowed_micro_regimes", []))
        if micro in blocked:
            continue
        if allowed and micro not in allowed:
            continue
        by_micro.append(sleeve)
    if by_micro:
        filtered = by_micro

    if has_l2_coverage is not None:
        by_l2 = []
        for sleeve in filtered:
            if bool(sleeve.get("require_l2_coverage", True)) and not bool(has_l2_coverage):
                continue
            by_l2.append(sleeve)
        if by_l2:
            filtered = by_l2

    selected = filtered[0] if filtered else candidates[0]
    return selected, selected.get("sleeve_id"), "multi"

def regime_build_momentum_route_candidates(
    self,
    session: TradingSession,
    flow_metrics: Dict[str, Any],
    momentum_cfg: Dict[str, Any],
) -> List[str]:
    micro_regime = (session.micro_regime or "MIXED").upper()
    has_l2 = bool(flow_metrics.get("has_l2_coverage", False))
    flow_score = float(flow_metrics.get("flow_score", 0.0) or 0.0)
    sleeves_raw = momentum_cfg.get("sleeves")
    sleeves = [item for item in sleeves_raw if isinstance(item, dict)] if isinstance(sleeves_raw, list) else []

    route_rank = {"impulse": 1, "continuation": 1, "defensive": 1}
    if micro_regime in {"CHOPPY", "ABSORPTION"}:
        route_rank["defensive"] = 3
    elif micro_regime in {"TRENDING_UP", "TRENDING_DOWN", "BREAKOUT"}:
        route_rank["impulse"] = 3
    else:
        route_rank["continuation"] = 2

    cfg_rows: List[Dict[str, Any]] = []
    if sleeves:
        for idx, sleeve in enumerate(sleeves):
            if not bool(sleeve.get("enabled", False)):
                continue
            normalized = self._normalize_momentum_diversification_config(sleeve, include_sleeves=False)
            normalized["sleeve_id"] = self._normalize_momentum_sleeve_id(
                sleeve.get("sleeve_id"),
                fallback=f"sleeve_{idx + 1}",
            )
            normalized["allocation_weight"] = max(
                0.0,
                min(1.0, float(self._safe_float(sleeve.get("allocation_weight"), 0.0) or 0.0)),
            )
            cfg_rows.append(normalized)
    else:
        cfg_rows.append(self._normalize_momentum_diversification_config(momentum_cfg, include_sleeves=False))

    ranked_routes: List[Tuple[float, int, List[str]]] = []
    for idx, cfg in enumerate(cfg_rows):
        if not bool(cfg.get("enabled", False)) or not bool(cfg.get("route_enabled", False)):
            continue

        route_map = cfg.get("micro_regime_routes", {})
        if not isinstance(route_map, dict):
            route_map = {}
        route_name = self._normalize_momentum_route_name(
            route_map.get(micro_regime),
            default="continuation",
        )

        if not has_l2 and bool(cfg.get("route_require_l2_coverage", True)):
            route_name = "defensive"
        elif (
            has_l2
            and flow_score >= float(cfg.get("route_flow_score_impulse", 62.0))
            and micro_regime in {"TRENDING_UP", "TRENDING_DOWN", "BREAKOUT"}
        ):
            route_name = "impulse"

        route_strategy_map = cfg.get("route_strategy_map", {})
        if not isinstance(route_strategy_map, dict):
            route_strategy_map = {}
        route_candidates = self._normalize_strategy_list(
            route_strategy_map.get(route_name, []),
            fallback=[],
        )
        if not route_candidates:
            continue

        weight = float(cfg.get("allocation_weight", 0.0) or 0.0)
        score = float(route_rank.get(route_name, 1))
        if has_l2 and route_name == "impulse":
            score += 0.2
        score += min(1.0, max(0.0, weight))
        ranked_routes.append((score, -idx, route_candidates))

    if not ranked_routes:
        return []
    ranked_routes.sort(reverse=True)
    return ranked_routes[0][2]

def regime_select_strategies(self, session: TradingSession) -> List[str]:
    """Select active strategies for the detected regime and ticker."""
    regime = session.detected_regime or Regime.MIXED
    micro_regime = (session.micro_regime or "MIXED").upper()
    ticker = session.ticker.upper()
    if not isinstance(getattr(session, "selection_warnings", None), list):
        session.selection_warnings = []
    else:
        session.selection_warnings.clear()

    def add_selection_warning(message: str) -> None:
        if message not in session.selection_warnings:
            session.selection_warnings.append(message)
        logger.warning(
            "strategy-selection warning run=%s ticker=%s date=%s: %s",
            session.run_id,
            ticker,
            session.date,
            message,
        )

    ticker_cfg = self.ticker_params.get(ticker.upper(), {})
    adaptive_cfg = self._normalize_adaptive_config(ticker_cfg.get("adaptive"))
    strict_preferences = self._safe_bool(adaptive_cfg.get("strict_preference_selection"), False)
    momentum_cfg = self._resolve_momentum_diversification(session, adaptive_cfg)
    if not session.momentum_diversification_override:
        session.momentum_diversification = dict(momentum_cfg)
    selection_mode = self._normalize_strategy_selection_mode(
        session.strategy_selection_mode or ticker_cfg.get("strategy_selection_mode")
    )
    max_active = self._normalize_max_active_strategies(
        session.max_active_strategies or ticker_cfg.get("max_active_strategies"), default=3
    )

    # Optional AOS regime filter: if the regime is explicitly disallowed, skip trading.
    # Check session override first (from run config), then fallback to ticker config (AOS loop matching).
    if getattr(session, "regime_filter_override", None) is not None:
        allowed_regimes_cfg = set(session.regime_filter_override)
    else:
        allowed_regimes_cfg = {
            str(r).strip().upper()
            for r in ticker_cfg.get("regime_filter", [])
            if str(r).strip()
        }
    if allowed_regimes_cfg and regime.value not in allowed_regimes_cfg:
        add_selection_warning(
            f"regime {regime.value} blocked by regime_filter={sorted(allowed_regimes_cfg)}"
        )
        return []

    # Build ordered candidates.
    candidates: List[str] = []
    primary = ticker_cfg.get("strategy")
    backup = ticker_cfg.get("backup_strategy")
    if primary:
        candidates.append(primary)
    if backup:
        candidates.append(backup)

    adaptive_micro_preferences = adaptive_cfg.get("micro_regime_preferences", {})
    micro_candidates: List[str] = []
    if isinstance(adaptive_micro_preferences, dict):
        micro_candidates = list(adaptive_micro_preferences.get(micro_regime, []))

    adaptive_regime_preferences = adaptive_cfg.get("regime_preferences", {})
    macro_candidates: List[str] = []
    if isinstance(adaptive_regime_preferences, dict):
        macro_candidates = list(adaptive_regime_preferences.get(regime.value, []))

    if strict_preferences:
        if micro_candidates:
            candidates.extend(micro_candidates)
        else:
            add_selection_warning(
                f"missing micro_regime_preferences entry for micro_regime={micro_regime}"
            )
        if macro_candidates:
            candidates.extend(macro_candidates)
        else:
            add_selection_warning(
                f"missing regime_preferences entry for macro_regime={regime.value}"
            )
    else:
        # Flow-first micro-regime preferences (adaptive override -> defaults).
        if not micro_candidates:
            micro_candidates = list(self.micro_regime_preference.get(micro_regime, []))
        candidates.extend(micro_candidates)

        # Macro regime ordering: adaptive override first, then ticker/global defaults.
        candidates.extend(macro_candidates)
        ticker_prefs = self.ticker_preferences.get(ticker, {})
        candidates.extend(ticker_prefs.get(regime, []))
        candidates.extend(self.default_preference.get(regime, []))

    flow_metrics = self._calculate_order_flow_metrics(session.bars, lookback=min(20, len(session.bars)))
    if not strict_preferences:
        # If L2 coverage is present on current bars, bias toward flow strategies.
        # Otherwise, add OHLCV-based strategies as fallbacks.
        flow_bias_enabled = self._safe_bool(adaptive_cfg.get("flow_bias_enabled"), True)
        use_ohlcv_fallbacks = self._safe_bool(adaptive_cfg.get("use_ohlcv_fallbacks"), True)
        flow_bias_strategies = self._normalize_strategy_list(
            adaptive_cfg.get("flow_bias_strategies"),
            fallback=list(self.DEFAULT_FLOW_BIAS_STRATEGIES),
        )

        if flow_metrics.get("has_l2_coverage", False):
            if flow_bias_enabled and flow_bias_strategies:
                candidates = flow_bias_strategies + candidates
        elif use_ohlcv_fallbacks:
            # No L2 data available — add OHLCV strategies so the evidence engine
            # has strategy signals to work with.
            ohlcv_fallbacks = {
                Regime.TRENDING: ['momentum', 'pullback'],
                Regime.CHOPPY: ['mean_reversion'],
                Regime.MIXED: ['pullback', 'mean_reversion'],
            }
            candidates.extend(ohlcv_fallbacks.get(regime, ['pullback', 'mean_reversion']))

    route_enabled = bool(momentum_cfg.get("enabled", False)) and bool(
        momentum_cfg.get("route_enabled", False)
    )
    if route_enabled or not strict_preferences:
        route_candidates = self._build_momentum_route_candidates(
            session=session,
            flow_metrics=flow_metrics,
            momentum_cfg=momentum_cfg,
        )
        if route_candidates:
            candidates = route_candidates + candidates

    filtered: List[str] = []
    seen = set()
    for raw_name in candidates:
        name = self._canonical_strategy_key(raw_name)
        if not name or name in seen:
            continue
        strat = self.strategies.get(name)
        if not strat:
            continue
        if not getattr(strat, "enabled", True):
            continue
        if hasattr(strat, "allowed_regimes") and regime not in strat.allowed_regimes:
            continue
        seen.add(name)
        filtered.append(name)

    if not filtered:
        if strict_preferences:
            if candidates:
                add_selection_warning(
                    f"no enabled strategies available for strict candidates micro={micro_regime} macro={regime.value}"
                )
            else:
                add_selection_warning(
                    f"no strict strategy candidates resolved for micro={micro_regime} macro={regime.value}"
                )
            return []

        # Fallback: any enabled strategy that supports this regime.
        for name, strat in self.strategies.items():
            if not getattr(strat, "enabled", True):
                continue
            if regime not in getattr(strat, "allowed_regimes", [regime]):
                continue
            filtered.append(name)

    if selection_mode == "all_enabled" and strict_preferences:
        return list(filtered)

    eligible_enabled: List[str] = []
    for name, strat in self.strategies.items():
        if not getattr(strat, "enabled", True):
            continue
        if regime not in getattr(strat, "allowed_regimes", [regime]):
            continue
        eligible_enabled.append(name)

    if selection_mode == "all_enabled":
        selected: List[str] = []
        selected_seen = set()
        for name in filtered + eligible_enabled:
            if name in selected_seen:
                continue
            selected_seen.add(name)
            selected.append(name)
        return selected

    return filtered[:max_active]
