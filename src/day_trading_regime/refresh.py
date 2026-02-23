"""Regime refresh cadence helper extracted from day_trading_regime_impl."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from ..day_trading_models import TradingSession
from ..strategies.base_strategy import Regime


def regime_maybe_refresh_regime(
    self,
    session: TradingSession,
    current_bar_index: int,
    timestamp: datetime,
) -> Optional[Dict[str, Any]]:
    """
    Re-evaluate regime every N bars during trading.
    Returns update payload only when macro/micro regime or strategy set changed.
    """

    if current_bar_index < 0:
        return None
    orch = session.orchestrator
    adaptive_enabled = bool(
        getattr(getattr(orch, "config", None), "use_adaptive_regime", False)
    )
    adaptive_path_active = bool(
        orch and adaptive_enabled and orch.current_feature_vector
    )
    adaptive_state = orch.current_regime_state if adaptive_path_active else None
    refresh_interval = max(1, int(session.regime_refresh_bars))
    if adaptive_state is not None:
        transition_velocity = max(
            0.0,
            min(1.0, float(getattr(adaptive_state, "transition_velocity", 0.0) or 0.0)),
        )
        refresh_interval = max(
            1,
            int(round(refresh_interval * max(0.0, 1.0 - transition_velocity))),
        )
        if str(getattr(adaptive_state, "micro_regime", "")).upper() in {"BREAKOUT", "TRANSITION"}:
            refresh_interval = 1
    if session.last_regime_refresh_bar_index >= 0:
        bars_since_refresh = current_bar_index - session.last_regime_refresh_bar_index
        if bars_since_refresh < refresh_interval:
            return None
    if len(session.bars) < 20:
        return None

    prev_macro = session.detected_regime or Regime.MIXED
    prev_micro = session.micro_regime
    prev_active = list(session.active_strategies)
    ticker_cfg = self.ticker_params.get(session.ticker.upper(), {})
    adaptive_cfg = self._normalize_adaptive_config(ticker_cfg.get("adaptive"))

    new_macro = self._detect_regime(session)

    if not adaptive_path_active and new_macro != prev_macro:
        pending = getattr(session, "_pending_regime", None)
        if pending == new_macro:
            session.detected_regime = new_macro
            session._pending_regime = None
        else:
            session._pending_regime = new_macro
            session.last_regime_refresh_bar_index = current_bar_index
            return None
    else:
        session._pending_regime = None
        session.detected_regime = new_macro

    requested_active = self._select_strategies(session)
    effective_active = list(requested_active)
    switch_guard = {
        "blocked": False,
        "reasons": [],
        "requested": list(requested_active),
        "effective": list(requested_active),
        "min_active_bars_before_switch": int(
            adaptive_cfg.get("min_active_bars_before_switch", 0)
        ),
        "switch_cooldown_bars": int(adaptive_cfg.get("switch_cooldown_bars", 0)),
    }

    strategy_changed = requested_active != prev_active
    if strategy_changed and prev_active:
        min_active_bars = self._normalize_non_negative_int(
            adaptive_cfg.get("min_active_bars_before_switch"), default=0
        )
        cooldown_bars = self._normalize_non_negative_int(
            adaptive_cfg.get("switch_cooldown_bars"), default=0
        )
        last_switch_bar = int(getattr(session, "last_strategy_switch_bar_index", -1))
        bars_since_switch = (
            current_bar_index - last_switch_bar if last_switch_bar >= 0 else 1_000_000
        )

        if min_active_bars > 0 and bars_since_switch < min_active_bars:
            switch_guard["blocked"] = True
            switch_guard["reasons"].append(
                f"min_active_bars ({bars_since_switch}/{min_active_bars})"
            )
        if cooldown_bars > 0 and bars_since_switch < cooldown_bars:
            switch_guard["blocked"] = True
            switch_guard["reasons"].append(
                f"switch_cooldown ({bars_since_switch}/{cooldown_bars})"
            )

        if switch_guard["blocked"]:
            effective_active = list(prev_active)

    session.active_strategies = effective_active
    session.selected_strategy = (
        "adaptive"
        if len(effective_active) > 1
        else (effective_active[0] if effective_active else None)
    )
    if effective_active != prev_active:
        session.last_strategy_switch_bar_index = current_bar_index
    session.last_regime_refresh_bar_index = current_bar_index
    switch_guard["effective"] = list(effective_active)

    changed = (
        new_macro != prev_macro
        or session.micro_regime != prev_micro
        or effective_active != prev_active
    )
    if not changed:
        return None

    source_bars = session.bars[-80:] if len(session.bars) >= 80 else session.bars
    indicators = self._calculate_indicators(source_bars, session=session)
    flow = indicators.get("order_flow") or {}
    payload = {
        "timestamp": timestamp.isoformat(),
        "bar_index": current_bar_index,
        "regime": new_macro.value,
        "micro_regime": session.micro_regime,
        "strategies": list(effective_active),
        "strategy": session.selected_strategy,
        "selection_warnings": list(getattr(session, "selection_warnings", [])),
        "previous_regime": prev_macro.value,
        "previous_micro_regime": prev_micro,
        "previous_strategies": list(prev_active),
        "switch_guard": switch_guard,
        "indicators": {
            "trend_efficiency": self._calc_trend_efficiency(session.bars),
            "volatility": self._calc_volatility(session.bars),
            "atr": self._latest_indicator_value(indicators, "atr", source_bars),
            "adx": self._latest_indicator_value(indicators, "adx"),
            "flow_score": float(flow.get("flow_score", 0.0) or 0.0),
            "signed_aggression": float(flow.get("signed_aggression", 0.0) or 0.0),
            "absorption_rate": float(flow.get("absorption_rate", 0.0) or 0.0),
            "book_pressure_avg": float(flow.get("book_pressure_avg", 0.0) or 0.0),
            "book_pressure_trend": float(flow.get("book_pressure_trend", 0.0) or 0.0),
            "large_trader_activity": float(flow.get("large_trader_activity", 0.0) or 0.0),
            "vwap_execution_flow": float(flow.get("vwap_execution_flow", 0.0) or 0.0),
        },
    }
    session.regime_history.append(payload)
    return payload
