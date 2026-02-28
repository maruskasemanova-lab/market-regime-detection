"""Apply/merge helpers for DayTradingManager run defaults."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from ..day_trading_models import TradingConfig
from .run_defaults import collect_non_none_run_default_overrides


def apply_run_defaults(
    *,
    manager: Any,
    key: Tuple[str, str],
    local_values: Mapping[str, Any],
    momentum_diversification: Any,
    trading_config: Any,
) -> None:
    """Merge explicit overrides for a run and persist canonical values."""

    defaults = manager.run_defaults.get(key, {})
    if not isinstance(defaults, dict):
        defaults = {}

    updates: Dict[str, Any] = {}
    if isinstance(trading_config, TradingConfig):
        updates.update(trading_config.to_session_params())
    elif isinstance(trading_config, dict):
        updates.update(trading_config)

    updates.update(collect_non_none_run_default_overrides(local_values))

    if momentum_diversification is not None:
        updates["momentum_diversification"] = manager._normalize_momentum_diversification_config(
            momentum_diversification
        )
    elif "momentum_diversification" in updates and isinstance(updates["momentum_diversification"], dict):
        updates["momentum_diversification"] = manager._normalize_momentum_diversification_config(
            updates["momentum_diversification"]
        )

    if not updates:
        return

    validated = manager._canonical_trading_config({**defaults, **updates}).to_session_params()
    for field_name in updates:
        defaults[field_name] = validated[field_name]
    manager.run_defaults[key] = defaults


__all__ = ["apply_run_defaults"]

