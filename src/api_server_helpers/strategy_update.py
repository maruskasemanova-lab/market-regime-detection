"""Helpers for applying strategy update payloads in API routes."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

from ..strategy_formula_engine import StrategyFormulaValidationError


def _normalize_mode(value: Any) -> Optional[str]:
    normalized = str(value or "").strip().lower()
    if normalized in {"global", "custom"}:
        return normalized
    return None


def _normalize_optional_positive(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _set_on_pair(
    *,
    strat: Any,
    dtm_strat: Any,
    key: str,
    value: Any,
) -> None:
    setattr(strat, key, value)
    if dtm_strat and hasattr(dtm_strat, key):
        setattr(dtm_strat, key, value)


def apply_strategy_param_updates(
    *,
    strat: Any,
    dtm_strat: Any,
    params: Mapping[str, Any],
    validate_formula: Callable[[Any], Mapping[str, Any]],
    coerce_regimes: Callable[[Any], Any],
) -> Dict[str, Any]:
    """Apply update params to strategy objects and return changed fields."""
    updated_fields: Dict[str, Any] = {}

    for key, val in params.items():
        if key in {"custom_entry_formula", "custom_exit_formula"}:
            normalized_formula = str(val or "").strip()
            try:
                formula_info = validate_formula(normalized_formula)
            except StrategyFormulaValidationError as exc:
                raise ValueError(f"Invalid {key}: {exc}") from exc
            normalized_formula = str(formula_info.get("normalized", "") or "")
            _set_on_pair(
                strat=strat,
                dtm_strat=dtm_strat,
                key=key,
                value=normalized_formula,
            )
            updated_fields[key] = normalized_formula
            continue

        if key in {"custom_entry_formula_enabled", "custom_exit_formula_enabled"}:
            normalized_enabled = bool(val)
            _set_on_pair(
                strat=strat,
                dtm_strat=dtm_strat,
                key=key,
                value=normalized_enabled,
            )
            updated_fields[key] = normalized_enabled
            continue

        if key == "allowed_regimes" and isinstance(val, list):
            regimes = coerce_regimes(val)
            setattr(strat, "allowed_regimes", regimes)
            updated_fields[key] = [regime.value for regime in regimes]
            if dtm_strat:
                setattr(dtm_strat, "allowed_regimes", regimes)
            continue

        if key in {"trailing_stop_mode", "exit_mode"}:
            mode = _normalize_mode(val)
            if mode is None:
                continue
            _set_on_pair(strat=strat, dtm_strat=dtm_strat, key="exit_mode", value=mode)
            _set_on_pair(strat=strat, dtm_strat=dtm_strat, key="trailing_stop_mode", value=mode)
            updated_fields["exit_mode"] = mode
            updated_fields["trailing_stop_mode"] = mode
            continue

        if key == "risk_mode":
            mode = _normalize_mode(val)
            if mode is None:
                continue
            _set_on_pair(strat=strat, dtm_strat=dtm_strat, key="risk_mode", value=mode)
            updated_fields[key] = mode
            continue

        if key in {
            "global_trailing_stop_pct",
            "global_rr_ratio",
            "global_atr_stop_multiplier",
            "global_volume_stop_pct",
            "global_min_stop_loss_pct",
        }:
            normalized_global = _normalize_optional_positive(val)
            _set_on_pair(
                strat=strat,
                dtm_strat=dtm_strat,
                key=key,
                value=normalized_global,
            )
            updated_fields[key] = normalized_global
            continue

        if key == "trailing_stop_pct":
            try:
                normalized_trailing = float(val)
            except (TypeError, ValueError):
                continue
            if normalized_trailing <= 0:
                continue
            _set_on_pair(
                strat=strat,
                dtm_strat=dtm_strat,
                key="trailing_stop_pct",
                value=normalized_trailing,
            )
            updated_fields[key] = normalized_trailing
            continue

        if hasattr(strat, key) and isinstance(val, (int, float, bool, str, type(None))):
            _set_on_pair(
                strat=strat,
                dtm_strat=dtm_strat,
                key=key,
                value=val,
            )
            updated_fields[key] = val

    return updated_fields


__all__ = [
    "apply_strategy_param_updates",
]
