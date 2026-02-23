"""Runtime exit-policy formula hook definitions and evaluators."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Tuple

from .runtime_formula_engine import (
    RuntimeFormulaEvaluationError,
    RuntimeFormulaValidationError,
    evaluate_runtime_formula,
    validate_runtime_formula,
)


RUNTIME_EXIT_FORMULA_SPECS: Dict[str, Dict[str, Any]] = {
    "break_even_movement": {
        "enabled_key": "break_even_movement_formula_enabled",
        "formula_key": "break_even_movement_formula",
        "allowed_variables": (
            "side",
            "bars_held_count",
            "min_hold_for_break_even",
            "bars_held_meets_min_hold",
            "entry_price",
            "mfe_pct",
            "mfe_r",
            "risk_abs",
            "risk_pct",
            "atr_1m_pct",
            "atr_5m_pct",
            "regime_5m",
            "l2_bias_5m",
            "spread_bps",
            "required_mfe_pct",
            "required_r",
            "required_r_mfe_pct",
            "base_activation_mfe_pct",
            "atr_activation_mfe_pct",
            "base_r",
            "bias_threshold",
            "directional_bias",
            "use_levels",
            "use_l2",
            "proof_required",
            "proof_passed",
            "no_go_blocked",
            "movement_passed",
            "levels_proof_enabled",
            "levels_proof_passed",
            "levels_no_go_blocked",
            "levels_nearest_distance_pct",
            "levels_confluence",
            "levels_tests",
            "l2_proof_enabled",
            "l2_proof_passed",
            "l2_signed_aggression",
            "l2_imbalance",
            "l2_book_pressure",
            "l2_spread_bps",
            "break_even_active",
            "trailing_stop_active",
            "partial_tp_filled",
        ),
    },
    "break_even_proof": {
        "enabled_key": "break_even_proof_formula_enabled",
        "formula_key": "break_even_proof_formula",
        "allowed_variables": (
            "side",
            "bars_held_count",
            "entry_price",
            "mfe_pct",
            "mfe_r",
            "risk_pct",
            "regime_5m",
            "spread_bps",
            "use_levels",
            "use_l2",
            "proof_required",
            "proof_passed",
            "no_go_blocked",
            "movement_passed",
            "levels_proof_enabled",
            "levels_proof_passed",
            "levels_no_go_blocked",
            "levels_nearest_distance_pct",
            "levels_confluence",
            "levels_tests",
            "l2_proof_enabled",
            "l2_proof_passed",
            "l2_signed_aggression",
            "l2_imbalance",
            "l2_book_pressure",
            "l2_spread_bps",
            "break_even_active",
            "partial_tp_filled",
        ),
    },
    "break_even_activation": {
        "enabled_key": "break_even_activation_formula_enabled",
        "formula_key": "break_even_activation_formula",
        "allowed_variables": (
            "side",
            "bars_held_count",
            "min_hold_for_break_even",
            "bars_held_meets_min_hold",
            "entry_price",
            "mfe_pct",
            "mfe_r",
            "risk_abs",
            "risk_pct",
            "atr_1m_pct",
            "atr_5m_pct",
            "regime_5m",
            "l2_bias_5m",
            "spread_bps",
            "required_mfe_pct",
            "required_r",
            "required_r_mfe_pct",
            "movement_passed",
            "proof_required",
            "proof_passed",
            "no_go_blocked",
            "use_levels",
            "use_l2",
            "levels_proof_enabled",
            "levels_proof_passed",
            "levels_no_go_blocked",
            "levels_nearest_distance_pct",
            "levels_confluence",
            "levels_tests",
            "l2_proof_enabled",
            "l2_proof_passed",
            "l2_signed_aggression",
            "l2_imbalance",
            "l2_book_pressure",
            "l2_spread_bps",
            "break_even_active",
            "trailing_stop_active",
            "partial_tp_filled",
        ),
    },
    "break_even_trailing_handoff": {
        "enabled_key": "break_even_trailing_handoff_formula_enabled",
        "formula_key": "break_even_trailing_handoff_formula",
        "allowed_variables": (
            "side",
            "regime",
            "regime_5m",
            "bars_held_count",
            "entry_price",
            "mfe_pct",
            "mfe_r",
            "trailing_stop_pct",
            "trailing_stop_active",
            "trailing_stop_price",
            "trailing_activation_pnl_met",
            "break_even_stop_active",
            "break_even_state",
            "skip_trailing_for_choppy",
            "trailing_enabled_in_choppy",
            "highest_price",
            "lowest_price",
            "stop_loss",
            "trailing_handoff_base",
        ),
    },
    "time_exit": {
        "enabled_key": "time_exit_formula_enabled",
        "formula_key": "time_exit_formula",
        "allowed_variables": (
            "side",
            "bars_held_count",
            "current_bar_index",
            "entry_price",
            "current_price",
            "time_exit_bars",
            "choppy_time_exit_bars",
            "regime",
            "base_limit_bars",
            "limit_bars",
            "signed_aggression",
            "flow_score",
            "flow_trend",
            "favorable",
            "quality_favorable",
            "is_profitable",
            "time_exit_base",
        ),
    },
    "adverse_flow_exit": {
        "enabled_key": "adverse_flow_exit_formula_enabled",
        "formula_key": "adverse_flow_exit_formula",
        "allowed_variables": (
            "side",
            "bars_held_count",
            "effective_min_hold",
            "min_hold_met",
            "has_l2_coverage",
            "signed_aggression",
            "directional_consistency",
            "book_pressure_avg",
            "threshold",
            "consistency_threshold",
            "book_pressure_threshold",
            "unrealized_pnl",
            "absorption_rate",
            "absorption_override",
            "flow_score_trend_3bar",
            "is_red_bar",
            "is_green_bar",
            "adverse_flow",
            "adverse_book",
            "extreme_aggression",
            "adverse_flow_base",
        ),
    },
}


RUNTIME_EXIT_FORMULA_FIELD_NAMES: Tuple[str, ...] = tuple(
    field_name
    for hook in RUNTIME_EXIT_FORMULA_SPECS.values()
    for field_name in (hook["enabled_key"], hook["formula_key"])
)


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off"}:
            return False
    return bool(default)


def runtime_exit_formula_allowed_variables(hook: str) -> Tuple[str, ...]:
    spec = RUNTIME_EXIT_FORMULA_SPECS.get(str(hook))
    if not isinstance(spec, dict):
        raise KeyError(f"Unknown runtime exit formula hook: {hook}")
    return tuple(spec.get("allowed_variables", ()) or ())


def normalize_runtime_exit_formula_fields(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and normalize runtime exit formula fields in a config-like mapping."""
    if not isinstance(raw, Mapping):
        return {}
    normalized: Dict[str, Any] = {}
    for hook_name, spec in RUNTIME_EXIT_FORMULA_SPECS.items():
        enabled_key = str(spec["enabled_key"])
        formula_key = str(spec["formula_key"])
        raw_enabled = raw.get(enabled_key, False)
        raw_formula = raw.get(formula_key, "")
        enabled = _coerce_bool(raw_enabled, default=False)
        formula_text = str(raw_formula or "").strip()
        normalized[enabled_key] = enabled
        normalized[formula_key] = formula_text
        if not formula_text:
            continue
        try:
            validate_runtime_formula(
                formula_text,
                allowed_variables=runtime_exit_formula_allowed_variables(hook_name),
            )
        except RuntimeFormulaValidationError as exc:
            raise ValueError(f"Invalid {formula_key}: {exc}") from exc
    return normalized


def evaluate_runtime_exit_formula(
    *,
    session: Any,
    hook: str,
    context: Mapping[str, Any],
    default_passed: bool,
) -> Dict[str, Any]:
    spec = RUNTIME_EXIT_FORMULA_SPECS.get(str(hook))
    if not isinstance(spec, dict):
        return {
            "hook": str(hook),
            "enabled": False,
            "formula": "",
            "passed": bool(default_passed),
            "error": f"Unknown runtime formula hook '{hook}'.",
            "used_fallback": True,
        }

    enabled_key = str(spec["enabled_key"])
    formula_key = str(spec["formula_key"])
    configured_enabled = bool(getattr(session, enabled_key, False))
    formula_text = str(getattr(session, formula_key, "") or "").strip()
    active = bool(configured_enabled and formula_text)
    result: Dict[str, Any] = {
        "hook": str(hook),
        "enabled": active,
        "configured_enabled": configured_enabled,
        "formula": formula_text,
        "passed": bool(default_passed),
        "error": None,
        "used_fallback": False,
    }
    if not active:
        return result

    allowed_variables = runtime_exit_formula_allowed_variables(str(hook))
    try:
        info = validate_runtime_formula(formula_text, allowed_variables=allowed_variables)
        result["variables"] = list(info.get("variables", []))
        result["passed"] = bool(
            evaluate_runtime_formula(
                formula_text,
                context,
                allowed_variables=allowed_variables,
            )
        )
        return result
    except (RuntimeFormulaValidationError, RuntimeFormulaEvaluationError) as exc:
        # Runtime config formulas are optional; preserve built-in decision on runtime errors.
        result["error"] = str(exc)
        result["used_fallback"] = True
        result["passed"] = bool(default_passed)
        return result


__all__ = [
    "RUNTIME_EXIT_FORMULA_FIELD_NAMES",
    "RUNTIME_EXIT_FORMULA_SPECS",
    "evaluate_runtime_exit_formula",
    "normalize_runtime_exit_formula_fields",
    "runtime_exit_formula_allowed_variables",
]
