from __future__ import annotations

from typing import Any, Dict

from .shared import to_float as _to_float


def build_update_diagnostics(
    *,
    pos: Any,
    current_bar_index: int,
    bars_held_count: int,
    entry_price: float,
    mfe_pct: float,
    mfe_r: float,
    risk_abs: float,
    risk_pct: float,
    required_mfe_pct: float,
    required_r: float,
    required_r_mfe_pct: float,
    movement_passed: bool,
    proof_required: bool,
    proof_passed: bool,
    level_proof: Dict[str, Any],
    l2_proof: Dict[str, Any],
    no_go_blocked: bool,
    activation_eligible: bool,
    atr_1m_pct: float,
    atr_5m_pct: float,
    regime_5m: str,
    l2_bias_5m: float,
    spread_bps: float | None,
    armed_this_bar: bool,
    moved_this_bar: bool,
    trailing_updated: bool,
    skip_trailing_for_choppy: bool,
    trailing_can_update: bool,
    move_payload: Dict[str, Any],
    last_computed_break_even: Dict[str, Any],
    movement_formula: Dict[str, Any],
    proof_formula: Dict[str, Any],
    activation_formula: Dict[str, Any],
    trailing_handoff_formula: Dict[str, Any],
) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {
        "active": bool(getattr(pos, "break_even_stop_active", False)),
        "state": str(getattr(pos, "break_even_state", "idle") or "idle"),
        "arm_bar_index": getattr(pos, "break_even_arm_bar_index", None),
        "move_bar_index": getattr(pos, "break_even_move_bar_index", None),
        "activation_reason": str(getattr(pos, "break_even_activation_reason", "") or ""),
        "break_even_move_reason": str(getattr(pos, "break_even_move_reason", "") or ""),
        "break_even_costs_pct": float(getattr(pos, "break_even_costs_pct", 0.0) or 0.0),
        "break_even_buffer_pct": float(getattr(pos, "break_even_buffer_pct", 0.0) or 0.0),
        "armed_this_bar": bool(armed_this_bar),
        "moved_this_bar": bool(moved_this_bar),
        "trailing_updated": bool(trailing_updated),
        "skip_trailing_for_choppy": bool(skip_trailing_for_choppy),
        "bar_index": current_bar_index,
        "bars_held": bars_held_count,
        "entry_price": entry_price,
        "stop_loss": _to_float(getattr(pos, "stop_loss", None), 0.0),
        "initial_stop_loss": _to_float(getattr(pos, "initial_stop_loss", None), 0.0),
        "mfe_pct": round(mfe_pct, 6),
        "mfe_r": round(mfe_r, 6) if risk_abs > 0.0 else None,
        "initial_risk_pct": round(risk_pct, 6) if risk_pct > 0.0 else None,
        "required_mfe_pct": round(required_mfe_pct, 6),
        "required_r": round(required_r, 6),
        "required_r_mfe_pct": round(required_r_mfe_pct, 6),
        "movement_passed": bool(movement_passed),
        "proof_required": bool(proof_required),
        "proof_passed": bool(proof_passed),
        "levels_proof": level_proof,
        "l2_proof": l2_proof,
        "no_go_blocked": bool(no_go_blocked),
        "activation_eligible": bool(activation_eligible),
        "atr_1m_pct": round(atr_1m_pct, 6),
        "atr_5m_pct": round(atr_5m_pct, 6),
        "regime_5m": regime_5m,
        "l2_bias_5m": round(l2_bias_5m, 6),
        "spread_bps": round(float(spread_bps), 6) if spread_bps is not None else None,
        "anti_spike_bars_remaining": int(
            max(0, getattr(pos, "break_even_anti_spike_bars_remaining", 0) or 0)
        ),
        "anti_spike_consecutive_hits": int(
            max(0, getattr(pos, "break_even_anti_spike_consecutive_hits", 0) or 0)
        ),
        "anti_spike_consecutive_hits_required": int(
            max(1, getattr(pos, "break_even_anti_spike_consecutive_hits_required", 2) or 2)
        ),
        "anti_spike_require_close_beyond": bool(
            getattr(pos, "break_even_anti_spike_require_close_beyond", True)
        ),
        "partial_tp_filled": bool(getattr(pos, "partial_tp_filled", False)),
        "partial_tp_size": float(getattr(pos, "partial_tp_size", 0.0) or 0.0),
        "partial_realized_r": round(float(getattr(pos, "partial_realized_r", 0.0) or 0.0), 6),
        "runtime_formulas": {
            "break_even_movement": dict(movement_formula or {}),
            "break_even_proof": dict(proof_formula or {}),
            "break_even_activation": dict(activation_formula or {}),
            "break_even_trailing_handoff": dict(trailing_handoff_formula or {}),
        },
        "trailing_can_update": bool(trailing_can_update),
    }
    if move_payload:
        diagnostics["computed_break_even"] = move_payload
    elif last_computed_break_even:
        diagnostics["computed_break_even"] = dict(last_computed_break_even)

    return diagnostics


def build_force_move_snapshot(*, pos: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "active": bool(getattr(pos, "break_even_stop_active", False)),
        "state": str(getattr(pos, "break_even_state", "idle") or "idle"),
        "arm_bar_index": getattr(pos, "break_even_arm_bar_index", None),
        "move_bar_index": getattr(pos, "break_even_move_bar_index", None),
        "activation_reason": str(getattr(pos, "break_even_activation_reason", "") or ""),
        "break_even_move_reason": str(getattr(pos, "break_even_move_reason", "") or ""),
        "break_even_costs_pct": float(getattr(pos, "break_even_costs_pct", 0.0) or 0.0),
        "break_even_buffer_pct": float(getattr(pos, "break_even_buffer_pct", 0.0) or 0.0),
        "stop_loss": _to_float(getattr(pos, "stop_loss", None), 0.0),
        "entry_price": _to_float(getattr(pos, "entry_price", None), 0.0),
        "computed_break_even": payload,
        "anti_spike_bars_remaining": int(
            max(0, getattr(pos, "break_even_anti_spike_bars_remaining", 0) or 0)
        ),
        "anti_spike_consecutive_hits_required": int(
            max(1, getattr(pos, "break_even_anti_spike_consecutive_hits_required", 2) or 2)
        ),
        "anti_spike_require_close_beyond": bool(
            getattr(pos, "break_even_anti_spike_require_close_beyond", True)
        ),
        "partial_tp_filled": bool(getattr(pos, "partial_tp_filled", False)),
        "partial_tp_size": float(getattr(pos, "partial_tp_size", 0.0) or 0.0),
        "partial_realized_r": round(float(getattr(pos, "partial_realized_r", 0.0) or 0.0), 6),
    }


__all__ = ["build_force_move_snapshot", "build_update_diagnostics"]
