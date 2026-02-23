from __future__ import annotations

from typing import Any, Dict, Optional

from .break_even_activation import (
    activate_break_even_if_eligible,
    advance_break_even_lock_state,
    evaluate_break_even_activation_context,
)
from .break_even_diagnostics import build_force_move_snapshot, build_update_diagnostics
from .break_even_state import move_position_to_break_even, sync_break_even_snapshot
from .break_even_trailing import apply_trailing_handoff


def update_trailing_from_close(session, pos, bar) -> Dict[str, Any]:
    """Update break-even/trailing state from the current 1m close.

    Break-even activation is close-confirmed and proof-gated:
      - movement threshold (MFE + min hold),
      - levels OR L2 evidence,
      - optional 5m no-go guard near strong opposite level.

    Returns a diagnostics snapshot suitable for logging/UI tooltips.
    """
    current_bar_index = max(0, len(getattr(session, "bars", []) or []) - 1)

    ctx = evaluate_break_even_activation_context(
        session=session,
        pos=pos,
        bar=bar,
        current_bar_index=current_bar_index,
    )

    activation = activate_break_even_if_eligible(
        session=session,
        pos=pos,
        bar=bar,
        current_bar_index=current_bar_index,
        movement_passed=bool(ctx["movement_passed"]),
        proof_passed=bool(ctx["proof_passed"]),
        no_go_blocked=bool(ctx["no_go_blocked"]),
        activation_eligible=bool(ctx.get("activation_eligible", False)),
        activation_formula=dict(ctx.get("activation_formula", {})),
        level_proof=dict(ctx["level_proof"]),
    )

    advance_break_even_lock_state(
        pos=pos,
        current_bar_index=current_bar_index,
    )

    trailing = apply_trailing_handoff(
        session=session,
        pos=pos,
        side=str(ctx["side"]),
        formula_context={
            "bars_held_count": int(ctx["bars_held_count"]),
            "entry_price": float(ctx["entry_price"]),
            "mfe_pct": float(ctx["mfe_pct"]),
            "mfe_r": float(ctx["mfe_r"]),
            "regime_5m": str(ctx["regime_5m"]),
        },
    )

    diagnostics = build_update_diagnostics(
        pos=pos,
        current_bar_index=current_bar_index,
        bars_held_count=int(ctx["bars_held_count"]),
        entry_price=float(ctx["entry_price"]),
        mfe_pct=float(ctx["mfe_pct"]),
        mfe_r=float(ctx["mfe_r"]),
        risk_abs=float(ctx["risk_abs"]),
        risk_pct=float(ctx["risk_pct"]),
        required_mfe_pct=float(ctx["required_mfe_pct"]),
        required_r=float(ctx["required_r"]),
        required_r_mfe_pct=float(ctx.get("required_r_mfe_pct", 0.0)),
        movement_passed=bool(ctx["movement_passed"]),
        proof_required=bool(ctx["proof_required"]),
        proof_passed=bool(ctx["proof_passed"]),
        level_proof=dict(ctx["level_proof"]),
        l2_proof=dict(ctx["l2_proof"]),
        no_go_blocked=bool(ctx["no_go_blocked"]),
        activation_eligible=bool(ctx.get("activation_eligible", False)),
        atr_1m_pct=float(ctx["atr_1m_pct"]),
        atr_5m_pct=float(ctx["atr_5m_pct"]),
        regime_5m=str(ctx["regime_5m"]),
        l2_bias_5m=float(ctx["l2_bias_5m"]),
        spread_bps=ctx["spread_bps"],
        armed_this_bar=bool(activation["armed_this_bar"]),
        moved_this_bar=bool(activation["moved_this_bar"]),
        trailing_updated=bool(trailing["trailing_updated"]),
        skip_trailing_for_choppy=bool(trailing["skip_trailing_for_choppy"]),
        trailing_can_update=bool(trailing.get("trailing_can_update", False)),
        move_payload=dict(activation["move_payload"]),
        last_computed_break_even=dict(ctx["last_computed_break_even"]),
        movement_formula=dict(ctx.get("movement_formula", {})),
        proof_formula=dict(ctx.get("proof_formula", {})),
        activation_formula=dict(ctx.get("activation_formula", {})),
        trailing_handoff_formula=dict(trailing.get("trailing_handoff_formula", {})),
    )

    sync_break_even_snapshot(pos, diagnostics)
    return diagnostics


def force_move_to_break_even(
    session,
    pos,
    *,
    bar: Optional[Any] = None,
    reason: str = "manual",
) -> Dict[str, Any]:
    current_bar_index = max(0, len(getattr(session, "bars", []) or []) - 1)
    payload = move_position_to_break_even(
        session=session,
        pos=pos,
        bar=bar,
        current_bar_index=current_bar_index,
        activation_reason=reason,
    )
    snapshot = build_force_move_snapshot(pos=pos, payload=payload)
    sync_break_even_snapshot(pos, snapshot)
    return snapshot


__all__ = ["force_move_to_break_even", "update_trailing_from_close"]
