from __future__ import annotations

from typing import Any, Dict

from .shared import to_float as _to_float
from ..runtime_exit_formulas import evaluate_runtime_exit_formula


def apply_trailing_handoff(
    session: Any,
    pos: Any,
    *,
    side: str,
    formula_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    try:
        from ..strategies.base_strategy import Regime
    except Exception:  # pragma: no cover - fallback for legacy import layouts
        from strategies.base_strategy import Regime

    trailing_updated = False
    skip_trailing_for_choppy = bool(
        session.detected_regime == Regime.CHOPPY and not bool(session.trailing_enabled_in_choppy)
    )
    trailing_can_update = bool(getattr(pos, "trailing_activation_pnl_met", False))
    try:
        regime_value = str(getattr(getattr(session, "detected_regime", None), "value", "") or "")
    except Exception:
        regime_value = ""
    formula_ctx = {
        "side": str(side or ""),
        "regime": regime_value,
        "regime_5m": str((formula_context or {}).get("regime_5m", "") or ""),
        "bars_held_count": int(max(0, (formula_context or {}).get("bars_held_count", 0) or 0)),
        "entry_price": float((formula_context or {}).get("entry_price", 0.0) or 0.0),
        "mfe_pct": float((formula_context or {}).get("mfe_pct", 0.0) or 0.0),
        "mfe_r": float((formula_context or {}).get("mfe_r", 0.0) or 0.0),
        "trailing_stop_pct": float(_to_float(getattr(session, "trailing_stop_pct", 0.8), 0.8)),
        "trailing_stop_active": bool(getattr(pos, "trailing_stop_active", False)),
        "trailing_stop_price": float(_to_float(getattr(pos, "trailing_stop_price", None), 0.0)),
        "trailing_activation_pnl_met": bool(getattr(pos, "trailing_activation_pnl_met", False)),
        "break_even_stop_active": bool(getattr(pos, "break_even_stop_active", False)),
        "break_even_state": str(getattr(pos, "break_even_state", "idle") or "idle"),
        "skip_trailing_for_choppy": bool(skip_trailing_for_choppy),
        "trailing_enabled_in_choppy": bool(getattr(session, "trailing_enabled_in_choppy", False)),
        "highest_price": float(_to_float(getattr(pos, "highest_price", None), 0.0)),
        "lowest_price": float(_to_float(getattr(pos, "lowest_price", None), 0.0)),
        "stop_loss": float(_to_float(getattr(pos, "stop_loss", None), 0.0)),
        "trailing_handoff_base": bool(trailing_can_update),
    }
    trailing_handoff_formula = evaluate_runtime_exit_formula(
        session=session,
        hook="break_even_trailing_handoff",
        context=formula_ctx,
        default_passed=trailing_can_update,
    )
    trailing_can_update = bool(trailing_handoff_formula.get("passed", trailing_can_update))

    if not skip_trailing_for_choppy and bool(getattr(pos, "trailing_stop_active", False)) and trailing_can_update:
        trailing_pct = max(0.0, _to_float(getattr(session, "trailing_stop_pct", 0.8), 0.8))
        if side == "long":
            new_stop = _to_float(getattr(pos, "highest_price", None), 0.0) * (1.0 - trailing_pct / 100.0)
            if new_stop > _to_float(getattr(pos, "trailing_stop_price", None), 0.0):
                pos.trailing_stop_price = new_stop
                trailing_updated = True
            if (
                bool(getattr(pos, "break_even_stop_active", False))
                and _to_float(getattr(pos, "trailing_stop_price", None), 0.0)
                >= _to_float(getattr(pos, "stop_loss", None), 0.0)
            ):
                pos.break_even_state = "handoff"
        else:
            lowest_price = _to_float(getattr(pos, "lowest_price", None), 0.0)
            if lowest_price > 0.0:
                new_stop = lowest_price * (1.0 + trailing_pct / 100.0)
                current_trailing = _to_float(getattr(pos, "trailing_stop_price", None), 0.0)
                if current_trailing <= 0.0 or new_stop < current_trailing:
                    pos.trailing_stop_price = new_stop
                    trailing_updated = True
            if (
                bool(getattr(pos, "break_even_stop_active", False))
                and _to_float(getattr(pos, "trailing_stop_price", None), 0.0) > 0.0
                and _to_float(getattr(pos, "trailing_stop_price", None), 0.0)
                <= _to_float(getattr(pos, "stop_loss", None), 0.0)
            ):
                pos.break_even_state = "handoff"

    return {
        "trailing_updated": bool(trailing_updated),
        "skip_trailing_for_choppy": bool(skip_trailing_for_choppy),
        "trailing_can_update": bool(trailing_can_update),
        "trailing_handoff_formula": trailing_handoff_formula,
    }


__all__ = ["apply_trailing_handoff"]
