from __future__ import annotations

import math
from typing import Any, Dict, List

from .break_even_config import cfg_bool, cfg_float, cfg_int, side_direction
from .break_even_market import (
    aggregate_recent_5m_bars,
    infer_5m_regime,
    intrabar_spread_bps,
    rolling_atr_pct,
    rolling_l2_bias_5m,
)
from .break_even_proofs import l2_proof_snapshot, level_proof_snapshot, resolve_intraday_levels_snapshot
from .break_even_state import initial_risk_snapshot, move_position_to_break_even
from .shared import bars_held as _bars_held
from .shared import to_float as _to_float
from ..runtime_exit_formulas import evaluate_runtime_exit_formula


def evaluate_break_even_activation_context(
    session: Any,
    pos: Any,
    bar: Any,
    *,
    current_bar_index: int,
) -> Dict[str, Any]:
    bars_held_count = _bars_held(pos, current_bar_index)
    min_hold_for_break_even = max(1, cfg_int(session, "break_even_min_hold_bars", 3))

    sig_meta = (
        dict(getattr(pos, "signal_metadata", {}))
        if isinstance(getattr(pos, "signal_metadata", None), dict)
        else {}
    )
    weak_l2_override = sig_meta.get("weak_l2_break_even_override")
    if isinstance(weak_l2_override, dict) and "break_even_min_hold_bars" in weak_l2_override:
        min_hold_for_break_even = max(1, int(weak_l2_override["break_even_min_hold_bars"]))

    side = str(getattr(pos, "side", "long")).strip().lower()
    entry_price = _to_float(getattr(pos, "entry_price", None), 0.0)
    last_break_even_snapshot = (
        dict(getattr(pos, "break_even_last_update", {}))
        if isinstance(getattr(pos, "break_even_last_update", None), dict)
        else {}
    )
    last_computed_break_even = (
        dict(last_break_even_snapshot.get("computed_break_even", {}))
        if isinstance(last_break_even_snapshot.get("computed_break_even"), dict)
        else {}
    )

    bar_high = _to_float(getattr(bar, "high", None), _to_float(getattr(bar, "close", None), 0.0))
    bar_low = _to_float(getattr(bar, "low", None), _to_float(getattr(bar, "close", None), 0.0))
    bar_close = _to_float(getattr(bar, "close", None), 0.0)

    if side == "long":
        if _to_float(getattr(pos, "highest_price", None), 0.0) <= 0.0:
            pos.highest_price = entry_price
        if bar_high > _to_float(getattr(pos, "highest_price", None), 0.0):
            pos.highest_price = bar_high
        mfe_abs = max(0.0, _to_float(getattr(pos, "highest_price", None), 0.0) - entry_price)
    else:
        lowest_px = _to_float(getattr(pos, "lowest_price", None), float("inf"))
        if not math.isfinite(lowest_px) or lowest_px <= 0.0:
            pos.lowest_price = entry_price
        if bar_low > 0.0 and bar_low < _to_float(getattr(pos, "lowest_price", None), float("inf")):
            pos.lowest_price = bar_low
        mfe_abs = max(0.0, entry_price - _to_float(getattr(pos, "lowest_price", None), entry_price))
    mfe_pct = ((mfe_abs / entry_price) * 100.0) if entry_price > 0.0 else 0.0

    initial_risk = initial_risk_snapshot(pos)
    risk_abs = float(initial_risk.get("risk_abs") or 0.0)
    risk_pct = float(initial_risk.get("risk_pct") or 0.0)
    mfe_r = (mfe_abs / risk_abs) if risk_abs > 0.0 else 0.0

    bars = getattr(session, "bars", [])
    atr_1m_pct = rolling_atr_pct(bars, window=14) if isinstance(bars, list) else 0.0
    agg_5m = aggregate_recent_5m_bars(bars) if isinstance(bars, list) else []
    atr_5m_pct = rolling_atr_pct(agg_5m, window=14) if agg_5m else 0.0
    regime_5m = infer_5m_regime(agg_5m)
    l2_bias_5m = rolling_l2_bias_5m(agg_5m) if agg_5m else 0.0

    spread_bps = intrabar_spread_bps(bar)
    levels_snapshot = resolve_intraday_levels_snapshot(session)
    level_proof = level_proof_snapshot(
        session=session,
        side=side,
        close_price=bar_close if bar_close > 0.0 else entry_price,
        levels_snapshot=levels_snapshot,
    )
    l2_proof = l2_proof_snapshot(
        session=session,
        side=side,
        bar=bar,
        spread_bps=spread_bps,
    )

    use_levels = cfg_bool(session, "break_even_activation_use_levels", True)
    use_l2 = cfg_bool(session, "break_even_activation_use_l2", True)
    proof_required = bool(use_levels or use_l2)
    proof_candidates: List[bool] = []
    if use_levels:
        proof_candidates.append(bool(level_proof.get("passed", False)))
    else:
        level_proof["enabled"] = False
    if use_l2:
        proof_candidates.append(bool(l2_proof.get("passed", False)))
    else:
        l2_proof["enabled"] = False
    proof_passed = bool(any(proof_candidates)) if proof_required else True
    no_go_blocked = bool(level_proof.get("no_go_blocked", False))

    base_activation_mfe_pct = max(
        0.0,
        cfg_float(
            session,
            "break_even_activation_min_mfe_pct",
            max(0.20, cfg_float(session, "trailing_activation_pct", 0.15)),
        ),
    )
    atr_activation_mfe_pct = max(
        0.0,
        cfg_float(session, "break_even_5m_mfe_atr_factor", 0.15) * max(0.0, atr_5m_pct),
    )
    required_mfe_pct = max(base_activation_mfe_pct, atr_activation_mfe_pct)

    base_r = max(0.0, cfg_float(session, "break_even_activation_min_r", 0.60))
    if regime_5m == "trending":
        required_r = max(base_r, cfg_float(session, "break_even_activation_min_r_trending_5m", 0.90))
    elif regime_5m == "choppy":
        required_r = max(0.0, cfg_float(session, "break_even_activation_min_r_choppy_5m", 0.60))
    else:
        required_r = base_r

    bias_threshold = max(0.0, cfg_float(session, "break_even_5m_l2_bias_threshold", 0.10))
    directional_bias = l2_bias_5m * side_direction(side)
    if directional_bias < -bias_threshold:
        tighten_factor = cfg_float(session, "break_even_5m_l2_bias_tighten_factor", 0.85)
        required_r *= max(0.50, min(1.0, tighten_factor))

    required_r_mfe_pct = required_r * risk_pct if risk_pct > 0.0 else 0.0
    required_mfe_pct = max(required_mfe_pct, required_r_mfe_pct)

    bars_held_meets_min_hold = bool(bars_held_count >= min_hold_for_break_even)
    movement_passed = bool(mfe_pct >= required_mfe_pct and bars_held_meets_min_hold)
    levels_nearest = (
        dict(level_proof.get("nearest_level", {}))
        if isinstance(level_proof.get("nearest_level"), dict)
        else {}
    )
    l2_metrics = (
        dict(l2_proof.get("metrics", {}))
        if isinstance(l2_proof.get("metrics"), dict)
        else {}
    )
    break_even_active = bool(getattr(pos, "break_even_stop_active", False))
    trailing_stop_active = bool(getattr(pos, "trailing_stop_active", False))
    partial_tp_filled = bool(getattr(pos, "partial_tp_filled", False))
    formula_context = {
        "side": side,
        "bars_held_count": int(bars_held_count),
        "min_hold_for_break_even": int(min_hold_for_break_even),
        "bars_held_meets_min_hold": bool(bars_held_meets_min_hold),
        "entry_price": float(entry_price),
        "mfe_pct": float(mfe_pct),
        "mfe_r": float(mfe_r),
        "risk_abs": float(risk_abs),
        "risk_pct": float(risk_pct),
        "atr_1m_pct": float(atr_1m_pct),
        "atr_5m_pct": float(atr_5m_pct),
        "regime_5m": str(regime_5m),
        "l2_bias_5m": float(l2_bias_5m),
        "spread_bps": float(spread_bps) if spread_bps is not None else 0.0,
        "required_mfe_pct": float(required_mfe_pct),
        "required_r": float(required_r),
        "required_r_mfe_pct": float(required_r_mfe_pct),
        "base_activation_mfe_pct": float(base_activation_mfe_pct),
        "atr_activation_mfe_pct": float(atr_activation_mfe_pct),
        "base_r": float(base_r),
        "bias_threshold": float(bias_threshold),
        "directional_bias": float(directional_bias),
        "use_levels": bool(use_levels),
        "use_l2": bool(use_l2),
        "proof_required": bool(proof_required),
        "proof_passed": bool(proof_passed),
        "no_go_blocked": bool(no_go_blocked),
        "movement_passed": bool(movement_passed),
        "levels_proof_enabled": bool(level_proof.get("enabled", False)),
        "levels_proof_passed": bool(level_proof.get("passed", False)),
        "levels_no_go_blocked": bool(level_proof.get("no_go_blocked", False)),
        "levels_nearest_distance_pct": float(levels_nearest.get("distance_pct") or 0.0),
        "levels_confluence": int(max(0.0, _to_float(level_proof.get("confluence"), 0.0))),
        "levels_tests": int(max(0.0, _to_float(level_proof.get("tests"), 0.0))),
        "l2_proof_enabled": bool(l2_proof.get("enabled", False)),
        "l2_proof_passed": bool(l2_proof.get("passed", False)),
        "l2_signed_aggression": float(l2_metrics.get("signed_aggression") or 0.0),
        "l2_imbalance": float(l2_metrics.get("imbalance") or 0.0),
        "l2_book_pressure": float(l2_metrics.get("book_pressure") or 0.0),
        "l2_spread_bps": float(l2_metrics.get("spread_bps") or 0.0),
        "break_even_active": bool(break_even_active),
        "trailing_stop_active": bool(trailing_stop_active),
        "partial_tp_filled": bool(partial_tp_filled),
    }
    movement_formula = evaluate_runtime_exit_formula(
        session=session,
        hook="break_even_movement",
        context=formula_context,
        default_passed=movement_passed,
    )
    movement_passed = bool(movement_formula.get("passed", movement_passed))
    formula_context["movement_passed"] = bool(movement_passed)

    proof_formula = evaluate_runtime_exit_formula(
        session=session,
        hook="break_even_proof",
        context=formula_context,
        default_passed=proof_passed,
    )
    proof_passed = bool(proof_formula.get("passed", proof_passed))
    formula_context["proof_passed"] = bool(proof_passed)

    activation_formula_default = bool(movement_passed and proof_passed and not no_go_blocked)
    activation_formula = evaluate_runtime_exit_formula(
        session=session,
        hook="break_even_activation",
        context=formula_context,
        default_passed=activation_formula_default,
    )
    activation_eligible = bool(activation_formula.get("passed", activation_formula_default))

    if movement_passed:
        pos.trailing_activation_pnl_met = True

    return {
        "bars_held_count": bars_held_count,
        "entry_price": entry_price,
        "last_computed_break_even": last_computed_break_even,
        "mfe_pct": mfe_pct,
        "mfe_r": mfe_r,
        "risk_abs": risk_abs,
        "risk_pct": risk_pct,
        "atr_1m_pct": atr_1m_pct,
        "atr_5m_pct": atr_5m_pct,
        "regime_5m": regime_5m,
        "l2_bias_5m": l2_bias_5m,
        "spread_bps": spread_bps,
        "level_proof": level_proof,
        "l2_proof": l2_proof,
        "proof_required": proof_required,
        "proof_passed": proof_passed,
        "no_go_blocked": no_go_blocked,
        "required_mfe_pct": required_mfe_pct,
        "required_r": required_r,
        "required_r_mfe_pct": required_r_mfe_pct,
        "base_activation_mfe_pct": base_activation_mfe_pct,
        "atr_activation_mfe_pct": atr_activation_mfe_pct,
        "base_r": base_r,
        "bars_held_meets_min_hold": bars_held_meets_min_hold,
        "min_hold_for_break_even": min_hold_for_break_even,
        "use_levels": use_levels,
        "use_l2": use_l2,
        "activation_eligible": activation_eligible,
        "movement_formula": movement_formula,
        "proof_formula": proof_formula,
        "activation_formula": activation_formula,
        "movement_passed": movement_passed,
        "side": side,
    }


def activate_break_even_if_eligible(
    session: Any,
    pos: Any,
    bar: Any,
    *,
    current_bar_index: int,
    movement_passed: bool,
    proof_passed: bool,
    no_go_blocked: bool,
    activation_eligible: bool,
    activation_formula: Dict[str, Any],
    level_proof: Dict[str, Any],
) -> Dict[str, Any]:
    armed_this_bar = False
    moved_this_bar = False
    move_payload: Dict[str, Any] = {}
    activation_reason_tokens: List[str] = []

    if not bool(getattr(pos, "break_even_stop_active", False)):
        if movement_passed:
            activation_reason_tokens.append("movement_threshold")
        if proof_passed:
            activation_reason_tokens.append(
                "levels_proof" if bool(level_proof.get("passed", False)) else "l2_proof"
            )
        if no_go_blocked:
            activation_reason_tokens.append("blocked_no_go")
        if bool(activation_formula.get("enabled", False)):
            activation_reason_tokens.append(
                "formula_pass" if bool(activation_formula.get("passed", False)) else "formula_block"
            )

        if activation_eligible:
            if str(getattr(pos, "break_even_state", "idle") or "idle") == "idle":
                pos.break_even_state = "armed"
                pos.break_even_arm_bar_index = current_bar_index
                armed_this_bar = True
            move_payload = move_position_to_break_even(
                session=session,
                pos=pos,
                bar=bar,
                current_bar_index=current_bar_index,
                activation_reason="|".join(activation_reason_tokens) or "close_confirmed",
            )
            moved_this_bar = bool(move_payload.get("applied", False))

    return {
        "armed_this_bar": armed_this_bar,
        "moved_this_bar": moved_this_bar,
        "move_payload": move_payload,
    }


def advance_break_even_lock_state(pos: Any, *, current_bar_index: int) -> None:
    if not bool(getattr(pos, "break_even_stop_active", False)):
        return
    move_bar_index = getattr(pos, "break_even_move_bar_index", None)
    if move_bar_index is not None and current_bar_index > int(move_bar_index):
        if str(getattr(pos, "break_even_state", "idle") or "idle") in {"armed", "moved"}:
            pos.break_even_state = "locked"
        remaining = int(max(0, getattr(pos, "break_even_anti_spike_bars_remaining", 0) or 0))
        if remaining > 0:
            pos.break_even_anti_spike_bars_remaining = max(0, remaining - 1)
            if pos.break_even_anti_spike_bars_remaining <= 0:
                pos.break_even_anti_spike_consecutive_hits = 0


__all__ = [
    "activate_break_even_if_eligible",
    "advance_break_even_lock_state",
    "evaluate_break_even_activation_context",
]
