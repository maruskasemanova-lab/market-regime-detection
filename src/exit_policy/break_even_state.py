from __future__ import annotations

from typing import Any, Dict, Optional

from .break_even_config import cfg_bool, cfg_float, cfg_int
from .break_even_market import aggregate_recent_5m_bars, intrabar_spread_bps, rolling_atr_pct
from .shared import to_float as _to_float


def initial_risk_snapshot(pos: Any) -> Dict[str, Optional[float]]:
    entry_price = _to_float(getattr(pos, "entry_price", None), 0.0)
    side = str(getattr(pos, "side", "")).strip().lower()
    initial_stop = _to_float(getattr(pos, "initial_stop_loss", None), 0.0)
    if initial_stop <= 0.0:
        initial_stop = _to_float(getattr(pos, "stop_loss", None), 0.0)

    risk_abs = 0.0
    if entry_price > 0.0 and initial_stop > 0.0:
        if side == "long":
            risk_abs = max(0.0, entry_price - initial_stop)
        else:
            risk_abs = max(0.0, initial_stop - entry_price)
    risk_pct = ((risk_abs / entry_price) * 100.0) if entry_price > 0.0 and risk_abs > 0.0 else 0.0
    return {
        "entry_price": entry_price if entry_price > 0.0 else None,
        "initial_stop_loss": initial_stop if initial_stop > 0.0 else None,
        "risk_abs": risk_abs if risk_abs > 0.0 else 0.0,
        "risk_pct": risk_pct if risk_pct > 0.0 else 0.0,
    }


def break_even_buffer_snapshot(
    *,
    session: Any,
    entry_price: float,
    atr_1m_pct: float,
    atr_5m_pct: float,
) -> Dict[str, float]:
    base_buffer_pct = max(0.0, cfg_float(session, "break_even_buffer_pct", 0.05))
    min_buffer_pct = max(0.0, cfg_float(session, "break_even_min_buffer_pct", 0.05))
    atr_buffer_k = max(0.0, cfg_float(session, "break_even_atr_buffer_k", 0.10))
    atr5_buffer_k = max(0.0, cfg_float(session, "break_even_5m_atr_buffer_k", 0.10))
    atr_buffer_pct = max(0.0, atr_buffer_k * max(0.0, atr_1m_pct))
    atr5_buffer_pct = max(0.0, atr5_buffer_k * max(0.0, atr_5m_pct))
    tick_size = max(0.0, cfg_float(session, "break_even_tick_size", 0.01))
    min_ticks = max(0, cfg_int(session, "break_even_min_tick_buffer", 1))
    tick_buffer_pct = 0.0
    if entry_price > 0.0 and tick_size > 0.0 and min_ticks > 0:
        tick_buffer_pct = ((tick_size * float(min_ticks)) / entry_price) * 100.0

    selected_pct = max(base_buffer_pct, min_buffer_pct, atr_buffer_pct, atr5_buffer_pct, tick_buffer_pct)
    return {
        "base_buffer_pct": base_buffer_pct,
        "min_buffer_pct": min_buffer_pct,
        "atr_buffer_pct": atr_buffer_pct,
        "atr5_buffer_pct": atr5_buffer_pct,
        "tick_buffer_pct": tick_buffer_pct,
        "selected_buffer_pct": selected_pct,
        "selected_buffer_abs": (entry_price * selected_pct / 100.0) if entry_price > 0.0 else 0.0,
    }


def compute_break_even_stop(
    *,
    session: Any,
    pos: Any,
    atr_1m_pct: float,
    atr_5m_pct: float,
    spread_bps: Optional[float],
) -> Dict[str, Any]:
    entry_price = _to_float(getattr(pos, "entry_price", None), 0.0)
    if entry_price <= 0.0:
        return {
            "valid": False,
            "reason": "invalid_entry_price",
        }

    side = str(getattr(pos, "side", "long")).strip().lower()
    base_costs_pct = max(0.0, cfg_float(session, "break_even_costs_pct", 0.03))
    spread_component_pct = max(0.0, float(spread_bps or 0.0)) / 200.0
    total_costs_pct = base_costs_pct + spread_component_pct
    buffer_info = break_even_buffer_snapshot(
        session=session,
        entry_price=entry_price,
        atr_1m_pct=atr_1m_pct,
        atr_5m_pct=atr_5m_pct,
    )
    buffer_abs = float(buffer_info.get("selected_buffer_abs", 0.0) or 0.0)
    if side == "long":
        stop_level = (entry_price * (1.0 + (total_costs_pct / 100.0))) + buffer_abs
    else:
        stop_level = (entry_price * (1.0 - (total_costs_pct / 100.0))) - buffer_abs

    return {
        "valid": True,
        "entry_price": entry_price,
        "side": side,
        "stop_level": stop_level,
        "base_costs_pct": base_costs_pct,
        "spread_component_pct": spread_component_pct,
        "total_costs_pct": total_costs_pct,
        "spread_bps": spread_bps,
        "buffer": buffer_info,
    }


def break_even_move_reason_from_activation_reason(activation_reason: str) -> str:
    tokens = {
        token.strip()
        for token in str(activation_reason or "").split("|")
        if token and token.strip()
    }
    if "partial_take_profit_protect" in tokens:
        return "partial_protect"
    if tokens & {"movement_threshold", "levels_proof", "l2_proof", "close_confirmed"}:
        return "proof"
    return "time_risk_off"


def sync_break_even_snapshot(pos: Any, snapshot: Dict[str, Any]) -> None:
    if not isinstance(snapshot, dict):
        snapshot = {}
    pos.break_even_last_update = dict(snapshot)
    signal_md = getattr(pos, "signal_metadata", None)
    if isinstance(signal_md, dict):
        signal_md["break_even"] = dict(snapshot)


def move_position_to_break_even(
    *,
    session: Any,
    pos: Any,
    bar: Optional[Any],
    current_bar_index: int,
    activation_reason: str,
) -> Dict[str, Any]:
    bars = getattr(session, "bars", [])
    atr_1m_pct = rolling_atr_pct(bars, window=14) if isinstance(bars, list) else 0.0
    agg_5m = aggregate_recent_5m_bars(bars) if isinstance(bars, list) else []
    atr_5m_pct = rolling_atr_pct(agg_5m, window=14) if agg_5m else 0.0
    spread_bps = intrabar_spread_bps(bar) if bar is not None else None
    stop_payload = compute_break_even_stop(
        session=session,
        pos=pos,
        atr_1m_pct=atr_1m_pct,
        atr_5m_pct=atr_5m_pct,
        spread_bps=spread_bps,
    )
    if not stop_payload.get("valid", False):
        return {"applied": False, **stop_payload}

    side = str(getattr(pos, "side", "long")).strip().lower()
    current_stop = _to_float(getattr(pos, "stop_loss", None), 0.0)
    target_stop = _to_float(stop_payload.get("stop_level"), 0.0)
    if target_stop <= 0.0:
        return {"applied": False, **stop_payload}

    applied = False
    if side == "long":
        if current_stop <= 0.0:
            pos.stop_loss = target_stop
            applied = True
        elif target_stop > current_stop:
            pos.stop_loss = target_stop
            applied = True
    else:
        if current_stop <= 0.0:
            pos.stop_loss = target_stop
            applied = True
        elif target_stop < current_stop:
            pos.stop_loss = target_stop
            applied = True

    move_reason = break_even_move_reason_from_activation_reason(activation_reason)
    pos.break_even_stop_active = True
    pos.break_even_state = "moved"
    if getattr(pos, "break_even_arm_bar_index", None) is None:
        pos.break_even_arm_bar_index = current_bar_index
    if getattr(pos, "break_even_move_bar_index", None) is None:
        pos.break_even_move_bar_index = current_bar_index
    pos.break_even_activation_reason = str(activation_reason or "")
    pos.break_even_move_reason = str(move_reason)
    pos.break_even_costs_pct = float(stop_payload.get("total_costs_pct", 0.0) or 0.0)
    buffer_payload = stop_payload.get("buffer")
    if isinstance(buffer_payload, dict):
        pos.break_even_buffer_pct = float(buffer_payload.get("selected_buffer_pct", 0.0) or 0.0)
    else:
        pos.break_even_buffer_pct = 0.0
    pos.break_even_anti_spike_bars_remaining = max(
        0,
        cfg_int(session, "break_even_anti_spike_bars", 1),
    )
    pos.break_even_anti_spike_consecutive_hits = 0
    pos.break_even_anti_spike_consecutive_hits_required = max(
        1,
        cfg_int(session, "break_even_anti_spike_hits_required", 2),
    )
    pos.break_even_anti_spike_require_close_beyond = cfg_bool(
        session,
        "break_even_anti_spike_require_close_beyond",
        True,
    )

    return {
        "applied": applied,
        "updated_stop_loss": _to_float(getattr(pos, "stop_loss", None), 0.0),
        "atr_1m_pct": atr_1m_pct,
        "atr_5m_pct": atr_5m_pct,
        **stop_payload,
    }


__all__ = [
    "initial_risk_snapshot",
    "move_position_to_break_even",
    "sync_break_even_snapshot",
]
