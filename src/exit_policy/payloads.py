from __future__ import annotations

from typing import Any, Dict, List, Optional

from .shared import to_float as _to_float


def build_position_closed_payload(
    trade, exit_reason: str, bars_held: Optional[int] = None,
) -> Dict[str, Any]:
    """Serialize a closed trade into a frontend/debug-friendly payload."""
    bars_held_count = 0
    try:
        if bars_held is not None:
            bars_held_count = max(0, int(bars_held))
    except (TypeError, ValueError):
        bars_held_count = 0
    position_notional_usd = (
        float(trade.entry_price * trade.size)
        if trade.entry_price and trade.size
        else None
    )
    cost_usd = float(trade.total_costs)
    cost_pct = (
        (cost_usd / position_notional_usd) * 100.0
        if position_notional_usd and position_notional_usd > 0
        else None
    )
    gross_pnl_dollars = (
        float((trade.gross_pnl_pct / 100.0) * position_notional_usd)
        if position_notional_usd and position_notional_usd > 0
        else 0.0
    )
    flow_snapshot = (
        dict(trade.flow_snapshot)
        if isinstance(getattr(trade, "flow_snapshot", None), dict)
        else {}
    )
    signal_metadata = (
        dict(trade.signal_metadata)
        if isinstance(getattr(trade, "signal_metadata", None), dict)
        else {}
    )
    strategy_key = str(getattr(trade, "strategy", "") or "").strip().lower()
    has_flow_snapshot = any(
        key in flow_snapshot
        for key in (
            "signed_aggression",
            "directional_consistency",
            "imbalance_avg",
            "book_pressure_avg",
            "book_pressure_trend",
            "delta_price_divergence",
        )
    )
    flow_strategy = bool(
        has_flow_snapshot
        or ("flow" in strategy_key)
        or bool(flow_snapshot.get("l2_confirmation_passed", False))
    )
    book_pressure_avg = _to_float(flow_snapshot.get("book_pressure_avg"), None)
    book_pressure_trend = _to_float(flow_snapshot.get("book_pressure_trend"), None)
    signed_aggression = _to_float(flow_snapshot.get("signed_aggression"), None)
    book_pressure_confirmed: Optional[bool] = None
    if book_pressure_avg is not None:
        if str(trade.side).lower() == "long":
            book_pressure_confirmed = bool(book_pressure_avg >= 0.0)
        elif str(trade.side).lower() == "short":
            book_pressure_confirmed = bool(book_pressure_avg <= 0.0)

    level_context = (
        signal_metadata.get("level_context")
        if isinstance(signal_metadata.get("level_context"), dict)
        else {}
    )
    level_context_stats = (
        level_context.get("stats")
        if isinstance(level_context.get("stats"), dict)
        else {}
    )
    level_context_profile = (
        level_context.get("volume_profile")
        if isinstance(level_context.get("volume_profile"), dict)
        else {}
    )
    level_context_opening = (
        level_context.get("opening_range")
        if isinstance(level_context.get("opening_range"), dict)
        else {}
    )
    level_context_poc_migration = (
        level_context.get("poc_migration")
        if isinstance(level_context.get("poc_migration"), dict)
        else {}
    )
    level_context_reasons = (
        [str(item) for item in level_context.get("reasons", []) if str(item).strip()]
        if isinstance(level_context.get("reasons"), list)
        else []
    )

    risk_controls = (
        signal_metadata.get("risk_controls")
        if isinstance(signal_metadata.get("risk_controls"), dict)
        else {}
    )
    context_risk = (
        signal_metadata.get("context_risk")
        if isinstance(signal_metadata.get("context_risk"), dict)
        else {}
    )
    break_even = (
        signal_metadata.get("break_even")
        if isinstance(signal_metadata.get("break_even"), dict)
        else {}
    )
    effective_stop_loss = _to_float(risk_controls.get("effective_stop_loss"), 0.0)
    strategy_stop_loss = _to_float(risk_controls.get("strategy_stop_loss"), 0.0)
    stop_distance_pct: Optional[float] = None
    if trade.entry_price and effective_stop_loss > 0.0:
        stop_distance_pct = abs(float(trade.entry_price) - effective_stop_loss) / float(
            trade.entry_price
        ) * 100.0

    stop_reason_keys = {"stop_loss", "breakeven_stop", "trailing_stop"}
    normalized_exit_reason = str(exit_reason or "").strip().lower()
    is_stop_exit = bool(
        normalized_exit_reason in stop_reason_keys
        or normalized_exit_reason.endswith("_stop")
        or "stop" in normalized_exit_reason
    )
    is_first_bar_stop_loss = bool(is_stop_exit and bars_held_count <= 1)
    near_tested_levels_count = int(
        _to_float(level_context_stats.get("near_tested_levels_count"), 0.0)
    )
    near_confluence_score = int(
        _to_float(level_context_stats.get("near_confluence_score"), 0.0)
    )
    near_memory_levels_count = int(
        _to_float(level_context_stats.get("near_memory_levels_count"), 0.0)
    )
    value_area_position = str(
        level_context_profile.get("value_area_position", "unknown")
    ).strip().lower()
    poc_on_trade_side = bool(level_context_profile.get("poc_on_trade_side", False))
    opening_range_breakout_direction = str(
        level_context_opening.get("breakout_direction", "") or ""
    ).strip().lower()
    poc_migration_bias = str(
        level_context_poc_migration.get("regime_bias", "unknown") or "unknown"
    ).strip().lower()
    vwap_distance_pct = _to_float(signal_metadata.get("vwap_distance_pct"), None)
    bars_since_vwap = _to_float(signal_metadata.get("bars_since_vwap"), None)

    first_bar_stop_tags: List[str] = []
    if is_first_bar_stop_loss:
        if stop_distance_pct is not None and stop_distance_pct <= 0.25:
            first_bar_stop_tags.append("tight_stop_distance")
        if near_tested_levels_count <= 0:
            first_bar_stop_tags.append("missing_tested_level_confluence")
        if near_confluence_score <= 1:
            first_bar_stop_tags.append("low_confluence_score")
        if value_area_position == "inside":
            first_bar_stop_tags.append("entered_inside_value_area")
        if not poc_on_trade_side:
            first_bar_stop_tags.append("poc_not_on_trade_side")
        if "vwap_blocked_recent_volume_break" in level_context_reasons:
            first_bar_stop_tags.append("recent_break_context")
        if (
            str(trade.side).lower() == "long"
            and signed_aggression is not None
            and float(signed_aggression) < -0.05
        ):
            first_bar_stop_tags.append("adverse_flow_at_entry")
        if (
            str(trade.side).lower() == "short"
            and signed_aggression is not None
            and float(signed_aggression) > 0.05
        ):
            first_bar_stop_tags.append("adverse_flow_at_entry")

    entry_quality_diagnostics: Dict[str, Any] = {
        "strategy_key": strategy_key,
        "is_stop_exit": is_stop_exit,
        "is_first_bar_stop_loss": is_first_bar_stop_loss,
        "bars_held": int(bars_held_count),
        "stop_distance_pct": (
            round(float(stop_distance_pct), 4) if stop_distance_pct is not None else None
        ),
        "vwap_distance_pct": (
            round(float(vwap_distance_pct), 4) if vwap_distance_pct is not None else None
        ),
        "bars_since_vwap": (
            int(round(float(bars_since_vwap))) if bars_since_vwap is not None else None
        ),
        "level_context_passed": bool(level_context.get("passed", True)),
        "level_context_reason": str(level_context.get("reason", "") or ""),
        "level_context_reasons": level_context_reasons,
        "near_tested_levels_count": int(near_tested_levels_count),
        "near_confluence_score": int(near_confluence_score),
        "near_memory_levels_count": int(near_memory_levels_count),
        "value_area_position": value_area_position,
        "poc_on_trade_side": bool(poc_on_trade_side),
        "opening_range_breakout_direction": (
            opening_range_breakout_direction or None
        ),
        "poc_migration_bias": poc_migration_bias,
        "first_bar_stop_tags": list(dict.fromkeys(first_bar_stop_tags)),
        "risk_controls": {
            "stop_loss_mode": str(risk_controls.get("stop_loss_mode", "")),
            "fixed_stop_loss_pct": _to_float(
                risk_controls.get("fixed_stop_loss_pct"), None
            ),
            "strategy_stop_loss": (
                float(strategy_stop_loss) if strategy_stop_loss > 0.0 else None
            ),
            "effective_stop_loss": (
                float(effective_stop_loss) if effective_stop_loss > 0.0 else None
            ),
            "context_risk": dict(context_risk) if context_risk else None,
        },
        "break_even": dict(break_even) if break_even else None,
    }

    return {
        'schema_version': 2,
        'exit_price': trade.exit_price,
        'side': trade.side,
        'exit_reason': exit_reason,
        'pnl_pct': trade.pnl_pct,
        'pnl_dollars': trade.pnl_dollars,
        'pnl_usd': trade.pnl_dollars,
        'strategy': trade.strategy,
        'entry_price': trade.entry_price,
        'entry_time': trade.entry_time.isoformat(),
        'exit_time': trade.exit_time.isoformat(),
        'size': trade.size,
        'bars_held': bars_held_count,
        'signal_bar_index': trade.signal_bar_index,
        'entry_bar_index': trade.entry_bar_index,
        'signal_timestamp': trade.signal_timestamp,
        'signal_price': trade.signal_price,
        'position_notional_usd': position_notional_usd,
        'cost_usd': cost_usd,
        'cost_pct': cost_pct,
        'costs': {
            'slippage': round(trade.slippage, 4),
            'commission': round(trade.commission, 4),
            'reg_fee': round(trade.reg_fee, 4),
            'sec_fee': round(trade.sec_fee, 6),
            'finra_fee': round(trade.finra_fee, 6),
            'market_impact': round(trade.market_impact, 4),
            'total': round(trade.total_costs, 4),
        },
        'gross_pnl_pct': trade.gross_pnl_pct,
        'gross_pnl_dollars': gross_pnl_dollars,
        'flow_strategy': flow_strategy,
        'book_pressure_confirmed': book_pressure_confirmed,
        'book_pressure_avg': book_pressure_avg,
        'book_pressure_trend': book_pressure_trend,
        'signed_aggression': signed_aggression,
        'flow_snapshot': flow_snapshot,
        'level_context': level_context if level_context else None,
        'entry_quality_diagnostics': entry_quality_diagnostics,
        'signal_metadata': signal_metadata,
        'break_even': dict(break_even) if break_even else None,
    }


__all__ = ["build_position_closed_payload"]
