"""Active-position lifecycle helpers for runtime processing."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict

from .context_aware_risk import ContextRiskConfig, apply_context_trailing
from .day_trading_models import BarData, TradingSession
from .trading_config import TradingConfig


def manage_active_position_lifecycle(
    *,
    manager: Any,
    session: TradingSession,
    bar: BarData,
    timestamp: datetime,
    current_bar_index: int,
    result: Dict[str, Any],
    formula_indicators: Callable[[], Dict[str, Any]],
) -> None:
    """Manage exits/trailing/partial logic for an active position."""
    if not session.active_position:
        return

    pos = session.active_position
    ee = manager.exit_engine

    # ── 0. Position Context Monitor: detect regime/flow changes ──
    if session._position_context:
        ctx_flow = manager._calculate_order_flow_metrics(
            session.bars,
            lookback=min(20, len(session.bars)),
        )
        ctx_indicators = manager._calculate_indicators(
            session.bars[-100:] if len(session.bars) >= 100 else session.bars,
            session=session,
        )
        ctx_atr = float(manager._latest_indicator_value(ctx_indicators, "atr") or 0.0)
        context_events = session._position_context.update(
            current_regime=(session.detected_regime.value if session.detected_regime else "MIXED"),
            current_micro=(session.micro_regime or "MIXED"),
            current_flow=ctx_flow,
            current_bar_index=current_bar_index,
            current_atr=ctx_atr,
        )
        if context_events:
            result["context_events"] = [e.to_dict() for e in context_events]

    # ── 0b. Context-aware exit policy: react to regime/flow shifts ──
    if session._context_exit_policy and session._position_context:
        # Feed accumulated events (not just new ones) to the policy.
        all_ctx_events = session._position_context.events
        if all_ctx_events:
            ctx_exit_decision = session._context_exit_policy.evaluate(
                context_events=all_ctx_events,
                session=session,
                pos=pos,
                current_bar_index=current_bar_index,
            )
            if ctx_exit_decision and ctx_exit_decision.should_exit:
                trade = manager._close_position(
                    session,
                    ctx_exit_decision.exit_price,
                    timestamp,
                    ctx_exit_decision.reason,
                    bar_volume=bar.volume,
                )
                session.last_exit_bar_index = current_bar_index
                result["trade_closed"] = trade.to_dict()
                result["action"] = f"position_closed_{ctx_exit_decision.reason}"
                bars_held = len(
                    [
                        b
                        for b in session.bars
                        if b.timestamp >= trade.entry_time and b.timestamp <= trade.exit_time
                    ]
                )
                result["position_closed"] = ee.build_position_closed_payload(
                    trade=trade,
                    exit_reason=ctx_exit_decision.reason,
                    bars_held=bars_held,
                )
                result["context_exit_metrics"] = ctx_exit_decision.metrics

    # ── 0c. Optional custom exit formula gate ──
    if session.active_position:
        strategy_key = manager._canonical_strategy_key(
            session.active_position.strategy_name or ""
        )
        strategy_obj = manager.strategies.get(strategy_key)
        if strategy_obj is not None:
            exit_formula_ctx = manager.strategy_evaluator.build_strategy_formula_context(
                session=session,
                bar=bar,
                indicators=formula_indicators(),
                flow=manager._calculate_order_flow_metrics(
                    session.bars,
                    lookback=min(20, len(session.bars)),
                ),
                current_bar_index=current_bar_index,
                position=session.active_position,
            )
            custom_exit_formula = manager.strategy_evaluator.evaluate_strategy_custom_formula(
                strategy=strategy_obj,
                formula_type="exit",
                context=exit_formula_ctx,
            )
            if custom_exit_formula.get("enabled", False):
                result["custom_exit_formula"] = custom_exit_formula
            if (
                custom_exit_formula.get("enabled", False)
                and custom_exit_formula.get("passed", False)
            ):
                trade = manager._close_position(
                    session,
                    bar.close,
                    timestamp,
                    "custom_formula_exit",
                    bar_volume=bar.volume,
                )
                session.last_exit_bar_index = current_bar_index
                result["trade_closed"] = trade.to_dict()
                result["action"] = "position_closed_custom_formula_exit"
                bars_held = len(
                    [
                        b
                        for b in session.bars
                        if b.timestamp >= trade.entry_time and b.timestamp <= trade.exit_time
                    ]
                )
                result["position_closed"] = ee.build_position_closed_payload(
                    trade=trade,
                    exit_reason="custom_formula_exit",
                    bars_held=bars_held,
                )

    # ── 1. Hard exits (SL/TP) ──
    if session.active_position:
        exit_result = ee.resolve_exit_for_bar(pos, bar)
        if exit_result:
            exit_reason, exit_fill_price = exit_result
            trade = manager._close_position(
                session,
                exit_fill_price,
                timestamp,
                exit_reason,
                bar_volume=bar.volume,
            )
            session.last_exit_bar_index = current_bar_index
            result["trade_closed"] = trade.to_dict()
            result["action"] = f"position_closed_{exit_reason}"
            bars_held = len(
                [
                    b
                    for b in session.bars
                    if b.timestamp >= trade.entry_time and b.timestamp <= trade.exit_time
                ]
            )
            result["position_closed"] = ee.build_position_closed_payload(
                trade=trade,
                exit_reason=exit_reason,
                bars_held=bars_held,
            )

    # ── 2. Partial scale-out ──
    if session.active_position:
        flow_8 = manager._calculate_order_flow_metrics(
            session.bars,
            lookback=min(8, len(session.bars)),
        )
        partial_decision = ee.should_take_partial_profit(
            session,
            session.active_position,
            bar,
            flow_8,
        )
        if partial_decision:
            session.active_position.partial_take_profit_price = partial_decision["partial_price"]
            close_size = max(
                0.0,
                min(
                    session.active_position.size,
                    session.active_position.size * partial_decision["close_fraction"],
                ),
            )
            if close_size > 0:
                pre_partial_size = max(0.0, float(session.active_position.size or 0.0))
                partial_exit_price = float(partial_decision["exit_price"])
                entry_price = float(session.active_position.entry_price or 0.0)
                side = str(session.active_position.side or "long").strip().lower()
                initial_stop = float(session.active_position.initial_stop_loss or 0.0)
                if initial_stop <= 0.0:
                    initial_stop = float(session.active_position.stop_loss or 0.0)
                if side == "long":
                    risk_abs = max(0.0, entry_price - initial_stop)
                    realized_abs = max(0.0, partial_exit_price - entry_price)
                else:
                    risk_abs = max(0.0, initial_stop - entry_price)
                    realized_abs = max(0.0, entry_price - partial_exit_price)
                partial_realized_r = 0.0
                if risk_abs > 0.0 and pre_partial_size > 0.0:
                    partial_fraction = max(
                        0.0,
                        min(1.0, float(close_size) / float(pre_partial_size)),
                    )
                    partial_realized_r = (realized_abs / risk_abs) * partial_fraction
                session.active_position.partial_tp_filled = True
                session.active_position.partial_tp_size = float(close_size)
                session.active_position.partial_realized_r = float(partial_realized_r)
                partial_trade = manager.trade_engine.build_trade_record(
                    session=session,
                    pos=session.active_position,
                    exit_price=partial_exit_price,
                    exit_time=timestamp,
                    reason=partial_decision["reason"],
                    shares=close_size,
                    bar_volume=bar.volume,
                )
                session.active_position.size = max(
                    0.0,
                    session.active_position.size - close_size,
                )
                session.active_position.partial_exit_done = True
                # After partial, move remaining risk to a cost-aware break-even stop.
                # Skip forced BE for flow deterioration partials when configured.
                skip_be = (
                    partial_decision.get("reason") == "partial_take_profit_flow_deterioration"
                    and getattr(session, "partial_flow_deterioration_skip_be", True)
                )
                if not skip_be:
                    be_after_partial = ee.force_move_to_break_even(
                        session=session,
                        pos=session.active_position,
                        bar=bar,
                        reason="partial_take_profit_protect",
                    )
                    if isinstance(be_after_partial, dict):
                        result["break_even"] = be_after_partial
                if session.active_position.size <= 0:
                    session.active_position = None
                result["partial_trade_closed"] = partial_trade.to_dict()
                result["action"] = "partial_take_profit"

    # ── 3. Momentum fail-fast ──
    if session.active_position:
        ticker_cfg = manager.ticker_params.get(session.ticker.upper(), {})
        adaptive_cfg = manager._normalize_adaptive_config(ticker_cfg.get("adaptive"))
        momentum_cfg_all = manager._resolve_momentum_diversification(session, adaptive_cfg)
        strategy_key = manager._canonical_strategy_key(
            session.active_position.strategy_name or ""
        )
        signal_md = (
            session.active_position.signal_metadata
            if isinstance(session.active_position.signal_metadata, dict)
            else {}
        )
        momentum_md = signal_md.get("momentum_diversification") if isinstance(signal_md, dict) else {}
        preferred_sleeve_id = ""
        if isinstance(momentum_md, dict):
            preferred_sleeve_id = str(
                momentum_md.get("selected_sleeve_id") or momentum_md.get("sleeve_id") or ""
            ).strip()
        ff_flow = manager._calculate_order_flow_metrics(
            session.bars,
            lookback=min(8, len(session.bars)),
        )
        selected_cfg, selected_sleeve_id, _ = manager._select_momentum_sleeve(
            momentum_cfg_all,
            strategy_key=strategy_key,
            micro_regime=(session.micro_regime or "MIXED").upper(),
            has_l2_coverage=bool(ff_flow.get("has_l2_coverage", False)),
            preferred_sleeve_id=preferred_sleeve_id,
        )
        apply_to = manager._normalize_strategy_list(
            selected_cfg.get("apply_to_strategies"),
            fallback=list(manager.DEFAULT_MOMENTUM_STRATEGIES),
        )
        should_exit_fail_fast, fail_fast_metrics = ee.should_momentum_fail_fast_exit(
            pos=session.active_position,
            current_bar_index=current_bar_index,
            flow_metrics=ff_flow,
            momentum_config=selected_cfg,
            momentum_sleeve_id=selected_sleeve_id,
            apply_to_strategies=apply_to,
        )
        if fail_fast_metrics.get("enabled"):
            result["momentum_fail_fast"] = fail_fast_metrics
        if should_exit_fail_fast:
            trade = manager._close_position(
                session,
                bar.close,
                timestamp,
                "momentum_fail_fast",
                bar_volume=bar.volume,
            )
            session.last_exit_bar_index = current_bar_index
            result["trade_closed"] = trade.to_dict()
            result["action"] = "position_closed_momentum_fail_fast"
            bars_held = len(
                [
                    b
                    for b in session.bars
                    if b.timestamp >= trade.entry_time and b.timestamp <= trade.exit_time
                ]
            )
            result["position_closed"] = ee.build_position_closed_payload(
                trade=trade,
                exit_reason="momentum_fail_fast",
                bars_held=bars_held,
            )

    # ── 4. Time-based exit ──
    if session.active_position:
        time_flow = manager._calculate_order_flow_metrics(
            session.bars,
            lookback=min(8, len(session.bars)),
        )
        if ee.should_time_exit(session, session.active_position, current_bar_index, time_flow):
            trade = manager._close_position(
                session,
                bar.close,
                timestamp,
                "time_exit",
                bar_volume=bar.volume,
            )
            session.last_exit_bar_index = current_bar_index
            result["trade_closed"] = trade.to_dict()
            result["action"] = "position_closed_time_exit"
            bars_held = len(
                [
                    b
                    for b in session.bars
                    if b.timestamp >= trade.entry_time and b.timestamp <= trade.exit_time
                ]
            )
            result["position_closed"] = ee.build_position_closed_payload(
                trade=trade,
                exit_reason="time_exit",
                bars_held=bars_held,
            )

    # ── 5. Adverse flow exit ──
    if session.active_position:
        adv_flow = manager._calculate_order_flow_metrics(
            session.bars,
            lookback=min(12, len(session.bars)),
        )
        should_exit_adverse, adverse_metrics = ee.should_adverse_flow_exit(
            session,
            session.active_position,
            current_bar_index,
            adv_flow,
        )
        if should_exit_adverse:
            trade = manager._close_position(
                session,
                bar.close,
                timestamp,
                "adverse_flow",
                bar_volume=bar.volume,
            )
            session.last_exit_bar_index = current_bar_index
            result["trade_closed"] = trade.to_dict()
            result["action"] = "position_closed_adverse_flow"
            bars_held = len(
                [
                    b
                    for b in session.bars
                    if b.timestamp >= trade.entry_time and b.timestamp <= trade.exit_time
                ]
            )
            result["position_closed"] = ee.build_position_closed_payload(
                trade=trade,
                exit_reason="adverse_flow",
                bars_held=bars_held,
            )
            result["adverse_flow"] = adverse_metrics

    # ── 6. Trailing stop update (effective next bar) ──
    if session.active_position:
        be_update = ee.update_trailing_from_close(session, session.active_position, bar)
        if isinstance(be_update, dict) and be_update:
            result["break_even"] = be_update
        cfg = (
            session.config
            if isinstance(getattr(session, "config", None), TradingConfig)
            else TradingConfig()
        )
        ctx_cfg = ContextRiskConfig.from_config_obj(cfg)
        if bool(ctx_cfg.enabled):
            intraday_payload = manager._intraday_levels_indicator_payload(session)
            poc_payload = (
                intraday_payload.get("poc_migration", {})
                if isinstance(intraday_payload.get("poc_migration"), dict)
                else {}
            )
            trailing_metrics = apply_context_trailing(
                position=session.active_position,
                current_price=float(bar.close),
                levels_payload=intraday_payload,
                poc_migration_bias=str(poc_payload.get("regime_bias", "unknown")),
                config=ctx_cfg,
            )
            if trailing_metrics.get("applied", False):
                result["context_risk_trailing"] = trailing_metrics
        # Flow-deterioration trailing tightener: tighten trailing stop
        # when multi-bar flow trend is consistently declining.
        trail_flow = manager._calculate_order_flow_metrics(
            session.bars,
            lookback=min(12, len(session.bars)),
        )
        if ee.maybe_tighten_trailing_on_flow_deterioration(
            session,
            session.active_position,
            trail_flow,
        ):
            result["flow_deterioration_tighten"] = True
