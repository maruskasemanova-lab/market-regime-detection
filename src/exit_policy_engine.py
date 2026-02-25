"""
Exit Policy Engine - All position exit logic extracted from DayTradingManager.

Responsibilities:
- SL/TP evaluation (bar-level and intrabar-level)
- Trailing stop updates (regime-aware)
- Time-based exits
- Adverse flow exits (L2-based)
- Momentum fail-fast exits
- Partial take-profit logic
- Position-closed payload building

This module now acts as a compatibility facade while concrete logic is split
into focused modules under ``src/exit_policy``.
"""
from __future__ import annotations

from datetime import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .exit_policy.break_even import force_move_to_break_even as _force_move_to_break_even
from .exit_policy.break_even import update_trailing_from_close as _update_trailing_from_close
from .exit_policy.context_policy import ContextAwareExitPolicy, DEFAULT_CONTEXT_EXIT_CONFIG
from .exit_policy.payloads import build_position_closed_payload as _build_position_closed_payload
from .exit_policy.shared import bars_held as _bars_held
from .exit_policy.shared import is_midday_window as _is_midday_window
from .exit_policy.shared import safe_intrabar_quote as _safe_intrabar_quote
from .exit_policy.types import ExitContext, ExitDecision
from .runtime_exit_formulas import evaluate_runtime_exit_formula


class ExitPolicyEngine:
    """Manages all position exit logic."""

    @staticmethod
    def effective_stop_for_position(pos) -> Tuple[Optional[float], Optional[str]]:
        """Return effective stop level and reason for the current side."""
        candidates: List[tuple] = []
        if pos.stop_loss and pos.stop_loss > 0:
            if pos.break_even_stop_active:
                candidates.append(("breakeven_stop", float(pos.stop_loss)))
            else:
                candidates.append(("stop_loss", float(pos.stop_loss)))
        if pos.trailing_stop_active and pos.trailing_stop_price and pos.trailing_stop_price > 0:
            candidates.append(("trailing_stop", float(pos.trailing_stop_price)))

        if not candidates:
            return None, None

        if pos.side == "long":
            reason, level = max(candidates, key=lambda x: x[1])
        else:
            reason, level = min(candidates, key=lambda x: x[1])
        return level, reason

    @staticmethod
    def resolve_exit_for_bar(pos, bar, *, is_entry_bar: bool = False) -> Optional[Tuple[str, float]]:
        """Resolve exit for the current bar with conservative tie-break.

        If both stop and target are hit on the same bar, stop wins.

        When *is_entry_bar* is True the position was opened on this bar's
        open.  We cannot know the exact fill second, so:
        - Skip intrabar-quote SL checks (pre-fill quotes would be noise).
        - Use bar.close instead of bar.low/high for bar-level SL to avoid
          false triggers from transient dips before the fill.
        """
        stop_level, stop_reason = ExitPolicyEngine.effective_stop_for_position(pos)

        # On the entry bar we skip intrabar SL: the quotes include prices
        # from before the position was filled, which would cause spurious
        # immediate stop-outs.
        intrabar_quotes = getattr(bar, "intrabar_quotes_1s", None)
        if intrabar_quotes and not is_entry_bar:
            intrabar_exit = ExitPolicyEngine._resolve_exit_from_intrabar_quotes(
                pos=pos,
                stop_level=stop_level,
                stop_reason=stop_reason,
                intrabar_quotes=intrabar_quotes,
            )
            if intrabar_exit:
                return intrabar_exit

        # On the entry bar, use bar.close for SL instead of bar.low/high.
        # bar.low may dip below SL transiently before the position is
        # filled at bar.open; bar.close reflects the actual end-of-bar
        # state which is a fair SL reference.
        if pos.side == "long":
            sl_check_price = bar.close if is_entry_bar else bar.low
            stop_hit = stop_level is not None and sl_check_price <= stop_level
            tp_hit = pos.take_profit > 0 and bar.high >= pos.take_profit
        else:
            sl_check_price = bar.close if is_entry_bar else bar.high
            stop_hit = stop_level is not None and sl_check_price >= stop_level
            tp_hit = pos.take_profit > 0 and bar.low <= pos.take_profit

        if stop_hit:
            return (stop_reason or "stop_loss", float(stop_level))
        if tp_hit:
            return ("take_profit", float(pos.take_profit))
        return None

    @staticmethod
    def _resolve_exit_from_intrabar_quotes(
        pos,
        stop_level: Optional[float],
        stop_reason: Optional[str],
        intrabar_quotes: List[Dict[str, float]],
    ) -> Optional[Tuple[str, float]]:
        """Resolve SL/TP using 1-second bid/ask sequence for the current minute.

        No look-ahead guarantee:
        - Uses only the quotes embedded for this bar's minute.
        - Processes in second order; first trigger wins.
        - Same-second conflicts remain conservative (stop wins).
        """
        if not intrabar_quotes:
            return None

        rows = sorted(
            (row for row in intrabar_quotes if isinstance(row, dict)),
            key=lambda row: int(_safe_intrabar_quote(row.get("s", 0))),
        )
        if not rows:
            return None

        apply_be_anti_spike = bool(
            (stop_reason or "") == "breakeven_stop"
            and bool(getattr(pos, "break_even_stop_active", False))
            and int(max(0, getattr(pos, "break_even_anti_spike_bars_remaining", 0) or 0)) > 0
        )
        anti_spike_hits_required = int(
            max(1, getattr(pos, "break_even_anti_spike_consecutive_hits_required", 2) or 2)
        )
        anti_spike_require_close_beyond = bool(
            getattr(pos, "break_even_anti_spike_require_close_beyond", True)
        )
        anti_spike_hits = int(max(0, getattr(pos, "break_even_anti_spike_consecutive_hits", 0) or 0))

        for row in rows:
            bid = _safe_intrabar_quote(row.get("bid"))
            ask = _safe_intrabar_quote(row.get("ask"))
            if bid <= 0 and ask <= 0:
                continue

            if pos.side == "long":
                stop_hit = stop_level is not None and bid > 0 and bid <= float(stop_level)
                tp_hit = pos.take_profit > 0 and bid > 0 and bid >= float(pos.take_profit)
            else:
                stop_hit = stop_level is not None and ask > 0 and ask >= float(stop_level)
                tp_hit = pos.take_profit > 0 and ask > 0 and ask <= float(pos.take_profit)

            if stop_hit:
                if apply_be_anti_spike:
                    close_beyond = False
                    if anti_spike_require_close_beyond:
                        if bid > 0 and ask > 0:
                            micro_close = (bid + ask) / 2.0
                        else:
                            micro_close = bid if pos.side == "long" else ask
                        if pos.side == "long":
                            close_beyond = micro_close > 0 and micro_close <= float(stop_level)
                        else:
                            close_beyond = micro_close > 0 and micro_close >= float(stop_level)
                    # OR semantics for anti-spike confirmation:
                    # stop can trigger either via 1s close-beyond OR via required consecutive touches.
                    if close_beyond:
                        setattr(pos, "break_even_anti_spike_consecutive_hits", 0)
                        return (stop_reason or "stop_loss", float(stop_level))
                    anti_spike_hits += 1
                    setattr(pos, "break_even_anti_spike_consecutive_hits", anti_spike_hits)
                    if anti_spike_hits < anti_spike_hits_required:
                        continue
                return (stop_reason or "stop_loss", float(stop_level))
            if tp_hit:
                if apply_be_anti_spike:
                    anti_spike_hits = 0
                    setattr(pos, "break_even_anti_spike_consecutive_hits", anti_spike_hits)
                return ("take_profit", float(pos.take_profit))

            if apply_be_anti_spike:
                anti_spike_hits = 0
                setattr(pos, "break_even_anti_spike_consecutive_hits", anti_spike_hits)

        return None

    @staticmethod
    def update_trailing_from_close(session, pos, bar) -> Dict[str, Any]:
        """Update break-even/trailing state from the current 1m close."""
        return _update_trailing_from_close(session=session, pos=pos, bar=bar)

    @staticmethod
    def force_move_to_break_even(
        session,
        pos,
        *,
        bar: Optional[Any] = None,
        reason: str = "manual",
    ) -> Dict[str, Any]:
        return _force_move_to_break_even(
            session=session,
            pos=pos,
            bar=bar,
            reason=reason,
        )

    @staticmethod
    def maybe_tighten_trailing_on_flow_deterioration(
        session,
        pos,
        flow_metrics: Dict[str, Any],
    ) -> bool:
        """Tighten trailing stop when flow quality is consistently deteriorating."""
        if not pos.trailing_stop_active or not pos.break_even_stop_active:
            return False
        if not flow_metrics.get("has_l2_coverage", False):
            return False

        flow_trend = float(flow_metrics.get("flow_score_trend_3bar", 0.0) or 0.0)
        tighten_threshold = float(
            getattr(session, "flow_deterioration_trailing_threshold", -15.0) or -15.0
        )
        tighten_pct = max(
            0.0,
            min(0.5, float(getattr(session, "flow_deterioration_trailing_tighten_pct", 0.20) or 0.20)),
        )

        if flow_trend >= tighten_threshold or tighten_pct <= 0:
            return False

        if pos.side == "long" and pos.trailing_stop_price > 0:
            distance = pos.highest_price - pos.trailing_stop_price
            if distance > 0:
                pos.trailing_stop_price = pos.trailing_stop_price + (distance * tighten_pct)
                return True
        elif pos.side == "short" and pos.trailing_stop_price > 0:
            distance = pos.trailing_stop_price - pos.lowest_price
            if distance > 0:
                pos.trailing_stop_price = pos.trailing_stop_price - (distance * tighten_pct)
                return True

        return False

    @staticmethod
    def maybe_tighten_trailing_on_tcbbo_reversal(
        session,
        pos,
        bar,
    ) -> Dict[str, Any]:
        """Tighten trailing stop when TCBBO options flow reverses against position.

        Soft tightener — reduces cushion between trailing stop and extreme price,
        but never triggers a hard exit.  Returns metrics dict for diagnostics.
        """
        metrics: Dict[str, Any] = {"applied": False, "enabled": False}

        if not getattr(session, 'tcbbo_exit_tighten_enabled', False):
            return metrics
        metrics["enabled"] = True

        if not pos.trailing_stop_active:
            return metrics

        lookback = max(1, int(getattr(session, 'tcbbo_exit_lookback_bars', 5)))
        window = session.bars[-lookback:] if session.bars else []
        covered = [b for b in window if getattr(b, 'tcbbo_has_data', False)]
        if len(covered) < 2:
            metrics["reason"] = "insufficient_tcbbo_coverage"
            return metrics

        # Compute directional flow against position
        net = sum(float(getattr(b, 'tcbbo_net_premium', 0) or 0) for b in covered)
        direction = 1.0 if pos.side == "long" else -1.0
        contra_flow = net * direction  # Negative = flow against position

        tighten_threshold = -float(getattr(session, 'tcbbo_exit_contra_threshold', 50000))
        tighten_pct = max(0.0, min(0.5, float(
            getattr(session, 'tcbbo_exit_tighten_pct', 0.15)
        )))

        metrics.update({
            "contra_flow": round(contra_flow, 2),
            "tighten_threshold": tighten_threshold,
            "tighten_pct": tighten_pct,
            "covered_bars": len(covered),
        })

        if contra_flow >= tighten_threshold or tighten_pct <= 0:
            return metrics

        # Tighten trailing stop
        if pos.side == "long" and pos.trailing_stop_price > 0:
            distance = pos.highest_price - pos.trailing_stop_price
            if distance > 0:
                pos.trailing_stop_price += distance * tighten_pct
                metrics["applied"] = True
                metrics["new_trailing_stop"] = round(pos.trailing_stop_price, 4)
        elif pos.side == "short" and pos.trailing_stop_price > 0:
            distance = pos.trailing_stop_price - pos.lowest_price
            if distance > 0:
                pos.trailing_stop_price -= distance * tighten_pct
                metrics["applied"] = True
                metrics["new_trailing_stop"] = round(pos.trailing_stop_price, 4)

        return metrics

    @staticmethod
    def should_time_exit(session, pos, current_bar_index: int, flow_metrics: Dict[str, Any]) -> bool:
        """Check if position should be closed due to time limit."""
        if session.time_exit_bars <= 0:
            return False

        try:
            from .strategies.base_strategy import Regime
        except Exception:  # pragma: no cover - fallback for legacy import layouts
            from strategies.base_strategy import Regime

        if session.detected_regime == Regime.CHOPPY:
            base_limit = float(session.choppy_time_exit_bars)
        elif session.detected_regime == Regime.TRENDING:
            base_limit = float(session.time_exit_bars) * 1.15
        else:
            base_limit = float(session.time_exit_bars)

        limit_bars = base_limit
        signed_aggression = float(flow_metrics.get("signed_aggression", 0.0) or 0.0)
        flow_score = float(flow_metrics.get("flow_score", 0.0) or 0.0)
        flow_trend = float(flow_metrics.get("flow_score_trend_3bar", 0.0) or 0.0)

        favorable = (
            (pos.side == "long" and signed_aggression >= 0.10)
            or (pos.side == "short" and signed_aggression <= -0.10)
        )
        current_price = session.bars[-1].close if session.bars else pos.entry_price
        is_profitable = (
            (pos.side == "long" and current_price > pos.entry_price)
            or (pos.side == "short" and current_price < pos.entry_price)
        )
        quality_favorable = is_profitable and flow_score > 60.0 and flow_trend > 0

        if quality_favorable:
            limit_bars *= 1.3
        elif favorable:
            limit_bars *= 1.5

        base_should_exit = _bars_held(pos, current_bar_index) >= int(limit_bars)
        regime_name = str(getattr(getattr(session, "detected_regime", None), "value", "") or "")
        formula_ctx = {
            "side": str(getattr(pos, "side", "") or ""),
            "bars_held_count": int(_bars_held(pos, current_bar_index)),
            "current_bar_index": int(current_bar_index),
            "entry_price": float(getattr(pos, "entry_price", 0.0) or 0.0),
            "current_price": float(current_price or 0.0),
            "time_exit_bars": int(getattr(session, "time_exit_bars", 0) or 0),
            "choppy_time_exit_bars": int(getattr(session, "choppy_time_exit_bars", 0) or 0),
            "regime": regime_name,
            "base_limit_bars": float(base_limit),
            "limit_bars": float(limit_bars),
            "signed_aggression": float(signed_aggression),
            "flow_score": float(flow_score),
            "flow_trend": float(flow_trend),
            "favorable": bool(favorable),
            "quality_favorable": bool(quality_favorable),
            "is_profitable": bool(is_profitable),
            "time_exit_base": bool(base_should_exit),
        }
        formula_result = evaluate_runtime_exit_formula(
            session=session,
            hook="time_exit",
            context=formula_ctx,
            default_passed=base_should_exit,
        )
        return bool(formula_result.get("passed", base_should_exit))

    @staticmethod
    def should_adverse_flow_exit(
        session,
        pos,
        current_bar_index: int,
        flow_metrics: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if L2 pressure has flipped against the position."""
        metrics: Dict[str, Any] = {
            "signed_aggression": 0.0,
            "directional_consistency": 0.0,
            "book_pressure_avg": 0.0,
        }
        if not session.adverse_flow_exit_enabled:
            return False, metrics

        bars_held_count = _bars_held(pos, current_bar_index)

        current_price = session.bars[-1].close if session.bars else pos.entry_price
        if pos.side == "long":
            unrealized_pnl = current_price - pos.entry_price
        else:
            unrealized_pnl = pos.entry_price - current_price

        effective_min_hold = int(session.adverse_flow_min_hold_bars)
        if unrealized_pnl > 0:
            effective_min_hold = max(effective_min_hold, 8)

        if bars_held_count < effective_min_hold:
            return False, metrics

        if not flow_metrics.get("has_l2_coverage", False):
            return False, metrics

        signed = float(flow_metrics.get("signed_aggression", 0.0) or 0.0)
        consistency = float(flow_metrics.get("directional_consistency", 0.0) or 0.0)
        book_pressure_avg = float(flow_metrics.get("book_pressure_avg", 0.0) or 0.0)
        threshold = max(0.02, float(session.adverse_flow_threshold))

        consistency_thr = max(0.02, float(session.adverse_flow_consistency_threshold))
        book_pressure_thr = max(0.02, float(session.adverse_book_pressure_threshold))

        metrics = {
            "signed_aggression": signed,
            "directional_consistency": consistency,
            "book_pressure_avg": book_pressure_avg,
            "threshold": threshold,
            "consistency_threshold": consistency_thr,
            "book_pressure_threshold": book_pressure_thr,
            "unrealized_pnl": unrealized_pnl,
            "effective_min_hold": effective_min_hold,
            "bars_held": bars_held_count,
        }

        absorption_rate = float(flow_metrics.get("absorption_rate", 0.0) or 0.0)
        absorption_thr = max(
            0.0,
            float(getattr(session, "adverse_flow_absorption_threshold", 0.5) or 0.5),
        )
        absorption_override = bool(
            getattr(session, "adverse_flow_absorption_override", False)
        ) and absorption_rate >= absorption_thr

        single_metric_mult = max(
            1.0,
            float(getattr(session, "adverse_flow_single_metric_multiplier", 2.0) or 2.0),
        )

        flow_trend = float(flow_metrics.get("flow_score_trend_3bar", 0.0) or 0.0)

        metrics["absorption_rate"] = absorption_rate
        metrics["absorption_override"] = absorption_override
        metrics["flow_score_trend_3bar"] = flow_trend

        is_red_bar = session.bars[-1].close < session.bars[-1].open if session.bars else False
        is_green_bar = session.bars[-1].close > session.bars[-1].open if session.bars else False

        if pos.side == "long":
            adverse_flow = signed <= -threshold and consistency >= consistency_thr
            adverse_book = book_pressure_avg <= -book_pressure_thr
            extreme_aggression = signed <= -(threshold * single_metric_mult) and is_red_bar
        else:
            adverse_flow = signed >= threshold and consistency >= consistency_thr
            adverse_book = book_pressure_avg >= book_pressure_thr
            extreme_aggression = signed >= (threshold * single_metric_mult) and is_green_bar

        should_exit = (adverse_flow and adverse_book) or extreme_aggression

        if should_exit and absorption_override and not extreme_aggression:
            metrics["suppressed_by_absorption"] = True
            should_exit = False

        formula_ctx = {
            "side": str(getattr(pos, "side", "") or ""),
            "bars_held_count": int(bars_held_count),
            "effective_min_hold": int(effective_min_hold),
            "min_hold_met": bool(bars_held_count >= effective_min_hold),
            "has_l2_coverage": bool(flow_metrics.get("has_l2_coverage", False)),
            "signed_aggression": float(signed),
            "directional_consistency": float(consistency),
            "book_pressure_avg": float(book_pressure_avg),
            "threshold": float(threshold),
            "consistency_threshold": float(consistency_thr),
            "book_pressure_threshold": float(book_pressure_thr),
            "unrealized_pnl": float(unrealized_pnl),
            "absorption_rate": float(absorption_rate),
            "absorption_override": bool(absorption_override),
            "flow_score_trend_3bar": float(flow_trend),
            "is_red_bar": bool(is_red_bar),
            "is_green_bar": bool(is_green_bar),
            "adverse_flow": bool(adverse_flow),
            "adverse_book": bool(adverse_book),
            "extreme_aggression": bool(extreme_aggression),
            "adverse_flow_base": bool(should_exit),
        }
        formula_result = evaluate_runtime_exit_formula(
            session=session,
            hook="adverse_flow_exit",
            context=formula_ctx,
            default_passed=should_exit,
        )
        metrics["runtime_formula"] = formula_result
        should_exit = bool(formula_result.get("passed", should_exit))
        return should_exit, metrics

    @staticmethod
    def should_momentum_fail_fast_exit(
        pos,
        current_bar_index: int,
        flow_metrics: Dict[str, Any],
        momentum_config: Dict[str, Any],
        momentum_sleeve_id: str = "",
        apply_to_strategies: Optional[List[str]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Early exit if L2 momentum reverses within max_bars of entry."""
        strategy_key = (pos.strategy_name or "").lower().strip().replace(" ", "_").replace("-", "_")
        if apply_to_strategies is None:
            apply_to_strategies = ["momentum_flow", "momentum", "pullback"]

        metrics: Dict[str, Any] = {
            "enabled": bool(momentum_config.get("enabled", False))
            and bool(momentum_config.get("fail_fast_exit_enabled", False)),
            "strategy_key": strategy_key,
            "apply_to_strategies": apply_to_strategies,
            "selected_sleeve_id": momentum_sleeve_id,
            "bars_held": 0,
            "passed": True,
        }
        if not metrics["enabled"] or strategy_key not in apply_to_strategies:
            return False, metrics

        bars_held_count = _bars_held(pos, current_bar_index)
        metrics["bars_held"] = bars_held_count
        max_bars = max(1, int(momentum_config.get("fail_fast_max_bars", 3)))
        if bars_held_count <= 0 or bars_held_count > max_bars:
            return False, metrics

        if not flow_metrics.get("has_l2_coverage", False):
            metrics["has_l2_coverage"] = False
            return False, metrics

        direction = 1.0 if pos.side == "long" else -1.0
        directional_signed = float(flow_metrics.get("signed_aggression", 0.0) or 0.0) * direction
        directional_book = float(flow_metrics.get("book_pressure_avg", 0.0) or 0.0) * direction
        directional_consistency = float(flow_metrics.get("directional_consistency", 0.0) or 0.0)

        signed_max = min(0.0, float(momentum_config.get("fail_fast_signed_aggression_max", -0.05)))
        book_max = min(0.0, float(momentum_config.get("fail_fast_book_pressure_max", -0.08)))
        consistency_max = max(
            0.0,
            min(1.0, float(momentum_config.get("fail_fast_directional_consistency_max", 0.35))),
        )

        adverse_signed = directional_signed <= signed_max
        adverse_book = directional_book <= book_max
        weak_consistency = directional_consistency <= consistency_max
        should_exit = bool((adverse_signed and adverse_book) or (adverse_signed and weak_consistency))

        metrics.update(
            {
                "has_l2_coverage": True,
                "directional_signed_aggression": directional_signed,
                "signed_aggression_max": signed_max,
                "directional_book_pressure": directional_book,
                "book_pressure_max": book_max,
                "directional_consistency": directional_consistency,
                "directional_consistency_max": consistency_max,
                "adverse_signed": adverse_signed,
                "adverse_book": adverse_book,
                "weak_consistency": weak_consistency,
                "passed": not should_exit,
            }
        )
        if should_exit:
            metrics["reason"] = "momentum_fail_fast_l2_flip"
        return should_exit, metrics

    @staticmethod
    def partial_take_profit_price(session, pos) -> float:
        """Compute the 1R partial take-profit target price."""
        if pos.stop_loss <= 0:
            return 0.0
        rr = max(0.25, float(session.partial_take_profit_rr))
        risk = abs(pos.entry_price - pos.stop_loss)
        if risk <= 0:
            return 0.0
        if pos.side == "long":
            return pos.entry_price + (risk * rr)
        return pos.entry_price - (risk * rr)

    @staticmethod
    def should_take_partial_profit(
        session,
        pos,
        bar,
        flow_metrics: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Evaluate whether to take a partial profit."""
        if not session.enable_partial_take_profit:
            return None
        if pos.partial_exit_done or pos.size <= 0:
            return None
        if pos.stop_loss <= 0:
            return None

        partial_price = pos.partial_take_profit_price or ExitPolicyEngine.partial_take_profit_price(session, pos)
        if partial_price <= 0:
            return None

        hit_rr_target = (bar.high >= partial_price) if pos.side == "long" else (bar.low <= partial_price)

        signed_aggression = float(flow_metrics.get("signed_aggression", 0.0) or 0.0)
        flow_trend = float(flow_metrics.get("flow_score_trend_3bar", 0.0) or 0.0)

        if pos.side == "long":
            flow_deteriorating = (
                signed_aggression <= -0.15 or flow_trend <= -10.0
            ) and bar.close > pos.entry_price
        else:
            flow_deteriorating = (
                signed_aggression >= 0.15 or flow_trend >= 10.0
            ) and bar.close < pos.entry_price

        if flow_deteriorating and not hit_rr_target:
            current_bar_index = max(0, len(session.bars) - 1)
            if _bars_held(pos, current_bar_index) < 3:
                return None
            # Enforce minimum R threshold for flow deterioration partials
            flow_min_r = float(getattr(session, "partial_flow_deterioration_min_r", 0.5) or 0.5)
            if flow_min_r > 0 and pos.stop_loss > 0:
                risk = abs(pos.entry_price - pos.stop_loss)
                if risk > 0:
                    if pos.side == "long":
                        unrealized_r = (bar.close - pos.entry_price) / risk
                    else:
                        unrealized_r = (pos.entry_price - bar.close) / risk
                    if unrealized_r < flow_min_r:
                        return None

        if not hit_rr_target and not flow_deteriorating:
            return None

        close_fraction = (
            0.5
            if flow_deteriorating and not hit_rr_target
            else float(session.partial_take_profit_fraction)
        )
        close_fraction = min(0.95, max(0.05, close_fraction))

        reason = "partial_take_profit"
        if flow_deteriorating and not hit_rr_target:
            reason = "partial_take_profit_flow_deterioration"

        exit_price = bar.close if (flow_deteriorating and not hit_rr_target) else partial_price

        return {
            "reason": reason,
            "exit_price": exit_price,
            "close_fraction": close_fraction,
            "partial_price": partial_price,
            "hit_rr_target": hit_rr_target,
            "flow_deteriorating": flow_deteriorating,
        }

    @staticmethod
    def build_position_closed_payload(
        trade,
        exit_reason: str,
        bars_held: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Serialize a closed trade into a frontend/debug-friendly payload."""
        return _build_position_closed_payload(
            trade=trade,
            exit_reason=exit_reason,
            bars_held=bars_held,
        )

    @staticmethod
    def effective_trailing_stop_pct(
        session,
        signal,
        extract_confirming_sources_fn: Callable,
        trailing_multiplier_fn: Callable,
    ) -> float:
        """Compute effective trailing distance for the active position."""
        try:
            from .strategies.base_strategy import Regime
        except Exception:  # pragma: no cover - fallback for legacy import layouts
            from strategies.base_strategy import Regime

        base_trailing = float(signal.trailing_stop_pct or 0.8)
        confirming_sources = extract_confirming_sources_fn(signal)
        trailing = base_trailing * trailing_multiplier_fn(confirming_sources)

        if session.detected_regime == Regime.TRENDING:
            trailing *= 1.05

        return max(0.4, min(2.5, trailing))

    @staticmethod
    def time_of_day_threshold_boost(bar_time: time, strategy_key: str = "") -> float:
        """Return extra threshold points for low-conviction time-of-day windows.

        MR/rotation thrive during quiet midday ranges so the penalty is
        reduced to +3 instead of the default +7.
        """
        if _is_midday_window(bar_time):
            _MIDDAY_RELAXED = {"mean_reversion", "rotation", "vwap_magnet", "volumeprofile"}
            if str(strategy_key or "").strip().lower() in _MIDDAY_RELAXED:
                return 3.0
            return 7.0
        return 0.0


__all__ = [
    "ContextAwareExitPolicy",
    "DEFAULT_CONTEXT_EXIT_CONFIG",
    "ExitContext",
    "ExitDecision",
    "ExitPolicyEngine",
]
