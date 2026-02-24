"""Runtime method implementations extracted from DayTradingManager."""

from __future__ import annotations

from datetime import datetime, time
from typing import Any, Dict, List, Optional
import logging

from .day_trading_models import BarData, SessionPhase, TradingSession
from .trading_config import TradingConfig
from .strategies.base_strategy import Regime, Signal, SignalType
from .day_trading_runtime_intrabar import (
    calculate_intrabar_1s_snapshot as _calculate_intrabar_1s_snapshot_impl,
)
from .day_trading_runtime_portfolio import (
    cooldown_bars_remaining as _cooldown_bars_remaining_impl,
    portfolio_drawdown_snapshot as _portfolio_drawdown_snapshot_impl,
    realized_pnl_dollars as _realized_pnl_dollars_impl,
    unrealized_pnl_dollars as _unrealized_pnl_dollars_impl,
)
from .day_trading_runtime_sweep import (
    feature_vector_value as _feature_vector_value_impl,
    order_flow_metadata_snapshot as _order_flow_metadata_snapshot_impl,
    resolve_liquidity_sweep_confirmation as _resolve_liquidity_sweep_confirmation_impl,
    to_optional_float as _to_optional_float_impl,
)
from .day_trading_runtime_entry_quality import (
    runtime_evaluate_intraday_levels_entry_quality as _runtime_evaluate_intraday_levels_entry_quality_impl,
)
from .day_trading_runtime_pending import (
    process_pending_signal_entry as _process_pending_signal_entry_impl,
)
from .day_trading_runtime_position import (
    manage_active_position_lifecycle as _manage_active_position_lifecycle_impl,
)
from .day_trading_runtime.intrabar_slice import (
    runtime_evaluate_intrabar_slice as _runtime_evaluate_intrabar_slice_impl,
)
from .golden_setup_detector import build_golden_config_from_trading_config

logger = logging.getLogger(__name__)


def _normalize_direction_token(raw: Any) -> str:
    token = str(raw or "").strip().lower()
    if token in {"bullish", "long", "buy", "up", "positive"}:
        return "bullish"
    if token in {"bearish", "short", "sell", "down", "negative"}:
        return "bearish"
    return ""


def _signal_direction(signal: Optional[Signal]) -> str:
    if signal is None:
        return ""
    if signal.signal_type == SignalType.BUY:
        return "bullish"
    if signal.signal_type == SignalType.SELL:
        return "bearish"
    return ""


def runtime_process_trading_bar(
    self, 
    session: TradingSession, 
    bar: BarData, 
    timestamp: datetime,
    warmup_only: bool = False,
) -> Dict[str, Any]:
    """Process a trading bar - manage positions and generate signals."""
    logger.debug("runtime_process_trading_bar entered at %s", timestamp.isoformat())

    sweep_enabled = bool(
        isinstance(getattr(session, "config", None), TradingConfig)
        and getattr(session.config, "liquidity_sweep_detection_enabled", False)
    )
    default_sweep_reason = "disabled"
    if sweep_enabled:
        default_sweep_reason = (
            "strategy_not_selected"
            if not bool(getattr(session, "selected_strategy", None))
            else "not_evaluated"
        )
    default_liquidity_sweep: Dict[str, Any] = {
        "enabled": sweep_enabled,
        "sweep_detected": False,
        "reason": default_sweep_reason,
    }
    if not bool(getattr(session, "selected_strategy", None)):
        warnings = [
            str(item)
            for item in getattr(session, "selection_warnings", [])
            if str(item).strip()
        ]
        if warnings:
            default_liquidity_sweep["selection_warnings"] = warnings

    result = {
        'action': 'trading',
        'micro_regime': session.micro_regime,
        'liquidity_sweep': default_liquidity_sweep,
    }
    current_price = bar.close
    current_bar_index = max(0, len(session.bars) - 1)
    session_key = self._get_session_key(session.run_id, session.ticker, session.date)
    formula_indicators_cache: Optional[Dict[str, Any]] = None

    def _formula_indicators() -> Dict[str, Any]:
        nonlocal formula_indicators_cache
        if formula_indicators_cache is None:
            bars_data = session.bars[-100:] if len(session.bars) >= 100 else session.bars
            formula_indicators_cache = self._calculate_indicators(bars_data, session=session)
        return formula_indicators_cache

    if warmup_only:
        if getattr(session, "pending_signal", None) is not None:
            session.pending_signal = None
            session.pending_signal_bar_index = -1
            result["dropped_pending_signal"] = True

        regime_update = self._maybe_refresh_regime(session, current_bar_index, timestamp)
        if regime_update:
            result['regime_update'] = regime_update
            result['regime'] = regime_update.get("regime")
            result['micro_regime'] = regime_update.get("micro_regime")
            result['strategies'] = regime_update.get("strategies", [])
            result['strategy'] = regime_update.get("strategy")
            result['indicators'] = regime_update.get("indicators", {})

        result["action"] = "warmup_only"
        result["warmup_only"] = True
        result["reason"] = "Trading disabled for warmup bars"
        return result

    portfolio_drawdown = _portfolio_drawdown_snapshot_impl(
        manager=self,
        run_id=session.run_id,
        mark_session=session,
        mark_price=current_price,
        mark_bar_volume=bar.volume,
    )
    if portfolio_drawdown.get("enabled", False):
        result["portfolio_drawdown"] = portfolio_drawdown

    if portfolio_drawdown.get("halted", False):
        if session.pending_signal is not None:
            session.pending_signal = None
            session.pending_signal_bar_index = -1
            result["dropped_pending_signal"] = True

        if session.active_position:
            trade = self._close_position(
                session,
                current_price,
                timestamp,
                "portfolio_drawdown_halt",
                bar_volume=bar.volume,
            )
            session.last_exit_bar_index = current_bar_index
            result["trade_closed"] = trade.to_dict()
            bars_held = len(
                [b for b in session.bars if b.timestamp >= trade.entry_time and b.timestamp <= trade.exit_time]
            )
            result["position_closed"] = self.gate_engine.build_position_closed_payload(
                trade=trade,
                exit_reason="portfolio_drawdown_halt",
                bars_held=bars_held,
            )

        session.phase = SessionPhase.END_OF_DAY
        session.end_price = current_price
        self._persist_intraday_levels_memory(session)
        result["action"] = "portfolio_drawdown_halt"
        result["reason"] = (
            f"Run-level drawdown {portfolio_drawdown.get('drawdown_pct', 0.0):.2f}% "
            f"<= -{portfolio_drawdown.get('halt_threshold_pct', 0.0):.2f}%"
        )
        result["portfolio_halt_triggered"] = bool(portfolio_drawdown.get("halt_triggered", False))
        result["session_summary"] = self._get_session_summary(session)
        return result

    # Execute previous-bar signal at current bar open (no same-bar signal fill).
    should_return = _process_pending_signal_entry_impl(
        manager=self,
        session=session,
        bar=bar,
        timestamp=timestamp,
        current_bar_index=current_bar_index,
        result=result,
        formula_indicators=_formula_indicators,
    )
    if should_return:
        return result
    
    # If we have an active position, manage it via the exit policy engine.
    _manage_active_position_lifecycle_impl(
        manager=self,
        session=session,
        bar=bar,
        timestamp=timestamp,
        current_bar_index=current_bar_index,
        result=result,
        formula_indicators=_formula_indicators,
    )
    
    # Custom Rule: Max Daily Loss Circuit Breaker (realized + unrealized).
    current_realized_pnl = _realized_pnl_dollars_impl(session)
    current_unrealized_pnl = _unrealized_pnl_dollars_impl(
        session=session,
        current_price=current_price,
        trading_costs=self.trading_costs,
        bar_volume=bar.volume,
    )
    current_total_pnl = current_realized_pnl + current_unrealized_pnl

    if current_total_pnl < -self.max_daily_loss:
        if session.active_position:
            trade = self._close_position(
                session,
                current_price,
                timestamp,
                "max_daily_loss",
                bar_volume=bar.volume,
            )
            result['trade_closed'] = trade.to_dict()
            result['action'] = 'max_loss_stop'
            bars_held = len([b for b in session.bars if b.timestamp >= trade.entry_time and b.timestamp <= trade.exit_time])
            result['position_closed'] = self.gate_engine.build_position_closed_payload(
                trade=trade,
                exit_reason='max_daily_loss',
                bars_held=bars_held,
            )
        result['max_daily_loss_trigger'] = {
            'realized_pnl_dollars': round(current_realized_pnl, 4),
            'unrealized_pnl_dollars': round(current_unrealized_pnl, 4),
            'total_pnl_dollars': round(current_total_pnl, 4),
            'max_daily_loss': float(self.max_daily_loss),
        }

        # Stop trading for the day
        session.phase = SessionPhase.END_OF_DAY
        session.end_price = current_price
        self._persist_intraday_levels_memory(session)
        result['session_summary'] = self._get_session_summary(session)
        return result


    regime_update = self._maybe_refresh_regime(session, current_bar_index, timestamp)
    if regime_update:
        result['regime_update'] = regime_update
        result['regime'] = regime_update.get("regime")
        result['micro_regime'] = regime_update.get("micro_regime")
        result['strategies'] = regime_update.get("strategies", [])
        result['strategy'] = regime_update.get("strategy")
        result['indicators'] = regime_update.get("indicators", {})

    if not session.active_position and session.selected_strategy:
        cooldown_remaining = _cooldown_bars_remaining_impl(session, current_bar_index)
        if cooldown_remaining > 0:
            result['action'] = 'consecutive_loss_cooldown'
            result['reason'] = f'Consecutive-loss cooldown: {cooldown_remaining} bars remaining'
            result['cooldown_bars_remaining'] = cooldown_remaining
            return result

        if str(session.micro_regime or "").upper() == "UNKNOWN":
            result['action'] = 'regime_warmup'
            result['reason'] = 'ADX warmup incomplete; signal generation paused.'
            return result

        # Check trade limits
        max_daily_trades = self._resolve_max_daily_trades(session.ticker, session=session)
        if max_daily_trades is not None and len(session.trades) >= max_daily_trades:
            result['action'] = 'trade_limit_reached'
            result['reason'] = f'Max trades per day ({max_daily_trades}) reached'
            return result
        
        # Check cooldown between trades
        last_trade_bar = self.last_trade_bar_index.get(session_key, -self.trade_cooldown_bars)
        bars_since_last_trade = current_bar_index - last_trade_bar
        if bars_since_last_trade < self.trade_cooldown_bars:
            result['action'] = 'cooldown_active'
            result['reason'] = f'Cooldown: {self.trade_cooldown_bars - bars_since_last_trade} bars remaining'
            return result
        
        # Optional ticker-specific time filter from AOS config.
        bar_time = self._to_market_time(timestamp).time()
        bar_hour = bar_time.hour
        
        ticker_aos_config = self.ticker_params.get(session.ticker.upper(), {})
        trading_hours = ticker_aos_config.get('trading_hours', None)

        if trading_hours and ticker_aos_config.get("time_filter_enabled", True):
            if bar_hour not in trading_hours:
                result['action'] = 'time_filter'
                result['reason'] = f'Hour {bar_hour}:00 not in allowed hours {trading_hours}'
                return result
        
        # ── Decision Engine ────────────────────────────────────
        orch = session.orchestrator
        if orch is None:
            raise RuntimeError("Orchestrator is not initialized")

        bars_data = session.bars[-100:] if len(session.bars) >= 100 else session.bars
        ohlcv = {
            'open': [b.open for b in bars_data],
            'high': [b.high for b in bars_data],
            'low': [b.low for b in bars_data],
            'close': [b.close for b in bars_data],
            'volume': [b.volume for b in bars_data]
        }
        indicators = self._calculate_indicators(bars_data, session=session)
        regime = session.detected_regime or Regime.MIXED
        flow_metrics = dict(indicators.get('order_flow') or {})
        fv = orch.current_feature_vector
        l2_aggression_z = _feature_vector_value_impl(fv, "l2_aggression_z", 0.0)
        l2_book_pressure_z = _feature_vector_value_impl(fv, "l2_book_pressure_z", 0.0)
        flow_metrics["l2_aggression_z"] = l2_aggression_z
        flow_metrics["l2_book_pressure_z"] = l2_book_pressure_z
        golden_cfg = build_golden_config_from_trading_config(
            session.config if isinstance(getattr(session, "config", None), TradingConfig) else TradingConfig()
        )
        golden_setup_eval = self.golden_setup_detector.evaluate(
            bar=bar,
            bars=bars_data,
            flow_metrics=flow_metrics,
            intraday_levels_state=(
                dict(session.intraday_levels_state)
                if isinstance(getattr(session, "intraday_levels_state", None), dict)
                else {}
            ),
            current_price=float(current_price),
            vwap=_to_optional_float_impl(getattr(bar, "vwap", None)),
            regime=str(session.micro_regime or ""),
            golden_entries_today=int(getattr(session, "golden_setup_entries_today", 0) or 0),
            last_golden_bar_index=int(
                getattr(session, "golden_setup_last_entry_bar_index", -99) or -99
            ),
            current_bar_index=current_bar_index,
            config=golden_cfg,
        )
        golden_setup_payload = golden_setup_eval.to_dict()
        session.golden_setup_result = dict(golden_setup_payload)
        result["golden_setup"] = dict(golden_setup_payload)

        ticker_cfg = self.ticker_params.get(session.ticker.upper(), {})
        is_long_only = bool(ticker_cfg.get("long_only", False))

        if not bool(session.potential_sweep_active):
            sweep_detection = self._detect_liquidity_sweep(
                session=session,
                current_price=current_price,
                fv=fv,
                flow_metrics=flow_metrics,
            )
        else:
            sweep_detection = {
                "enabled": bool(
                    isinstance(session.config, TradingConfig)
                    and getattr(session.config, "liquidity_sweep_detection_enabled", False)
                ),
                "sweep_detected": False,
                "reason": "awaiting_confirmation",
                **(
                    dict(session.potential_sweep_context)
                    if isinstance(session.potential_sweep_context, dict)
                    else {}
                ),
            }
        
        reason = sweep_detection.get('reason')
        if reason and reason != 'disabled' and reason != 'awaiting_confirmation':
            logger.debug(
                "Liquidity sweep detection reason=%s payload=%s",
                reason,
                sweep_detection,
            )
            
        result["liquidity_sweep"] = dict(sweep_detection)
        
        sweep_confirmation = _resolve_liquidity_sweep_confirmation_impl(
            session=session,
            current_bar_index=current_bar_index,
            current_price=current_price,
            flow_metrics=flow_metrics,
        )
        if sweep_confirmation.get("active") or sweep_confirmation.get("reason") == "confirmed":
            result["liquidity_sweep_confirmation"] = dict(sweep_confirmation)
        if bool(sweep_confirmation.get("confirmed", False)):
            sweep_direction = str(sweep_confirmation.get("direction", "long")).strip().lower()
            atr_now = (
                _to_optional_float_impl(self._latest_indicator_value(indicators, "atr", bars_data))
                or 0.0
            )
            level_price = _to_optional_float_impl(sweep_confirmation.get("level_price")) or current_price
            risk_buffer_abs = max(current_price * 0.001, atr_now * 0.25)
            strategy_name = str(session.selected_strategy or "").strip()
            if not strategy_name:
                strategy_name = (
                    session.active_strategies[0]
                    if isinstance(session.active_strategies, list) and session.active_strategies
                    else "absorption_reversal"
                )
            if sweep_direction == "short":
                signal_type = SignalType.SELL
                base_sl = max(current_price + risk_buffer_abs, level_price + risk_buffer_abs)
                risk_abs = max(base_sl - current_price, current_price * 0.002)
                base_tp = max(0.0, current_price - (risk_abs * 2.0))
                sweep_reason = (
                    f"Liquidity sweep confirmed at resistance {level_price:.4f}; "
                    "absorption exhausted and flow flipped lower."
                )
            else:
                sweep_direction = "long"
                signal_type = SignalType.BUY
                base_sl = max(0.0, min(current_price - risk_buffer_abs, level_price - risk_buffer_abs))
                risk_abs = max(current_price - base_sl, current_price * 0.002)
                base_tp = current_price + (risk_abs * 2.0)
                sweep_reason = (
                    f"Liquidity sweep confirmed at support {level_price:.4f}; "
                    "absorption held and flow flipped higher."
                )

            sweep_signal = Signal(
                strategy_name=strategy_name,
                signal_type=signal_type,
                price=float(current_price),
                timestamp=timestamp,
                confidence=92.0,
                stop_loss=float(base_sl),
                take_profit=float(base_tp),
                trailing_stop=True,
                trailing_stop_pct=0.8,
                reasoning=sweep_reason,
                metadata={
                    "sweep_triggered": True,
                    "sweep_detected": True,
                    "liquidity_sweep": {
                        **dict(sweep_confirmation),
                        "detected": True,
                        "atr": atr_now,
                    },
                },
            )
            sweep_signal.metadata["order_flow"] = _order_flow_metadata_snapshot_impl(flow_metrics)
            sweep_signal.metadata["intraday_levels_payload"] = self._intraday_levels_indicator_payload(
                session
            )
            sweep_signal.metadata["level_context"] = {
                "gate": "liquidity_sweep_confirmation",
                "passed": True,
                "reason": "liquidity_sweep_confirmed",
                "direction": sweep_direction,
                "level_price": level_price,
                "vwap_execution_flow": float(
                    sweep_confirmation.get("vwap_execution_flow", 0.0) or 0.0
                ),
            }
            risk_pct = abs(float(sweep_signal.stop_loss) - float(sweep_signal.price)) / max(
                float(sweep_signal.price), 1e-9
            ) * 100.0
            if risk_pct < 0.10:
                result["action"] = "liquidity_sweep_filtered"
                result["reason"] = "cost_aware_sweep_risk_too_small"
                result["signal_rejected"] = {
                    "gate": "cost_aware_sweep",
                    "risk_pct": round(risk_pct, 4),
                    "min_required": 0.10,
                    "timestamp": timestamp.isoformat(),
                }
                return result

            session.signals.append(sweep_signal)
            result["signal"] = sweep_signal.to_dict()
            result["signals"] = [sweep_signal.to_dict()]
            session.pending_signal = sweep_signal
            session.pending_signal_bar_index = current_bar_index
            result["action"] = "signal_queued"
            result["queued_for_next_bar"] = True
            result["sweep_detected"] = True
            return result

        mu_choppy_filter_enabled = self._is_mu_choppy_filter_enabled(
            session.ticker,
            session=session,
        )
        tcbbo_regime_override = self.gate_engine.tcbbo_directional_override(session)
        if bool(tcbbo_regime_override.get("enabled", False)):
            result["tcbbo_regime_override"] = dict(tcbbo_regime_override)
        golden_bypass_choppy = bool(golden_setup_payload.get("bypass_choppy", False))
        if golden_bypass_choppy:
            result["golden_setup_choppy_bypass"] = True

        if mu_choppy_filter_enabled and self._is_mu_choppy_blocked(session, regime):
            if (
                not bool(tcbbo_regime_override.get("applied", False))
                and not golden_bypass_choppy
            ):
                result['action'] = 'regime_filter'
                result['reason'] = "MU choppy regime filter active"
                result['signal_rejected'] = {
                    'gate': 'mu_choppy_filter',
                    'ticker': session.ticker,
                    'regime': regime.value if regime else None,
                    'micro_regime': session.micro_regime,
                    'timestamp': timestamp.isoformat(),
                    'mu_choppy_hard_block_enabled': bool(mu_choppy_filter_enabled),
                    'tcbbo_regime_override': dict(tcbbo_regime_override),
                    'golden_setup': dict(golden_setup_payload),
                }
                return result

        # ── 5s Intrabar Checkpoint Loop ──────────────────────
        raw_step = getattr(getattr(session, "config", None), "intrabar_eval_step_seconds", 5)
        try:
            step = int(raw_step)
        except (TypeError, ValueError):
            step = 5
        step = max(1, min(60, step))

        raw_quotes = getattr(bar, "intrabar_quotes_1s", None)
        checkpoints_meta = []

        def _quote_midpoint(row: Dict[str, Any]) -> Optional[float]:
            try:
                bid = float(row.get("bid", 0.0) or 0.0)
                ask = float(row.get("ask", 0.0) or 0.0)
            except (TypeError, ValueError):
                return None
            if bid > 0.0 and ask > 0.0:
                return (bid + ask) / 2.0
            if ask > 0.0:
                return ask
            if bid > 0.0:
                return bid
            return None

        if isinstance(raw_quotes, list) and raw_quotes:
            import copy

            normalized_by_second: Dict[int, Dict[str, Any]] = {}
            for item in raw_quotes:
                if not isinstance(item, dict):
                    continue
                try:
                    sec = int(float(item.get("s", 0) or 0))
                except (TypeError, ValueError):
                    continue
                if 0 <= sec <= 59:
                    normalized_by_second[sec] = dict(item, s=sec)

            ordered_seconds = sorted(normalized_by_second.keys())
            prefix_quotes: List[Dict[str, Any]] = []
            for idx, sec in enumerate(ordered_seconds):
                prefix_quotes.append(normalized_by_second[sec])
                is_boundary = (sec % step == 0) or (sec == 59) or (idx == len(ordered_seconds) - 1)
                if not is_boundary:
                    continue

                cp_bar = copy.copy(bar)
                cp_quotes = [dict(row) for row in prefix_quotes]
                cp_bar.intrabar_quotes_1s = cp_quotes

                mids: List[float] = []
                for row in cp_quotes:
                    mid = _quote_midpoint(row)
                    if mid is not None and mid > 0.0:
                        mids.append(mid)

                if mids:
                    base_open = float(bar.open) if float(bar.open) > 0.0 else mids[0]
                    cp_open = float(base_open)
                    cp_close = float(mids[-1])
                    cp_high = float(max([cp_open] + mids))
                    cp_low = float(min([cp_open] + mids))
                    cp_vwap: Optional[float] = float(sum(mids) / len(mids))
                else:
                    cp_open = float(bar.open)
                    cp_high = float(bar.high)
                    cp_low = float(bar.low)
                    cp_close = float(bar.close)
                    cp_vwap = _to_optional_float_impl(getattr(bar, "vwap", None))

                elapsed_ratio = min(1.0, max(1.0 / 60.0, float(sec + 1) / 60.0))
                cp_volume = float(bar.volume or 0.0) * elapsed_ratio

                cp_bar.open = cp_open
                cp_bar.high = cp_high
                cp_bar.low = cp_low
                cp_bar.close = cp_close
                cp_bar.volume = cp_volume
                cp_bar.vwap = cp_vwap

                cp_ts = timestamp.replace(second=sec, microsecond=0)
                checkpoints_meta.append((cp_bar, cp_ts, sec))

        if not checkpoints_meta:
            fallback_sec = max(0, min(int(getattr(timestamp, "second", 0) or 0), 59))
            checkpoints_meta.append((bar, timestamp.replace(microsecond=0), fallback_sec))

        intrabar_eval_trace = {
            "schema_version": 1,
            "source": "intrabar_quote_checkpoints",
            "minute_timestamp": timestamp.replace(second=0, microsecond=0).isoformat(),
            "step_seconds": step,
            "checkpoints": []
        }

        last_slice_res: Optional[Dict[str, Any]] = None
        trigger_slice_res: Optional[Dict[str, Any]] = None
        for cp_bar, cp_ts, sec in checkpoints_meta:
            slice_res = _runtime_evaluate_intrabar_slice_impl(self, session, cp_bar, cp_ts)
            cp_layer_scores = (
                slice_res.get("layer_scores")
                if isinstance(slice_res, dict)
                else None
            )
            cp_payload = {
                "timestamp": slice_res.get("timestamp", cp_ts.isoformat()),
                "offset_sec": sec,
                "layer_scores": cp_layer_scores,
                "intrabar_1s": _calculate_intrabar_1s_snapshot_impl(cp_bar),
                "provisional": True,
            }
            if "signal_rejected" in slice_res:
                cp_payload["signal_rejected"] = slice_res["signal_rejected"]
            if "candidate_diagnostics" in slice_res:
                cp_payload["candidate_diagnostics"] = slice_res["candidate_diagnostics"]

            intrabar_eval_trace["checkpoints"].append(cp_payload)
            last_slice_res = slice_res
            if (
                trigger_slice_res is None
                and slice_res.get("_raw_signal") is not None
                and bool((cp_layer_scores or {}).get("passed", False))
            ):
                trigger_slice_res = slice_res

        if intrabar_eval_trace["checkpoints"]:
            intrabar_eval_trace["checkpoints"][-1]["provisional"] = False
            intrabar_eval_trace["checkpoint_count"] = len(intrabar_eval_trace["checkpoints"])

        decision_slice_res = trigger_slice_res or last_slice_res or {}
        result["intrabar_eval_trace"] = intrabar_eval_trace
        if "layer_scores" in decision_slice_res:
            result["layer_scores"] = decision_slice_res["layer_scores"]
        if "signal_rejected" in decision_slice_res:
            result["signal_rejected"] = decision_slice_res["signal_rejected"]
        if "candidate_diagnostics" in decision_slice_res:
            result["candidate_diagnostics"] = decision_slice_res["candidate_diagnostics"]

        # Synthesize a mock Decision object so the subsequent gate sequence behaves identically.
        _ls = decision_slice_res.get("layer_scores", {})
        combined_score = float(decision_slice_res.get("_combined_score_raw", _ls.get("combined_score", 0.0)) or 0.0)
        effective_trade_threshold = float(_ls.get("threshold_used", 0.0))
        passed_trade_threshold = bool(
            decision_slice_res.get("_passed_trade_threshold", combined_score >= effective_trade_threshold)
        )
        tod_boost = float(_ls.get("tod_threshold_boost", 0.0))
        headwind_boost = float(_ls.get("headwind_threshold_boost", 0.0))
        headwind_metrics = _ls.get("cross_asset_headwind", {})
        required_confirming_sources = int(_ls.get("required_confirming_sources", 2))
        threshold_used_reason = str(_ls.get("threshold_used_reason", "base_threshold"))
        
        class DecisionProxy:
            pass
        decision = DecisionProxy()
        decision.combined_score = combined_score
        decision.execute = bool(decision_slice_res.get("_decision_execute", _ls.get("passed", False)))
        decision.signal = decision_slice_res.get("_raw_signal")
        decision.strategy_score = float(_ls.get("strategy_score", 0.0))
        decision.combined_raw = float(_ls.get("combined_raw", 0.0))
        decision.combined_norm_0_100 = _ls.get("combined_norm_0_100", None)
        decision.threshold = float(_ls.get("threshold", 0.0))
        decision.reasoning = str(
            decision_slice_res.get("_decision_reasoning", "Checkpoints evaluated")
            or "Checkpoints evaluated"
        )

        if decision.execute and decision.signal and passed_trade_threshold:
            signal = decision.signal
            _l2_strategy_key = str(signal.strategy_name or "").strip().lower()
            _L2_RELAXED_STRATEGIES = {"mean_reversion", "rotation", "vwap_magnet", "volumeprofile"}
            _l2_relaxed = _l2_strategy_key in _L2_RELAXED_STRATEGIES

            l2_passed, l2_metrics = self.gate_engine.passes_l2_confirmation(
                session,
                signal,
                flow_metrics=flow_metrics,
            )
            result['l2_confirmation'] = l2_metrics

            if _l2_relaxed and not l2_passed:
                l2_has_coverage = bool(l2_metrics.get("has_l2_coverage", False))
                if l2_has_coverage:
                    l2_passed = True
                    l2_metrics["relaxed_for_strategy"] = _l2_strategy_key
                    signal.metadata.setdefault("l2_relaxed_entry", True)

            if not l2_passed:
                # -- Weak L2 fast break-even: let weak L2 entries through with metadata --
                weak_l2_enabled = bool(getattr(session.config, "weak_l2_fast_break_even_enabled", False))
                hard_l2_block = bool(l2_metrics.get("hard_block", False))
                l2_reason = str(l2_metrics.get("reason", "") or "")
                l2_aggression = abs(
                    float(
                        l2_metrics.get(
                            "signed_aggression_avg",
                            l2_metrics.get("signed_aggression", 0.0),
                        ) or 0.0
                    )
                )
                weak_l2_threshold = float(getattr(session.config, "weak_l2_aggression_threshold", 0.05))
                if (
                    (not hard_l2_block)
                    and weak_l2_enabled
                    and l2_reason == "l2_confirmation_failed"
                    and l2_aggression <= weak_l2_threshold
                ):
                    override_hold = int(getattr(session.config, "weak_l2_break_even_min_hold_bars", 2))
                    signal.metadata["weak_l2_entry"] = True
                    signal.metadata["weak_l2_break_even_override"] = {
                        "break_even_min_hold_bars": override_hold,
                        "original_aggression": round(l2_aggression, 4),
                        "threshold": weak_l2_threshold,
                    }
                    result["weak_l2_entry"] = True
                    result["weak_l2_override"] = signal.metadata["weak_l2_break_even_override"]
                    # Fall through to remaining gates instead of blocking
                else:
                    result['action'] = 'l2_filtered'
                    result['reason'] = l2_metrics.get('reason', 'l2_confirmation_failed')
                    result['signal_rejected'] = {
                        'gate': 'l2_confirmation',
                        'strategy': signal.strategy_name,
                        'signal_type': signal.signal_type.value if signal.signal_type else None,
                        'confidence': round(signal.confidence, 1),
                        'combined_score': round(decision.combined_score, 1),
                        'threshold_used': effective_trade_threshold,
                        'regime': regime.value if regime else None,
                        'micro_regime': session.micro_regime,
                        'l2_metrics': l2_metrics,
                        'tod_threshold_boost': tod_boost,
                        'timestamp': timestamp.isoformat(),
                    }
                    return result

            # TCBBO options flow confirmation gate
            tcbbo_passed, tcbbo_metrics = self.gate_engine.passes_tcbbo_confirmation(session, signal)
            result['tcbbo_confirmation'] = tcbbo_metrics

            if not tcbbo_passed:
                result['action'] = 'tcbbo_filtered'
                result['reason'] = tcbbo_metrics.get('reason', 'tcbbo_confirmation_failed')
                result['signal_rejected'] = {
                    'gate': 'tcbbo_confirmation',
                    'strategy': signal.strategy_name,
                    'signal_type': signal.signal_type.value if signal.signal_type else None,
                    'confidence': round(signal.confidence, 1),
                    'combined_score': round(decision.combined_score, 1),
                    'threshold_used': effective_trade_threshold,
                    'regime': regime.value if regime else None,
                    'micro_regime': session.micro_regime,
                    'tcbbo_metrics': tcbbo_metrics,
                    'tod_threshold_boost': tod_boost,
                    'timestamp': timestamp.isoformat(),
                }
                return result

            # Apply TCBBO sweep confidence boost if aligned
            tcbbo_boost = float(tcbbo_metrics.get('confidence_boost', 0.0))
            if tcbbo_boost > 0:
                signal.confidence = min(100.0, signal.confidence + tcbbo_boost)
                signal.metadata.setdefault('tcbbo_confirmation', tcbbo_metrics)

            golden_conf_boost = 0.0
            golden_active = bool(golden_setup_payload.get("active", False))
            golden_setup_direction = _normalize_direction_token(
                golden_setup_payload.get("best_direction")
            )
            signal_direction = _signal_direction(signal)
            golden_applied = (
                golden_active
                and bool(golden_setup_direction)
                and golden_setup_direction == signal_direction
            )
            if golden_applied:
                golden_conf_boost = max(
                    0.0,
                    float(golden_setup_payload.get("confidence_boost", 0.0) or 0.0),
                )
                if golden_conf_boost > 0.0:
                    signal.confidence = min(100.0, signal.confidence + golden_conf_boost)
                signal.metadata["golden_setup"] = {
                    **dict(golden_setup_payload),
                    "applied": True,
                    "signal_direction": signal_direction,
                    "applied_confidence_boost": round(golden_conf_boost, 4),
                    "applied_threshold_relief": round(
                        float(result.get("layer_scores", {}).get("golden_setup_relief", 0.0) or 0.0),
                        4,
                    ),
                }
                result["golden_setup"] = dict(signal.metadata["golden_setup"])
            if isinstance(result.get("layer_scores"), dict):
                result["layer_scores"]["golden_setup_applied"] = bool(golden_applied)
                result["layer_scores"]["golden_setup_confidence_boost"] = round(
                    float(golden_conf_boost), 4
                )

            confirming_stats = self._confirming_source_stats(signal)
            confirming_sources = confirming_stats['confirming_sources']
            result['layer_scores']['confirming_sources'] = confirming_sources
            result['layer_scores']['confirming_sources_source'] = confirming_stats['count_source']
            result['layer_scores']['aligned_evidence_sources'] = confirming_stats['aligned_evidence_sources']
            result['layer_scores']['aligned_source_keys'] = confirming_stats['aligned_source_keys']

            _strategy_key = str(signal.strategy_name or "").strip().lower()
            _LEVEL_STRATEGIES = {"mean_reversion", "rotation", "vwap_magnet", "volumeprofile"}
            if _strategy_key in _LEVEL_STRATEGIES:
                required_confirming_sources = min(required_confirming_sources, 2)
            elif _strategy_key == "pullback":
                required_confirming_sources = min(
                    required_confirming_sources,
                    2 + (1 if (session.detected_regime or Regime.MIXED) in {Regime.CHOPPY, Regime.MIXED} else 0),
                )
            result['layer_scores']['required_confirming_sources'] = required_confirming_sources

            if confirming_sources < required_confirming_sources:
                result['action'] = 'confirming_sources_filtered'
                result['reason'] = (
                    f"Confirming sources {confirming_sources} below required "
                    f"{required_confirming_sources}"
                )
                aligned_set = set(confirming_stats['aligned_source_keys'])
                non_aligned_keys = []
                signal_direction = self.evidence_service.signal_direction(signal) if hasattr(self, 'evidence_service') else None
                raw_evidence = signal.metadata.get("evidence_sources", []) if isinstance(signal.metadata, dict) else []
                for src in (raw_evidence if isinstance(raw_evidence, list) else []):
                    if not isinstance(src, dict):
                        continue
                    src_type = str(src.get("type", "")).strip().lower()
                    src_name = str(src.get("name", "")).strip().lower()
                    if not src_type or not src_name:
                        continue
                    src_key = f"{src_type}:{src_name}"
                    if src_key not in aligned_set:
                        src_dir = str(src.get("direction", "")).strip().lower()
                        non_aligned_keys.append({"key": src_key, "direction": src_dir})
                seen_na = set()
                unique_non_aligned = []
                for item in non_aligned_keys:
                    if item["key"] not in seen_na:
                        seen_na.add(item["key"])
                        unique_non_aligned.append(item)
                result['signal_rejected'] = {
                    'gate': 'confirming_sources',
                    'strategy': signal.strategy_name,
                    'signal_type': signal.signal_type.value if signal.signal_type else None,
                    'confidence': round(signal.confidence, 1),
                    'combined_score': round(decision.combined_score, 1),
                    'threshold_used': effective_trade_threshold,
                    'regime': regime.value if regime else None,
                    'micro_regime': session.micro_regime,
                    'actual_confirming_sources': confirming_sources,
                    'required_confirming_sources': required_confirming_sources,
                    'aligned_evidence_sources': confirming_stats['aligned_evidence_sources'],
                    'count_source': confirming_stats['count_source'],
                    'aligned_source_keys': confirming_stats['aligned_source_keys'],
                    'non_aligned_source_keys': unique_non_aligned,
                    'signal_direction': signal_direction,
                    'tod_threshold_boost': tod_boost,
                    'timestamp': timestamp.isoformat(),
                }
                return result

            momentum_flow_passed, momentum_flow_metrics = (
                self._passes_momentum_flow_delta_confirmation(signal)
            )
            result['momentum_flow_confirmation'] = momentum_flow_metrics

            if not momentum_flow_passed:
                result['action'] = 'momentum_flow_filtered'
                result['reason'] = momentum_flow_metrics.get(
                    'reason',
                    'momentum_flow_delta_divergence_required',
                )
                result['signal_rejected'] = {
                    'gate': 'momentum_flow_delta_divergence',
                    'strategy': signal.strategy_name,
                    'signal_type': signal.signal_type.value if signal.signal_type else None,
                    'confidence': round(signal.confidence, 1),
                    'combined_score': round(decision.combined_score, 1),
                    'threshold_used': effective_trade_threshold,
                    'regime': regime.value if regime else None,
                    'micro_regime': session.micro_regime,
                    'momentum_flow_confirmation': momentum_flow_metrics,
                    'tod_threshold_boost': tod_boost,
                    'timestamp': timestamp.isoformat(),
                }
                return result

            momentum_diversification_passed, momentum_diversification_metrics = (
                self.gate_engine.passes_momentum_diversification_gate(
                    session=session,
                    signal=signal,
                    flow_metrics=flow_metrics,
                )
            )
            result['momentum_diversification'] = momentum_diversification_metrics

            if not momentum_diversification_passed:
                result['action'] = 'momentum_diversification_filtered'
                result['reason'] = momentum_diversification_metrics.get(
                    'reason',
                    'momentum_diversification_gate_failed',
                )
                result['signal_rejected'] = {
                    'gate': 'momentum_diversification',
                    'strategy': signal.strategy_name,
                    'signal_type': signal.signal_type.value if signal.signal_type else None,
                    'confidence': round(signal.confidence, 1),
                    'combined_score': round(decision.combined_score, 1),
                    'threshold_used': effective_trade_threshold,
                    'regime': regime.value if regime else None,
                    'micro_regime': session.micro_regime,
                    'momentum_diversification': momentum_diversification_metrics,
                    'tod_threshold_boost': tod_boost,
                    'timestamp': timestamp.isoformat(),
                }
                return result

            signal.metadata.setdefault('l2_confirmation', l2_metrics)
            signal.metadata.setdefault('momentum_flow_confirmation', momentum_flow_metrics)
            signal.metadata.setdefault('momentum_diversification', momentum_diversification_metrics)
            signal.metadata.setdefault(
                "order_flow",
                _order_flow_metadata_snapshot_impl(flow_metrics),
            )
            # Enrich signal metadata with regime context for post-mortem
            signal.metadata['regime'] = regime.value if regime else None
            signal.metadata['micro_regime'] = session.micro_regime
            signal.metadata.setdefault('layer_scores', {})
            signal.metadata['layer_scores'].update(result['layer_scores'])
            level_context = _runtime_evaluate_intraday_levels_entry_quality_impl(
                self=self,
                session=session,
                signal=signal,
                current_price=current_price,
                current_bar_index=current_bar_index,
            )
            result["level_context"] = level_context
            signal.metadata["level_context"] = level_context
            if not level_context.get("passed", True):
                result['action'] = 'intraday_levels_filtered'
                result['reason'] = level_context.get(
                    'reason',
                    'intraday_levels_entry_quality_failed',
                )
                result['signal_rejected'] = {
                    'gate': 'intraday_levels_entry_quality',
                    'strategy': signal.strategy_name,
                    'signal_type': signal.signal_type.value if signal.signal_type else None,
                    'confidence': round(signal.confidence, 1),
                    'combined_score': round(decision.combined_score, 1),
                    'threshold_used': effective_trade_threshold,
                    'regime': regime.value if regime else None,
                    'micro_regime': session.micro_regime,
                    'level_context': level_context,
                    'tod_threshold_boost': tod_boost,
                    'timestamp': timestamp.isoformat(),
                }
                return result
            target_override = _to_optional_float_impl(level_context.get("target_price_override"))
            if target_override is not None and target_override > 0.0:
                signal.take_profit = target_override
                signal.metadata["target_price_source"] = "intraday_levels_poc"
                signal.metadata["target_price_override"] = target_override
            strategy_key = self._canonical_strategy_key(signal.strategy_name or "")
            strategy_obj = self.strategies.get(strategy_key)
            if strategy_obj is not None:
                entry_formula_ctx = self.strategy_evaluator.build_strategy_formula_context(
                    session=session,
                    bar=bar,
                    indicators=indicators,
                    flow=flow_metrics,
                    current_bar_index=current_bar_index,
                    signal=signal,
                )
                custom_entry_formula = self.strategy_evaluator.evaluate_strategy_custom_formula(
                    strategy=strategy_obj,
                    formula_type="entry",
                    context=entry_formula_ctx,
                )
                if custom_entry_formula.get("enabled", False):
                    result["custom_entry_formula"] = custom_entry_formula
                if (
                    custom_entry_formula.get("enabled", False)
                    and not custom_entry_formula.get("passed", False)
                ):
                    result['action'] = 'custom_entry_formula_filtered'
                    result['reason'] = custom_entry_formula.get(
                        'error'
                    ) or 'Custom entry formula returned false.'
                    result['signal_rejected'] = {
                        'gate': 'custom_entry_formula',
                        'strategy': signal.strategy_name,
                        'signal_type': signal.signal_type.value if signal.signal_type else None,
                        'confidence': round(signal.confidence, 1),
                        'combined_score': round(decision.combined_score, 1),
                        'threshold_used': effective_trade_threshold,
                        'regime': regime.value if regime else None,
                        'micro_regime': session.micro_regime,
                        'custom_entry_formula': custom_entry_formula,
                        'timestamp': timestamp.isoformat(),
                    }
                    return result
            signal.metadata["intraday_levels_payload"] = self._intraday_levels_indicator_payload(
                session
            )
            session.signals.append(signal)
            result['signal'] = signal.to_dict()
            result['signals'] = [signal.to_dict()]  # Array format for frontend
            
            # Queue signal for next bar open (no same-bar execution).
            if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                logger.debug(
                    "Queue candidate before cost filter: bar=%s ts=%s strategy=%s stop=%s price=%s",
                    current_bar_index,
                    timestamp.isoformat(),
                    signal.strategy_name,
                    getattr(signal, "stop_loss", 0),
                    signal.price,
                )
                
                # Cost-aware filter: reject if expected risk is too small vs costs
                # This eliminates micro-moves where fees dominate returns
                if signal.stop_loss and signal.stop_loss > 0:
                    risk_pct = abs(signal.stop_loss - signal.price) / signal.price * 100
                    _cfg = getattr(session, "config", None)
                    min_risk_vs_costs = float(getattr(_cfg, "cost_gate_risk_multiplier", 5.0) or 5.0)
                    cost_pct = float(getattr(_cfg, "cost_gate_cost_pct", 0.02) or 0.02)
                    # Rotation/MR naturally have tight SLs; lower the bar.
                    _cost_strat = str(signal.strategy_name or "").strip().lower()
                    _TIGHT_SL_STRATS = {"rotation", "mean_reversion", "vwap_magnet", "volumeprofile"}
                    if _cost_strat in _TIGHT_SL_STRATS:
                        min_risk_vs_costs = min(min_risk_vs_costs, 3.0)
                    if risk_pct < min_risk_vs_costs * cost_pct:
                        logger.debug(
                            "Signal rejected by cost-aware gate: risk_pct=%.4f min_required=%.4f ts=%s",
                            risk_pct,
                            min_risk_vs_costs * cost_pct,
                            timestamp.isoformat(),
                        )
                        result['signal_rejected'] = {
                            'gate': 'cost_aware',
                            'risk_pct': round(risk_pct, 4),
                            'min_required': round(min_risk_vs_costs * cost_pct, 4),
                            'reasoning': f"Risk {risk_pct:.3f}% < {min_risk_vs_costs}× costs",
                        }
                        return result
                
                # Store EV margin on signal metadata for downstream relaxation
                _ev_margin = round(
                    float(decision.combined_score) - float(effective_trade_threshold), 2
                )
                signal.metadata["ev_margin"] = _ev_margin
                logger.debug(
                    "Signal queued: bar=%s ts=%s strategy=%s",
                    current_bar_index,
                    timestamp.isoformat(),
                    signal.strategy_name,
                )
                session.pending_signal = signal
                session.pending_signal_bar_index = current_bar_index
                result['action'] = 'signal_queued'
                result['queued_for_next_bar'] = True
                # ── Entry diagnostics: candidate breakdown ──
                cand_diag = signal.metadata.get("candidate_diagnostics", {})
                if cand_diag:
                    result['entry_diagnostics'] = {
                        'strategy_name': cand_diag.get('strategy_name'),
                        'candidate_strategies_count': cand_diag.get('candidate_strategies_count', 0),
                        'active_strategies_count': cand_diag.get('active_strategies_count', 0),
                        'active_strategies': cand_diag.get('active_strategies', []),
                        'top3': cand_diag.get('top3', []),
                        'sources_confirm_breakdown': {
                            'confirming_sources': result.get('layer_scores', {}).get('confirming_sources'),
                            'aligned_source_keys': result.get('layer_scores', {}).get('aligned_source_keys', []),
                            'combined_score': result.get('layer_scores', {}).get('combined_score'),
                            'threshold_used': result.get('layer_scores', {}).get('threshold_used'),
                            'l2_has_coverage': result.get('layer_scores', {}).get('l2_has_coverage'),
                        },
                    }
        elif decision.combined_score > 0 and (not decision.execute or not passed_trade_threshold):
            # Signal candidate existed but was rejected by threshold gate.
            # Emit rejection marker for post-mortem observability.
            result['signal_rejected'] = {
                'gate': 'cross_asset_headwind' if (decision.execute and not passed_trade_threshold) else 'threshold',
                'schema_version': 2,
                'combined_score': round(decision.combined_score, 1),
                'combined_raw': round(float(getattr(decision, "combined_raw", decision.combined_score) or 0.0), 1),
                'combined_norm_0_100': round(float(getattr(decision, "combined_norm_0_100", 0.0) or 0.0), 1),
                'threshold_used': effective_trade_threshold,
                'trade_gate_threshold': effective_trade_threshold,
                'threshold_used_reason': threshold_used_reason,
                'strategy_score': round(decision.strategy_score, 1),
                'regime': regime.value if regime else None,
                'micro_regime': session.micro_regime,
                'reasoning': decision.reasoning,
                'tod_threshold_boost': tod_boost,
                'headwind_threshold_boost': round(float(headwind_boost), 4),
                'cross_asset_headwind': headwind_metrics,
                'timestamp': timestamp.isoformat(),
            }

    logger.debug(
        "runtime_process_trading_bar completed: bar=%s has_liquidity_sweep=%s",
        current_bar_index,
        "liquidity_sweep" in result,
    )
    return result
