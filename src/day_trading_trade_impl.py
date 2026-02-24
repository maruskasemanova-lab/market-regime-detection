"""Position sizing and trade lifecycle implementations extracted from DayTradingManager."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import json
import math

from .day_trading_evidence_service import DayTradingEvidenceService
from .day_trading_models import BarData, DayTrade, TradingSession
from .exit_policy_engine import ContextAwareExitPolicy
from .position_context import EntrySnapshot, PositionContextMonitor
from .strategies.base_strategy import Position, Signal, SignalType


def _to_finite_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _build_entry_context_risk_payload(
    *,
    entry_price: float,
    side: str,
    effective_stop_loss: float,
    effective_take_profit: float,
    strategy_stop_loss: float,
    stop_loss_mode: str,
    fixed_stop_loss_pct: float,
    existing_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if isinstance(existing_payload, dict):
        payload.update(existing_payload)

    entry_px = _to_finite_float(entry_price)
    stop_px = _to_finite_float(effective_stop_loss)
    target_px = _to_finite_float(effective_take_profit)
    strategy_stop_px = _to_finite_float(strategy_stop_loss)
    is_long = str(side).strip().lower() == "long"

    if stop_px is not None and stop_px > 0.0:
        payload["adjusted_stop_loss"] = round(stop_px, 6)
    if target_px is not None and target_px > 0.0:
        payload["adjusted_take_profit"] = round(target_px, 6)
    if strategy_stop_px is not None and strategy_stop_px > 0.0:
        payload.setdefault("original_stop_loss", round(strategy_stop_px, 6))
    if target_px is not None and target_px > 0.0:
        payload.setdefault("original_take_profit", round(target_px, 6))

    if "skip" not in payload:
        payload["skip"] = False
    if "skip_reason" not in payload:
        payload["skip_reason"] = "ok"

    if entry_px is not None and entry_px > 0.0 and stop_px is not None and stop_px > 0.0:
        if is_long:
            risk_abs = max(0.0, entry_px - stop_px)
        else:
            risk_abs = max(0.0, stop_px - entry_px)
        if risk_abs > 0.0:
            risk_pct = (risk_abs / entry_px) * 100.0
            payload["risk_pct"] = round(risk_pct, 6)
            if target_px is not None and target_px > 0.0:
                if is_long:
                    room_abs = max(0.0, target_px - entry_px)
                else:
                    room_abs = max(0.0, entry_px - target_px)
                room_pct = (room_abs / entry_px) * 100.0
                payload["room_pct"] = round(room_pct, 6)
                payload["effective_rr"] = (
                    round(room_pct / risk_pct, 6) if risk_pct > 0.0 else None
                )

    sl_reason = str(payload.get("sl_reason") or "").strip()
    if not sl_reason:
        mode = str(stop_loss_mode or "").strip().lower()
        fixed_pct = max(0.0, float(fixed_stop_loss_pct or 0.0))
        if mode == "fixed":
            sl_reason = f"fixed_stop_loss_pct:{fixed_pct:.4f}"
        elif mode == "capped":
            if (
                stop_px is not None
                and stop_px > 0.0
                and strategy_stop_px is not None
                and strategy_stop_px > 0.0
                and abs(stop_px - strategy_stop_px) > 1e-9
            ):
                sl_reason = f"capped_fixed_floor:{fixed_pct:.4f}"
            else:
                sl_reason = "strategy_stop_loss"
        elif mode:
            sl_reason = f"{mode}_stop_loss"
        else:
            sl_reason = "strategy_stop_loss"
        payload["sl_reason"] = sl_reason

    tp_reason = str(payload.get("tp_reason") or "").strip()
    if not tp_reason:
        payload["tp_reason"] = (
            "strategy_take_profit" if target_px is not None and target_px > 0.0 else "no_take_profit"
        )

    return payload


def _compute_partial_realized_r(
    *,
    pos: Position,
    partial_exit_price: float,
    close_size: float,
    pre_partial_size: float,
) -> float:
    entry_price = float(getattr(pos, "entry_price", 0.0) or 0.0)
    if entry_price <= 0.0 or close_size <= 0.0 or pre_partial_size <= 0.0:
        return 0.0
    side = str(getattr(pos, "side", "long")).strip().lower()
    initial_stop = float(getattr(pos, "initial_stop_loss", 0.0) or 0.0)
    if initial_stop <= 0.0:
        initial_stop = float(getattr(pos, "stop_loss", 0.0) or 0.0)
    if initial_stop <= 0.0:
        return 0.0
    if side == "long":
        risk_abs = max(0.0, entry_price - initial_stop)
        realized_abs = max(0.0, float(partial_exit_price) - entry_price)
    else:
        risk_abs = max(0.0, initial_stop - entry_price)
        realized_abs = max(0.0, entry_price - float(partial_exit_price))
    if risk_abs <= 0.0:
        return 0.0
    sized_fraction = max(0.0, min(1.0, float(close_size) / float(pre_partial_size)))
    return (realized_abs / risk_abs) * sized_fraction



class TradeExecutionEngine:
    """Encapsulates position sizing, risk management, and trade lifecycle."""
    def __init__(
        self,
        config_service,
        evidence_service,
        exit_engine,
        ticker_params,
        get_session_key,
        manager=None,
    ):
        self.config_service = config_service
        self.evidence_service = evidence_service
        self.exit_engine = exit_engine
        self.ticker_params = ticker_params
        self.get_session_key = get_session_key
        self.manager = manager

    def calculate_position_size(
        self,
        session: TradingSession,
        signal: Signal,
        entry_price: float,
    ) -> float:
        """
        Fixed-notional position sizing.

        Default target is full account notional from FE (`account_size_usd`), with
        optional upper cap via `max_position_notional_pct`.
        """
        if entry_price <= 0:
            return 0.0

        capital = max(0.0, float(session.account_size_usd))
        if capital <= 0:
            return 0.0

        size_by_account_notional = capital / entry_price
        max_notional = capital * (max(1.0, float(session.max_position_notional_pct)) / 100.0)
        size_by_notional_cap = max_notional / entry_price
        desired_size = min(size_by_account_notional, size_by_notional_cap)
        if desired_size <= 0:
            return 0.0

        min_size = max(0.0, float(session.min_position_size))
        if desired_size < min_size and size_by_notional_cap >= min_size:
            desired_size = min_size
        return round(max(0.0, desired_size), 4)

    @staticmethod
    def extract_confirming_sources(signal: Signal) -> Optional[int]:
        return DayTradingEvidenceService.extract_confirming_sources(signal)

    @staticmethod
    def agreement_risk_multiplier(confirming_sources: Optional[int]) -> float:
        return DayTradingEvidenceService.agreement_risk_multiplier(confirming_sources)

    @staticmethod
    def trailing_multiplier(confirming_sources: Optional[int]) -> float:
        return DayTradingEvidenceService.trailing_multiplier(confirming_sources)

    def effective_trailing_stop_pct(self, session: TradingSession, signal: Signal) -> float:
        return self.exit_engine.effective_trailing_stop_pct(
            session=session,
            signal=signal,
            extract_confirming_sources_fn=self.extract_confirming_sources,
            trailing_multiplier_fn=self.trailing_multiplier,
        )

    @staticmethod
    def fixed_stop_price(entry_price: float, side: str, stop_loss_pct: float) -> float:
        stop_loss_pct = max(0.0, float(stop_loss_pct))
        if stop_loss_pct <= 0 or entry_price <= 0:
            return 0.0
        if side == "long":
            return entry_price * (1 - (stop_loss_pct / 100.0))
        return entry_price * (1 + (stop_loss_pct / 100.0))

    def resolve_stop_loss_for_entry(
        self,
        session: TradingSession,
        signal: Signal,
        entry_price: float,
        side: str,
    ) -> float:
        context_risk_md: Dict[str, Any] = {}
        if isinstance(getattr(signal, "metadata", None), dict):
            raw_context_md = signal.metadata.get("context_risk")
            if isinstance(raw_context_md, dict):
                context_risk_md = raw_context_md
        context_risk_active = bool(context_risk_md) and not bool(
            context_risk_md.get("skip", False)
        )
        try:
            context_min_sl_pct = max(
                0.0,
                float(
                    context_risk_md.get(
                        "configured_min_sl_pct",
                        context_risk_md.get("min_sl_pct", 0.0),
                    )
                    or 0.0
                ),
            )
        except (TypeError, ValueError):
            context_min_sl_pct = 0.0

        def _enforce_min_sl_floor(stop_price: float) -> float:
            if entry_price <= 0.0 or context_min_sl_pct <= 0.0:
                return float(stop_price)
            floor_stop = self.fixed_stop_price(entry_price, side, context_min_sl_pct)
            if floor_stop <= 0.0:
                return float(stop_price)
            if stop_price <= 0.0:
                return float(floor_stop)
            if side == "long" and stop_price > floor_stop:
                return float(floor_stop)
            if side != "long" and stop_price < floor_stop:
                return float(floor_stop)
            return float(stop_price)

        mode = self.manager.config_service.normalize_stop_loss_mode(session.stop_loss_mode)
        if context_risk_active and mode == "capped":
            # Context-aware risk is responsible for the entry SL placement.
            mode = "strategy"

        fixed_pct = max(0.0, float(session.fixed_stop_loss_pct))
        strategy_stop = float(signal.stop_loss) if signal.stop_loss else 0.0
        strategy_stop = _enforce_min_sl_floor(strategy_stop)
        if mode == "strategy" or fixed_pct <= 0.0 or entry_price <= 0:
            return strategy_stop

        fixed_stop = self.fixed_stop_price(entry_price, side, fixed_pct)
        fixed_stop = _enforce_min_sl_floor(fixed_stop)
        if fixed_stop <= 0.0:
            return strategy_stop
        if strategy_stop <= 0.0 or mode == "fixed":
            return fixed_stop

        # capped mode: keep strategy stop only if it is tighter than fixed stop.
        if side == "long":
            return max(strategy_stop, fixed_stop)
        return min(strategy_stop, fixed_stop)

    def simulate_entry_fill(
        self,
        desired_size: float,
        bar_volume: float,
        session: TradingSession,
    ) -> Tuple[float, float]:
        """
        Deterministic partial-fill simulation.

        If desired size exceeds max allowed participation on this bar, reduce fill size.
        """
        if desired_size <= 0:
            return 0.0, 0.0
        if bar_volume <= 0:
            return desired_size, 1.0

        max_participation = min(1.0, max(0.01, float(session.max_fill_participation_rate)))
        max_fillable = bar_volume * max_participation
        if desired_size <= max_fillable:
            return desired_size, 1.0

        raw_ratio = max_fillable / desired_size if desired_size > 0 else 0.0
        min_ratio = min(1.0, max(0.01, float(session.min_fill_ratio)))
        fill_ratio = min(1.0, max(min_ratio, raw_ratio))
        return round(desired_size * fill_ratio, 4), round(fill_ratio, 4)

    def bars_held(self, pos: Position, current_bar_index: int) -> int:
        entry_index = pos.entry_bar_index if pos.entry_bar_index is not None else current_bar_index
        return max(0, int(current_bar_index) - int(entry_index))

    def partial_take_profit_price(self, session: TradingSession, pos: Position) -> float:
        return self.exit_engine.partial_take_profit_price(session, pos)

    def build_trade_record(
        self,
        session: TradingSession,
        pos: Position,
        exit_price: float,
        exit_time: datetime,
        reason: str,
        shares: float,
        bar_volume: Optional[float] = None,
    ) -> DayTrade:
        shares = max(0.0, float(shares))
        if shares <= 0:
            raise ValueError("shares must be > 0 when building trade record")

        if pos.side == 'long':
            gross_pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
        else:
            gross_pnl_pct = (pos.entry_price - exit_price) / pos.entry_price * 100

        gross_pnl_dollars = gross_pnl_pct * pos.entry_price / 100 * shares

        costs = self.manager.trading_costs.calculate_costs(
            entry_price=pos.entry_price,
            exit_price=exit_price,
            shares=shares,
            side=pos.side,
            avg_bar_volume=bar_volume,
        )

        net_pnl_dollars = gross_pnl_dollars - costs['total']
        notional = pos.entry_price * shares
        net_pnl_pct = (net_pnl_dollars / notional * 100) if notional > 0 else 0.0

        session.trade_counter += 1
        signal_metadata = (
            dict(pos.signal_metadata)
            if isinstance(pos.signal_metadata, dict)
            else {}
        )
        be_snapshot: Dict[str, Any] = {}
        if isinstance(getattr(pos, "break_even_last_update", None), dict):
            be_snapshot.update(dict(pos.break_even_last_update))
        be_snapshot.update(
            {
                "active": bool(getattr(pos, "break_even_stop_active", False)),
                "state": str(getattr(pos, "break_even_state", "idle") or "idle"),
                "arm_bar_index": getattr(pos, "break_even_arm_bar_index", None),
                "move_bar_index": getattr(pos, "break_even_move_bar_index", None),
                "activation_reason": str(getattr(pos, "break_even_activation_reason", "") or ""),
                "break_even_move_reason": str(getattr(pos, "break_even_move_reason", "") or ""),
                "break_even_costs_pct": float(getattr(pos, "break_even_costs_pct", 0.0) or 0.0),
                "break_even_buffer_pct": float(getattr(pos, "break_even_buffer_pct", 0.0) or 0.0),
                "anti_spike_bars_remaining": int(
                    max(0, getattr(pos, "break_even_anti_spike_bars_remaining", 0) or 0)
                ),
                "anti_spike_consecutive_hits_required": int(
                    max(1, getattr(pos, "break_even_anti_spike_consecutive_hits_required", 2) or 2)
                ),
                "anti_spike_require_close_beyond": bool(
                    getattr(pos, "break_even_anti_spike_require_close_beyond", True)
                ),
                "initial_stop_loss": float(getattr(pos, "initial_stop_loss", 0.0) or 0.0),
                "stop_loss": float(getattr(pos, "stop_loss", 0.0) or 0.0),
                "entry_price": float(getattr(pos, "entry_price", 0.0) or 0.0),
                "partial_tp_filled": bool(getattr(pos, "partial_tp_filled", False)),
                "partial_tp_size": float(getattr(pos, "partial_tp_size", 0.0) or 0.0),
                "partial_realized_r": round(
                    float(getattr(pos, "partial_realized_r", 0.0) or 0.0),
                    6,
                ),
            }
        )
        signal_metadata["break_even"] = be_snapshot
        order_flow_md = signal_metadata.get("order_flow") if isinstance(signal_metadata, dict) else {}
        if not isinstance(order_flow_md, dict):
            order_flow_md = {}

        flow_snapshot: Dict[str, Any] = {}
        for key in (
            "signed_aggression",
            "directional_consistency",
            "imbalance_avg",
            "sweep_intensity",
            "book_pressure_avg",
            "book_pressure_trend",
            "absorption_rate",
            "delta_price_divergence",
            "delta_zscore",
            "price_change_pct",
            "delta_acceleration",
        ):
            if key in order_flow_md:
                flow_snapshot[key] = self.evidence_service.safe_float(order_flow_md.get(key), 0.0)

        l2_md = signal_metadata.get("l2_confirmation") if isinstance(signal_metadata, dict) else None
        if isinstance(l2_md, dict):
            flow_snapshot["l2_confirmation_passed"] = bool(l2_md.get("passed", False))

        trade = DayTrade(
            id=session.trade_counter,
            strategy=pos.strategy_name,
            side=pos.side,
            entry_price=pos.entry_price,
            entry_time=pos.entry_time,
            exit_price=exit_price,
            exit_time=exit_time,
            size=shares,
            pnl_pct=net_pnl_pct,
            pnl_dollars=net_pnl_dollars,
            exit_reason=reason,
            slippage=costs.get('slippage', 0.0),
            commission=costs.get('commission', 0.0),
            reg_fee=costs.get('reg_fee', 0.0),
            sec_fee=costs.get('sec_fee', 0.0),
            finra_fee=costs.get('finra_fee', 0.0),
            market_impact=costs.get('market_impact', 0.0),
            total_costs=costs.get('total', 0.0),
            gross_pnl_pct=gross_pnl_pct,
            signal_bar_index=pos.signal_bar_index,
            entry_bar_index=pos.entry_bar_index,
            signal_timestamp=pos.signal_timestamp,
            signal_price=pos.signal_price,
            signal_metadata=signal_metadata,
            flow_snapshot=flow_snapshot,
        )
        session.trades.append(trade)
        session.total_pnl += net_pnl_pct
        return trade

    def maybe_take_partial_profit(
        self,
        session: TradingSession,
        pos: Position,
        bar: BarData,
        timestamp: datetime,
    ) -> Optional[DayTrade]:
        flow = self.manager._calculate_order_flow_metrics(session.bars, lookback=min(8, len(session.bars)))
        decision = self.exit_engine.should_take_partial_profit(
            session=session,
            pos=pos,
            bar=bar,
            flow_metrics=flow,
        )
        if not decision:
            return None

        partial_price = float(decision.get("partial_price", 0.0) or 0.0)
        if partial_price > 0:
            pos.partial_take_profit_price = partial_price

        close_fraction = float(decision.get("close_fraction", 0.0) or 0.0)
        pre_partial_size = max(0.0, float(pos.size or 0.0))
        close_size = max(0.0, min(pos.size, pos.size * close_fraction))
        if close_size <= 0:
            return None

        partial_exit_price = float(decision.get("exit_price", bar.close))
        partial_realized_r = _compute_partial_realized_r(
            pos=pos,
            partial_exit_price=partial_exit_price,
            close_size=close_size,
            pre_partial_size=pre_partial_size,
        )
        pos.partial_tp_filled = True
        pos.partial_tp_size = float(close_size)
        pos.partial_realized_r = float(partial_realized_r)

        trade = self.build_trade_record(
            session=session,
            pos=pos,
            exit_price=partial_exit_price,
            exit_time=timestamp,
            reason=str(decision.get("reason", "partial_take_profit")),
            shares=close_size,
            bar_volume=bar.volume,
        )
        pos.size = max(0.0, pos.size - close_size)
        pos.partial_exit_done = True

        # After partial, protect remaining position with cost-aware break-even logic
        # — but only if MFE has reached the configurable minimum R threshold.
        _pp_min_r = max(0.0, float(
            getattr(session, "partial_protect_min_mfe_r", 0.0) or 0.0
        ))
        _should_be = True
        if _pp_min_r > 0 and pos.entry_price > 0:
            _init_sl = float(getattr(pos, "initial_stop_loss", 0.0) or 0.0)
            if _init_sl <= 0:
                _init_sl = float(getattr(pos, "stop_loss", 0.0) or 0.0)
            _risk = abs(pos.entry_price - _init_sl) if _init_sl > 0 else 0.0
            if _risk > 0:
                _side = str(getattr(pos, "side", "long")).strip().lower()
                if _side == "long":
                    _mfe = max(0.0, float(getattr(pos, "highest_price", pos.entry_price) or pos.entry_price) - pos.entry_price)
                else:
                    _mfe = max(0.0, pos.entry_price - float(getattr(pos, "lowest_price", pos.entry_price) or pos.entry_price))
                if (_mfe / _risk) < _pp_min_r:
                    _should_be = False
        if _should_be:
            self.exit_engine.force_move_to_break_even(
                session=session,
                pos=pos,
                bar=bar,
                reason="partial_take_profit_protect",
            )

        if pos.size <= 0:
            session.active_position = None
        return trade

    def should_time_exit(self, session: TradingSession, pos: Position, current_bar_index: int) -> bool:
        flow = self.manager._calculate_order_flow_metrics(session.bars, lookback=min(8, len(session.bars)))
        return self.exit_engine.should_time_exit(
            session=session,
            pos=pos,
            current_bar_index=current_bar_index,
            flow_metrics=flow,
        )

    def should_adverse_flow_exit(
        self,
        session: TradingSession,
        pos: Position,
        current_bar_index: int,
    ) -> Tuple[bool, Dict[str, float]]:
        flow = self.manager._calculate_order_flow_metrics(session.bars, lookback=min(12, len(session.bars)))
        return self.exit_engine.should_adverse_flow_exit(
            session=session,
            pos=pos,
            current_bar_index=current_bar_index,
            flow_metrics=flow,
        )

    def open_position(
        self,
        session: TradingSession,
        signal: Signal,
        entry_price: Optional[float] = None,
        entry_time: Optional[datetime] = None,
        signal_bar_index: Optional[int] = None,
        entry_bar_index: Optional[int] = None,
        entry_bar_volume: Optional[float] = None,
    ) -> Position:
        """Open a new position from signal."""
        side = 'long' if signal.signal_type == SignalType.BUY else 'short'
        fill_price = float(entry_price if entry_price is not None else signal.price)
        confirming_sources = self.extract_confirming_sources(signal)
        base_trailing_stop_pct = float(signal.trailing_stop_pct or 0.8)
        effective_trailing_stop_pct = self.effective_trailing_stop_pct(session, signal)
        effective_stop_loss = self.resolve_stop_loss_for_entry(
            session=session,
            signal=signal,
            entry_price=fill_price,
            side=side,
        )
        effective_take_profit = float(signal.take_profit) if signal.take_profit else 0.0
        fill_time = entry_time or signal.timestamp
        desired_size = self.calculate_position_size(session, signal, fill_price)
        size, fill_ratio = self.simulate_entry_fill(
            desired_size=desired_size,
            bar_volume=max(0.0, float(entry_bar_volume or 0.0)),
            session=session,
        )
        
        position = Position(
            strategy_name=signal.strategy_name,
            entry_price=fill_price,
            entry_time=fill_time,
            side=side,
            size=size,
            stop_loss=effective_stop_loss,
            take_profit=effective_take_profit,
            trailing_stop_active=signal.trailing_stop,
            highest_price=fill_price if side == 'long' else 0,
            lowest_price=fill_price if side == 'short' else float('inf'),
            fill_ratio=fill_ratio,
            initial_size=size,
            initial_stop_loss=float(effective_stop_loss if effective_stop_loss > 0 else 0.0),
        )
        # Optional audit metadata (keeps backwards compatibility).
        if signal_bar_index is not None:
            position.signal_bar_index = signal_bar_index
        if entry_bar_index is not None:
            position.entry_bar_index = entry_bar_index
        position.signal_timestamp = signal.timestamp.isoformat()
        position.signal_price = signal.price

        signal_metadata: Dict[str, Any] = (
            signal.metadata if isinstance(signal.metadata, dict) else {}
        )
        if not isinstance(signal.metadata, dict):
            signal.metadata = signal_metadata

        normalized_stop_loss_mode = self.manager.config_service.normalize_stop_loss_mode(session.stop_loss_mode)
        strategy_stop_loss = (
            round(float(signal.stop_loss), 6) if signal.stop_loss else 0.0
        )
        fixed_stop_loss_pct = round(max(0.0, float(session.fixed_stop_loss_pct)), 6)
        risk_controls = {
            "stop_loss_mode": normalized_stop_loss_mode,
            "fixed_stop_loss_pct": fixed_stop_loss_pct,
            "effective_stop_loss": round(float(effective_stop_loss), 6) if effective_stop_loss > 0 else 0.0,
            "strategy_stop_loss": strategy_stop_loss,
            "confirming_sources": confirming_sources,
            "base_trailing_stop_pct": round(base_trailing_stop_pct, 6),
            "effective_trailing_stop_pct": round(effective_trailing_stop_pct, 6),
        }
        existing_context_risk = (
            signal_metadata.get("context_risk")
            if isinstance(signal_metadata.get("context_risk"), dict)
            else None
        )
        signal_metadata["risk_controls"] = dict(risk_controls)
        signal_metadata["context_risk"] = _build_entry_context_risk_payload(
            entry_price=fill_price,
            side=side,
            effective_stop_loss=float(effective_stop_loss),
            effective_take_profit=float(effective_take_profit),
            strategy_stop_loss=float(signal.stop_loss or 0.0),
            stop_loss_mode=normalized_stop_loss_mode,
            fixed_stop_loss_pct=float(fixed_stop_loss_pct),
            existing_payload=existing_context_risk,
        )
        try:
            position.signal_metadata = json.loads(json.dumps(signal_metadata, default=str))
        except Exception:
            position.signal_metadata = dict(signal_metadata)
        position.signal_metadata["break_even"] = {
            "active": False,
            "state": "idle",
            "arm_bar_index": None,
            "move_bar_index": None,
            "activation_reason": "",
            "break_even_move_reason": "",
            "break_even_costs_pct": 0.0,
            "break_even_buffer_pct": 0.0,
            "anti_spike_bars_remaining": 0,
            "anti_spike_consecutive_hits_required": 2,
            "anti_spike_require_close_beyond": True,
            "initial_stop_loss": float(position.initial_stop_loss or 0.0),
            "stop_loss": float(position.stop_loss or 0.0),
            "entry_price": float(position.entry_price or 0.0),
            "partial_tp_filled": False,
            "partial_tp_size": 0.0,
            "partial_realized_r": 0.0,
        }
        if session.enable_partial_take_profit and position.stop_loss > 0:
            position.partial_take_profit_price = self.partial_take_profit_price(session, position)
        
        session.active_position = position
        session.trailing_stop_pct = effective_trailing_stop_pct

        # ── Position Context Monitor: snapshot entry conditions ──
        flow_at_entry = self.manager._calculate_order_flow_metrics(session.bars, lookback=min(20, len(session.bars)))
        indicators_at_entry = self.manager._calculate_indicators(
            session.bars[-100:] if len(session.bars) >= 100 else session.bars,
            session=session,
        )
        atr_at_entry = float(self.manager._latest_indicator_value(indicators_at_entry, "atr") or 0.0)

        ticker_cfg = self.ticker_params.get(session.ticker.upper(), {})
        context_config = ticker_cfg.get("adaptive", {}).get("context_exit_response", {})

        snapshot = EntrySnapshot(
            macro_regime=(session.detected_regime.value if session.detected_regime else "MIXED"),
            micro_regime=(session.micro_regime or "MIXED"),
            strategy_name=signal.strategy_name,
            flow_score=float(flow_at_entry.get("flow_score", 0.0) or 0.0),
            signed_aggression=float(flow_at_entry.get("signed_aggression", 0.0) or 0.0),
            book_pressure=float(flow_at_entry.get("book_pressure_avg", 0.0) or 0.0),
            directional_consistency=float(flow_at_entry.get("directional_consistency", 0.0) or 0.0),
            delta_acceleration=float(flow_at_entry.get("delta_acceleration", 0.0) or 0.0),
            entry_bar_index=entry_bar_index or 0,
            entry_price=fill_price,
            side=side,
            atr_at_entry=atr_at_entry,
            volatility_at_entry=float(flow_at_entry.get("realized_volatility_pct", 0.0) or 0.0),
        )
        session._position_context = PositionContextMonitor(snapshot, context_config)
        session._context_exit_policy = ContextAwareExitPolicy(context_config)

        return position

    def close_position(
        self,
        session: TradingSession,
        exit_price: float,
        exit_time: datetime,
        reason: str,
        bar_volume: Optional[float] = None,
    ) -> DayTrade:
        """Close position and record trade with trading costs."""
        pos = session.active_position
        trade = self.build_trade_record(
            session=session,
            pos=pos,
            exit_price=exit_price,
            exit_time=exit_time,
            reason=reason,
            shares=pos.size,
            bar_volume=bar_volume,
        )

        # Attach position context summary to the trade's flow_snapshot
        if session._position_context:
            context_summary = session._position_context.get_summary()
            if session._context_exit_policy:
                context_summary["exit_responses"] = session._context_exit_policy.get_applied_summary()
            if hasattr(trade, "flow_snapshot") and isinstance(trade.flow_snapshot, dict):
                trade.flow_snapshot["position_context"] = context_summary
            session._position_context = None
            session._context_exit_policy = None

        session.active_position = None

        # Consecutive-loss guardrail: after N losses, pause new entries for X bars.
        if trade.pnl_dollars < 0:
            session.consecutive_losses = int(session.consecutive_losses) + 1
        else:
            session.consecutive_losses = 0

        loss_limit = max(1, int(getattr(self.manager, "consecutive_loss_limit", 3)))
        cooldown_bars = max(0, int(getattr(self.manager, "consecutive_loss_cooldown_bars", 0)))
        if trade.pnl_dollars < 0 and session.consecutive_losses >= loss_limit and cooldown_bars > 0:
            current_bar_index = max(0, len(session.bars) - 1)
            session.loss_cooldown_until_bar_index = max(
                int(session.loss_cooldown_until_bar_index),
                int(current_bar_index + cooldown_bars),
            )

        # Record trade outcome for calibration + edge monitoring
        orch = session.orchestrator
        if orch:
            was_profitable = trade.pnl_pct > 0
            regime_label = (session.detected_regime.value
                            if session.detected_regime else 'MIXED')
            stop_dist = (abs(pos.entry_price - pos.stop_loss)
                         if pos.stop_loss else (pos.entry_price * 0.005))
            pnl_r = ((trade.pnl_pct / 100 * pos.entry_price) / stop_dist
                      if stop_dist > 0 else 0.0)
            signal_md = (pos.signal_metadata
                         if isinstance(pos.signal_metadata, dict) else {})
            raw_conf = self.evidence_service.extract_raw_confidence_from_metadata(signal_md)
            confirming_source_keys = self.manager._extract_confirming_source_keys_from_metadata(
                signal_metadata=signal_md,
                side=pos.side,
                strategy_name=pos.strategy_name,
            )

            orch.record_trade_outcome(
                strategy=pos.strategy_name,
                regime=regime_label,
                raw_confidence=raw_conf,
                was_profitable=was_profitable,
                pnl_r=pnl_r,
                bar_index=len(session.bars) - 1,
                confirming_sources=confirming_source_keys,
            )

        return trade
