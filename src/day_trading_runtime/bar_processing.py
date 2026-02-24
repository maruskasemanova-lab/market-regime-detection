"""Bar-level runtime processing helper extracted from runtime implementation."""

from __future__ import annotations

from datetime import datetime, time
from typing import Any, Dict

from ..day_trading_models import BarData, SessionPhase
from ..trading_config import TradingConfig


def runtime_process_bar(
    self,
    run_id: str,
    ticker: str,
    timestamp: datetime,
    bar_data: Dict[str, Any],
    warmup_only: bool = False,
) -> Dict[str, Any]:
    """
    Process a new bar for the session.

    Args:
        run_id: Unique run identifier
        ticker: Stock ticker
        timestamp: Bar timestamp
        bar_data: Dict with open, high, low, close, volume, vwap (optional)

    Returns:
        Processing result with signals, trades, etc.
    """

    market_ts = self._to_market_time(timestamp)
    date = market_ts.strftime("%Y-%m-%d")
    defaults = self.run_defaults.get((run_id, ticker), {})
    if isinstance(defaults, TradingConfig):
        defaults = defaults.to_session_params()
    if not isinstance(defaults, dict):
        defaults = {}
    session = self.get_or_create_session(
        run_id,
        ticker,
        date,
        regime_detection_minutes=defaults.get("regime_detection_minutes"),
        cold_start_each_day=bool(defaults.get("cold_start_each_day", False)),
    )
    if defaults:
        merged_cfg = self.config_service.canonical_trading_config(
            {**session.config.to_session_params(), **defaults}
        )
        self._apply_trading_config_to_session(
            session,
            merged_cfg,
            normalize_momentum=("momentum_diversification" in defaults),
        )
    if "momentum_diversification" not in defaults and session.momentum_diversification_override:
        session.momentum_diversification = {}
        session.momentum_diversification_override = False
    if session.phase == SessionPhase.CLOSED:
        return {"action": "skipped_finished_session"}

    bar = BarData(
        timestamp=timestamp,
        open=bar_data.get("open", 0),
        high=bar_data.get("high", 0),
        low=bar_data.get("low", 0),
        close=bar_data.get("close", 0),
        volume=bar_data.get("volume", 0),
        vwap=bar_data.get("vwap"),
        l2_delta=bar_data.get("l2_delta"),
        l2_buy_volume=bar_data.get("l2_buy_volume"),
        l2_sell_volume=bar_data.get("l2_sell_volume"),
        l2_volume=bar_data.get("l2_volume"),
        l2_imbalance=bar_data.get("l2_imbalance"),
        l2_bid_depth_total=bar_data.get("l2_bid_depth_total"),
        l2_ask_depth_total=bar_data.get("l2_ask_depth_total"),
        l2_book_pressure=bar_data.get("l2_book_pressure"),
        l2_book_pressure_change=bar_data.get("l2_book_pressure_change"),
        l2_iceberg_buy_count=bar_data.get("l2_iceberg_buy_count"),
        l2_iceberg_sell_count=bar_data.get("l2_iceberg_sell_count"),
        l2_iceberg_bias=bar_data.get("l2_iceberg_bias"),
        l2_quality_flags=bar_data.get("l2_quality_flags"),
        l2_quality=bar_data.get("l2_quality"),
        intrabar_quotes_1s=bar_data.get("intrabar_quotes_1s"),
        tcbbo_net_premium=bar_data.get("tcbbo_net_premium"),
        tcbbo_cumulative_net_premium=bar_data.get("tcbbo_cumulative_net_premium"),
        tcbbo_call_buy_premium=bar_data.get("tcbbo_call_buy_premium"),
        tcbbo_put_buy_premium=bar_data.get("tcbbo_put_buy_premium"),
        tcbbo_call_sell_premium=bar_data.get("tcbbo_call_sell_premium"),
        tcbbo_put_sell_premium=bar_data.get("tcbbo_put_sell_premium"),
        tcbbo_sweep_count=bar_data.get("tcbbo_sweep_count"),
        tcbbo_sweep_premium=bar_data.get("tcbbo_sweep_premium"),
        tcbbo_trade_count=bar_data.get("tcbbo_trade_count"),
        tcbbo_has_data=bar_data.get("tcbbo_has_data"),
    )

    # Orchestrator update is DEFERRED in TRADING phase to avoid look-ahead
    # bias: intrabar checkpoints must not see full-minute bar data.
    # Non-TRADING phases update immediately since no intrabar eval occurs.
    _is_trading_phase = session.phase == SessionPhase.TRADING
    if (
        not _is_trading_phase
        and session.orchestrator
        and session.orchestrator.config.use_evidence_engine
    ):
        session.orchestrator.update_bar(bar_data)

    bar_time = market_ts.time()
    result = {
        "session_key": self._get_session_key(run_id, ticker, date),
        "bar_timestamp": timestamp.isoformat(),
        "bar_price": bar.close,
        "phase": session.phase.value,
        "action": None,
        "signal": None,
        "trade_closed": None,
        "regime": session.detected_regime.value if session.detected_regime else None,
        "micro_regime": session.micro_regime,
        "strategy": session.selected_strategy,
    }

    is_premarket_time = bar_time < session.market_open
    if is_premarket_time and not session.premarket_trading_enabled:
        session.pre_market_bars.append(bar)
        session.phase = SessionPhase.PRE_MARKET
        result["action"] = "stored_pre_market_bar"

    elif session.phase == SessionPhase.PRE_MARKET:
        session.phase = SessionPhase.REGIME_DETECTION
        if is_premarket_time:
            session.pre_market_bars.append(bar)
        session.bars.append(bar)
        result["intraday_levels"] = self._update_intraday_levels(
            session,
            len(session.bars) - 1,
        )
        session.start_price = bar.close
        session.regime_start_ts = datetime.combine(
            market_ts.date(),
            bar_time if is_premarket_time else session.market_open,
            tzinfo=self.market_tz,
        )
        result["action"] = "started_regime_detection"

    elif session.phase == SessionPhase.REGIME_DETECTION:
        session.bars.append(bar)
        result["intraday_levels"] = self._update_intraday_levels(
            session,
            len(session.bars) - 1,
        )

        if session.regime_start_ts is None:
            session.regime_start_ts = datetime.combine(
                market_ts.date(),
                session.market_open,
                tzinfo=self.market_tz,
            )
        elapsed_minutes = (market_ts - session.regime_start_ts).total_seconds() / 60.0
        effective_regime_minutes = session.regime_detection_minutes
        if self.gate_engine.bar_has_l2_data(bar):
            effective_regime_minutes = min(effective_regime_minutes, 10)
        elif len(session.bars) >= 3:
            flow_probe = self._calculate_order_flow_metrics(
                session.bars,
                lookback=min(10, len(session.bars)),
            )
            if flow_probe.get("has_l2_coverage", False):
                effective_regime_minutes = min(effective_regime_minutes, 10)

        if elapsed_minutes >= effective_regime_minutes:
            regime = self._detect_regime(session)
            session.detected_regime = regime

            active_strategies = self._select_strategies(session)
            session.active_strategies = active_strategies
            session.selected_strategy = (
                "adaptive"
                if len(active_strategies) > 1
                else (active_strategies[0] if active_strategies else None)
            )
            session.last_regime_refresh_bar_index = max(0, len(session.bars) - 1)
            session.regime_history.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "bar_index": session.last_regime_refresh_bar_index,
                    "regime": regime.value,
                    "micro_regime": session.micro_regime,
                    "strategies": list(active_strategies),
                    "selection_warnings": list(getattr(session, "selection_warnings", [])),
                }
            )
            session.last_strategy_switch_bar_index = session.last_regime_refresh_bar_index

            session.phase = SessionPhase.TRADING
            result["action"] = "regime_detected"
            result["regime"] = regime.value
            result["micro_regime"] = session.micro_regime
            result["strategies"] = active_strategies
            result["strategy"] = session.selected_strategy
            result["selection_warnings"] = list(getattr(session, "selection_warnings", []))

            indicators = self._calculate_indicators(session.bars, session=session)
            flow = indicators.get("order_flow") or {}
            result["indicators"] = {
                "trend_efficiency": self._calc_trend_efficiency(session.bars),
                "volatility": self._calc_volatility(session.bars),
                "atr": self._latest_indicator_value(indicators, "atr", session.bars),
                "adx": self._latest_indicator_value(indicators, "adx"),
                "flow_score": float(flow.get("flow_score", 0.0) or 0.0),
                "signed_aggression": float(flow.get("signed_aggression", 0.0) or 0.0),
                "absorption_rate": float(flow.get("absorption_rate", 0.0) or 0.0),
                "book_pressure_avg": float(flow.get("book_pressure_avg", 0.0) or 0.0),
                "book_pressure_trend": float(flow.get("book_pressure_trend", 0.0) or 0.0),
                "large_trader_activity": float(flow.get("large_trader_activity", 0.0) or 0.0),
                "vwap_execution_flow": float(flow.get("vwap_execution_flow", 0.0) or 0.0),
                "intraday_levels": self._intraday_levels_indicator_payload(session),
            }
        else:
            result["action"] = "collecting_regime_data"
            minutes_elapsed = int(elapsed_minutes)
            result["minutes_remaining"] = max(0, effective_regime_minutes - minutes_elapsed)

    elif session.phase == SessionPhase.TRADING:
        session.bars.append(bar)
        # DEFER intraday levels update: checkpoints must see previous-bar state.
        # Save pre-bar feature vector so checkpoints inherit previous-bar L2.
        session._pre_bar_fv = (
            session.orchestrator.current_feature_vector
            if session.orchestrator
            else None
        )

        if bar_time >= session.market_close or bar_time >= time(15, 55):
            # End-of-day: update orchestrator + levels before closing position
            if session.orchestrator and session.orchestrator.config.use_evidence_engine:
                session.orchestrator.update_bar(bar_data)
            result["intraday_levels"] = self._update_intraday_levels(
                session,
                len(session.bars) - 1,
            )
            if session.active_position:
                trade = self._close_position(
                    session,
                    bar.close,
                    timestamp,
                    "end_of_day",
                    bar_volume=bar.volume,
                )
                result["trade_closed"] = trade.to_dict()
                bars_held = len(
                    [
                        b
                        for b in session.bars
                        if b.timestamp >= trade.entry_time and b.timestamp <= trade.exit_time
                    ]
                )
                result["position_closed"] = self.gate_engine.build_position_closed_payload(
                    trade=trade,
                    exit_reason="end_of_day",
                    bars_held=bars_held,
                )

            session.phase = SessionPhase.END_OF_DAY
            session.end_price = bar.close
            self._persist_intraday_levels_memory(session)
            result["action"] = "session_ended"
            result["session_summary"] = self._get_session_summary(session)
        else:
            trade_result = self._process_trading_bar(
                session,
                bar,
                timestamp,
                warmup_only=warmup_only,
            )
            result.update(trade_result)

            # POST-BAR: update orchestrator and intraday levels with full bar
            # now that intrabar checkpoint evaluation is done.
            if session.orchestrator and session.orchestrator.config.use_evidence_engine:
                session.orchestrator.update_bar(bar_data)
            result["intraday_levels"] = self._update_intraday_levels(
                session,
                len(session.bars) - 1,
            )

        # Clean up temporary pre-bar FV reference
        session._pre_bar_fv = None

    elif session.phase == SessionPhase.END_OF_DAY:
        session.end_price = bar.close
        result["action"] = "session_already_closed"

    if "intraday_levels" not in result:
        result["intraday_levels"] = self._get_intraday_levels_snapshot(session)
    result["phase"] = session.phase.value
    result["micro_regime"] = session.micro_regime
    return result
