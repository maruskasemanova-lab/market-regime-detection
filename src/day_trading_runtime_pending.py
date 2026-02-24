"""Pending-signal execution helpers for runtime processing."""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, Callable, Dict

from .context_aware_risk import ContextRiskConfig, adjust_entry_risk
from .day_trading_models import BarData, TradingSession
from .day_trading_runtime_intrabar import (
    intrabar_confirmation_snapshot,
    micro_confirmation_snapshot,
)
from .day_trading_runtime_portfolio import cooldown_bars_remaining
from .day_trading_runtime_sweep import to_optional_float
from .strategies.base_strategy import SignalType
from .trading_config import TradingConfig

logger = logging.getLogger(__name__)


def process_pending_signal_entry(
    *,
    manager: Any,
    session: TradingSession,
    bar: BarData,
    timestamp: datetime,
    current_bar_index: int,
    result: Dict[str, Any],
    formula_indicators: Callable[[], Dict[str, Any]],
) -> bool:
    """
    Try to execute pending signal at current bar open.

    Returns True when caller should return immediately from runtime bar processing.
    """
    if not (session.pending_signal and not session.active_position):
        return False

    pending_signal_age = current_bar_index - int(session.pending_signal_bar_index)
    pending_ttl_bars = max(1, int(getattr(manager, "pending_signal_ttl_bars", 3)))
    cooldown_remaining = cooldown_bars_remaining(session, current_bar_index)
    micro_regime = str(session.micro_regime or "").upper()

    sig_name = session.pending_signal.strategy_name

    if pending_signal_age > pending_ttl_bars or session.pending_signal_bar_index < 0:
        logger.debug(
            "Pending signal dropped as stale: age=%s ttl=%s strategy=%s",
            pending_signal_age,
            pending_ttl_bars,
            sig_name,
        )
        session.pending_signal = None
        session.pending_signal_bar_index = -1
        result["stale_pending_signal_dropped"] = True
        result["stale_pending_signal_age_bars"] = max(0, pending_signal_age)
        result["pending_signal_ttl_bars"] = pending_ttl_bars
        return False

    if cooldown_remaining > 0:
        logger.debug(
            "Pending signal dropped by cooldown: remaining=%s strategy=%s",
            cooldown_remaining,
            sig_name,
        )
        session.pending_signal = None
        session.pending_signal_bar_index = -1
        result["dropped_pending_signal"] = True
        result["action"] = "consecutive_loss_cooldown"
        result["reason"] = f"Consecutive-loss cooldown: {cooldown_remaining} bars remaining"
        result["cooldown_bars_remaining"] = cooldown_remaining
        return True

    if micro_regime == "UNKNOWN":
        logger.debug(
            "Micro regime UNKNOWN during pending entry evaluation: strategy=%s",
            sig_name,
        )
        result["regime_warmup"] = True
        result["regime_warmup_reason"] = "ADX warmup incomplete; applying standard confirmation gates."

    signal = session.pending_signal
    if not isinstance(signal.metadata, dict):
        signal.metadata = {}
    config = (
        session.config
        if isinstance(getattr(session, "config", None), TradingConfig)
        else TradingConfig()
    )
    # EV relaxation: read once, used for micro + intrabar.
    ev_margin_val = float(signal.metadata.get("ev_margin", 0.0) or 0.0)
    ev_relax_enabled = bool(getattr(config, "ev_relaxation_enabled", False))
    ev_relax_threshold = float(getattr(config, "ev_relaxation_threshold", 10.0))
    ev_relax_factor = float(getattr(config, "ev_relaxation_factor", 0.50))
    ev_relaxation_active = ev_relax_enabled and ev_margin_val >= ev_relax_threshold

    if bool(getattr(config, "intraday_levels_micro_confirmation_enabled", False)):
        level_context = (
            signal.metadata.get("level_context")
            if isinstance(signal.metadata, dict)
            and isinstance(signal.metadata.get("level_context"), dict)
            else {}
        )
        level_reason = str(level_context.get("reason") or "").strip().lower()
        is_liquidity_sweep_signal = level_reason == "liquidity_sweep_confirmed"
        sweep_override_enabled = bool(
            getattr(
                config,
                "intraday_levels_micro_confirmation_disable_for_sweep",
                False,
            )
        )
        if is_liquidity_sweep_signal and sweep_override_enabled:
            required_micro_bars = max(
                0,
                int(
                    getattr(
                        config,
                        "intraday_levels_micro_confirmation_sweep_bars",
                        0,
                    )
                ),
            )
        else:
            required_micro_bars = max(
                1,
                int(
                    getattr(
                        config,
                        "intraday_levels_micro_confirmation_bars",
                        2,
                    )
                ),
            )
        if ev_relaxation_active and required_micro_bars > 1:
            required_micro_bars = max(1, required_micro_bars - 1)

        micro_mode = str(
            getattr(config, "micro_confirmation_mode", "consecutive_close")
        ).strip().lower()
        micro_vol_delta_min = float(
            getattr(config, "micro_confirmation_volume_delta_min_pct", 0.60)
        )
        # MR bounce forms gradually; the first bar after reversal may still
        # close red.  Use volume_delta (1 bar directional flow) instead of
        # strictly monotonic consecutive closes.
        _pending_strategy = str(
            getattr(signal, "strategy_name", "") or ""
        ).strip().lower()
        _MR_BOUNCE_STRATS = {"mean_reversion", "absorption_reversal", "rotation"}
        if micro_mode == "consecutive_close" and _pending_strategy in _MR_BOUNCE_STRATS:
            micro_mode = "volume_delta"
        micro_confirmation = micro_confirmation_snapshot(
            session=session,
            signal=signal,
            current_bar_index=current_bar_index,
            signal_bar_index=session.pending_signal_bar_index,
            required_bars=required_micro_bars,
            mode=micro_mode,
            volume_delta_min_pct=micro_vol_delta_min,
        )
        result["micro_confirmation"] = micro_confirmation
        signal.metadata.setdefault("micro_confirmation", {}).update(micro_confirmation)
        if not micro_confirmation.get("ready", False):
            result["action"] = "pending_micro_confirmation"
            result["reason"] = "Awaiting consecutive close confirmation."
            return True
        if not micro_confirmation.get("passed", False):
            session.pending_signal = None
            session.pending_signal_bar_index = -1
            result["dropped_pending_signal"] = True
            result["action"] = "micro_confirmation_failed"
            result["reason"] = "Consecutive close confirmation failed."
            return True

        if bool(
            getattr(
                config,
                "intraday_levels_micro_confirmation_require_intrabar",
                False,
            )
        ):
            intrabar_min_coverage = max(
                0,
                int(
                    getattr(
                        config,
                        "intraday_levels_micro_confirmation_intrabar_min_coverage_points",
                        3,
                    )
                ),
            )
            intrabar_min_move = max(
                0.0,
                float(
                    getattr(
                        config,
                        "intraday_levels_micro_confirmation_intrabar_min_move_pct",
                        0.02,
                    )
                    or 0.0
                ),
            )
            intrabar_min_push = max(
                0.0,
                min(
                    1.0,
                    float(
                        getattr(
                            config,
                            "intraday_levels_micro_confirmation_intrabar_min_push_ratio",
                            0.10,
                        )
                        or 0.0
                    ),
                ),
            )
            if ev_relaxation_active:
                intrabar_min_coverage = max(
                    1,
                    int(intrabar_min_coverage * ev_relax_factor),
                )
                intrabar_min_move *= ev_relax_factor
                intrabar_min_push *= ev_relax_factor
                result["ev_relaxation"] = {
                    "active": True,
                    "ev_margin": ev_margin_val,
                    "factor": ev_relax_factor,
                }

            intrabar_confirmation = intrabar_confirmation_snapshot(
                session=session,
                signal=signal,
                current_bar_index=current_bar_index,
                signal_bar_index=session.pending_signal_bar_index,
                window_seconds=max(
                    1,
                    min(
                        60,
                        int(
                            getattr(
                                config,
                                "intraday_levels_micro_confirmation_intrabar_window_seconds",
                                5,
                            )
                        ),
                    ),
                ),
                min_coverage_points=intrabar_min_coverage,
                min_move_pct=intrabar_min_move,
                min_push_ratio=intrabar_min_push,
                max_spread_bps=max(
                    0.0,
                    float(
                        getattr(
                            config,
                            "intraday_levels_micro_confirmation_intrabar_max_spread_bps",
                            12.0,
                        )
                        or 0.0
                    ),
                ),
            )
            result["intrabar_confirmation"] = intrabar_confirmation
            signal.metadata.setdefault("micro_confirmation", {}).setdefault(
                "intrabar",
                {},
            ).update(intrabar_confirmation)
            if not intrabar_confirmation.get("ready", False):
                result["action"] = "pending_intrabar_confirmation"
                result["reason"] = "Awaiting intrabar flow confirmation."
                return True
            if not intrabar_confirmation.get("passed", False):
                session.pending_signal = None
                session.pending_signal_bar_index = -1
                result["dropped_pending_signal"] = True
                result["action"] = "intrabar_confirmation_failed"
                result["reason"] = "Intrabar confirmation gate failed."
                return True

    ctx_cfg = ContextRiskConfig.from_config_obj(config)
    if bool(ctx_cfg.enabled):
        levels_payload = {}
        md_levels = signal.metadata.get("intraday_levels_payload")
        if isinstance(md_levels, dict):
            levels_payload = dict(md_levels)
        if not levels_payload:
            levels_payload = manager._intraday_levels_indicator_payload(session)

        atr_for_context = (
            to_optional_float(
                manager._latest_indicator_value(
                    formula_indicators(),
                    "atr",
                    session.bars,
                )
            )
            or 0.0
        )
        risk_adjustment = adjust_entry_risk(
            entry_price=float(bar.open),
            side="long" if signal.signal_type == SignalType.BUY else "short",
            original_stop_loss=float(signal.stop_loss or 0.0),
            original_take_profit=float(signal.take_profit or 0.0),
            levels_payload=levels_payload,
            config=ctx_cfg,
            atr=atr_for_context,
            is_sweep_trade=bool(
                isinstance(signal.metadata, dict)
                and signal.metadata.get("sweep_triggered", False)
            ),
            strategy_key=str(signal.strategy_name or ""),
        )
        signal.metadata["context_risk"] = dict(risk_adjustment)
        result["context_risk"] = dict(risk_adjustment)
        if risk_adjustment.get("skip", False):
            session.pending_signal = None
            session.pending_signal_bar_index = -1
            result["dropped_pending_signal"] = True
            result["action"] = "context_risk_skip"
            result["reason"] = str(
                risk_adjustment.get("skip_reason", "context_risk_skip")
            )
            return True

        adjusted_sl = to_optional_float(risk_adjustment.get("adjusted_stop_loss"))
        adjusted_tp = to_optional_float(risk_adjustment.get("adjusted_take_profit"))
        if adjusted_sl is not None and adjusted_sl > 0.0:
            signal.stop_loss = float(adjusted_sl)
        if adjusted_tp is not None and adjusted_tp > 0.0:
            signal.take_profit = float(adjusted_tp)

    position = manager._open_position(
        session,
        signal,
        entry_price=bar.open,
        entry_time=timestamp,
        signal_bar_index=session.pending_signal_bar_index,
        entry_bar_index=current_bar_index,
        entry_bar_volume=bar.volume,
    )
    session.pending_signal = None
    session.pending_signal_bar_index = -1

    if position.size > 0:
        session_key = manager._get_session_key(session.run_id, session.ticker, session.date)
        manager.last_trade_bar_index[session_key] = current_bar_index
        result["action"] = "position_opened"
        result["position"] = position.to_dict()
        result["position_opened"] = {
            "entry_price": position.entry_price,
            "side": position.side,
            "strategy": position.strategy_name,
            "size": position.size,
            "fill_ratio": position.fill_ratio,
            "stop_loss": position.stop_loss,
            "take_profit": position.take_profit,
            "partial_take_profit_price": position.partial_take_profit_price,
            "reasoning": signal.reasoning,
            "confidence": signal.confidence,
            "metadata": (
                dict(position.signal_metadata)
                if isinstance(getattr(position, "signal_metadata", None), dict)
                else (
                    dict(signal.metadata)
                    if isinstance(getattr(signal, "metadata", None), dict)
                    else {}
                )
            ),
        }
        golden_setup_md = (
            signal.metadata.get("golden_setup")
            if isinstance(getattr(signal, "metadata", None), dict)
            else None
        )
        if isinstance(golden_setup_md, dict) and bool(golden_setup_md.get("applied", False)):
            session.golden_setup_entries_today = int(
                getattr(session, "golden_setup_entries_today", 0) or 0
            ) + 1
            session.golden_setup_last_entry_bar_index = int(current_bar_index)
            if isinstance(getattr(session, "golden_setup_result", None), dict):
                session.golden_setup_result["entries_today"] = int(
                    session.golden_setup_entries_today
                )
            result["golden_setup_entry"] = {
                "count_today": int(session.golden_setup_entries_today),
                "bar_index": int(current_bar_index),
                "setup": golden_setup_md.get("best_setup"),
                "direction": golden_setup_md.get("best_direction"),
                "confidence_boost": golden_setup_md.get("applied_confidence_boost", 0.0),
                "threshold_relief": golden_setup_md.get("applied_threshold_relief", 0.0),
            }
        if isinstance(position.signal_metadata, dict) and isinstance(
            position.signal_metadata.get("break_even"),
            dict,
        ):
            result["break_even"] = dict(position.signal_metadata.get("break_even") or {})
    else:
        session.active_position = None
        result["action"] = "insufficient_fill"
        result["reason"] = "Position size after risk/fill constraints is zero."

    return False
