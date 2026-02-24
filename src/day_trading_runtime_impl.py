"""Runtime method facade used by DayTradingManager."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
import logging

from .day_trading_models import BarData, TradingSession
from .trading_config import TradingConfig
from .strategies.base_strategy import Signal
from .day_trading_runtime_intrabar import (
    intrabar_confirmation_snapshot as _intrabar_confirmation_snapshot_impl,
    micro_confirmation_snapshot as _micro_confirmation_snapshot_impl,
)
from .day_trading_runtime_sweep import (
    runtime_detect_liquidity_sweep as _runtime_detect_liquidity_sweep_impl,
)
from .day_trading_runtime_entry_quality import (
    runtime_evaluate_intraday_levels_entry_quality as _runtime_evaluate_intraday_levels_entry_quality_impl,
)
from .day_trading_runtime.signal_generation import (
    runtime_calculate_indicators as _runtime_calculate_indicators_impl,
    runtime_generate_signal as _runtime_generate_signal_impl,
)
from .day_trading_runtime.intrabar_slice import (
    runtime_evaluate_intrabar_slice as _runtime_evaluate_intrabar_slice_impl,
)
from .day_trading_runtime.bar_processing import (
    runtime_process_bar as _runtime_process_bar_impl,
)
from .day_trading_runtime.intrabar_trace import (
    attach_intrabar_eval_trace as _attach_intrabar_eval_trace_impl,
)
from .day_trading_runtime_trading_bar import (
    runtime_process_trading_bar as _runtime_process_trading_bar_impl,
)

logger = logging.getLogger(__name__)


def _micro_confirmation_snapshot(
    *,
    session: TradingSession,
    signal: Signal,
    current_bar_index: int,
    signal_bar_index: int,
    required_bars: int,
    mode: str = "consecutive_close",
    volume_delta_min_pct: float = 0.60,
) -> Dict[str, Any]:
    return _micro_confirmation_snapshot_impl(
        session=session,
        signal=signal,
        current_bar_index=current_bar_index,
        signal_bar_index=signal_bar_index,
        required_bars=required_bars,
        mode=mode,
        volume_delta_min_pct=volume_delta_min_pct,
    )


def _intrabar_confirmation_snapshot(
    *,
    session: TradingSession,
    signal: Signal,
    current_bar_index: int,
    signal_bar_index: int,
    window_seconds: int,
    min_coverage_points: int,
    min_move_pct: float,
    min_push_ratio: float,
    max_spread_bps: float,
) -> Dict[str, Any]:
    return _intrabar_confirmation_snapshot_impl(
        session=session,
        signal=signal,
        current_bar_index=current_bar_index,
        signal_bar_index=signal_bar_index,
        window_seconds=window_seconds,
        min_coverage_points=min_coverage_points,
        min_move_pct=min_move_pct,
        min_push_ratio=min_push_ratio,
        max_spread_bps=max_spread_bps,
    )


def runtime_detect_liquidity_sweep(
    self,
    *,
    session: TradingSession,
    current_price: float,
    fv: Any,
    flow_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = (
        session.config
        if isinstance(getattr(session, "config", None), TradingConfig)
        else TradingConfig()
    )
    logger.debug(
        "runtime_detect_liquidity_sweep entered; enabled=%s",
        bool(getattr(config, "liquidity_sweep_detection_enabled", False)),
    )
    return _runtime_detect_liquidity_sweep_impl(
        session=session,
        current_price=current_price,
        fv=fv,
        flow_metrics=flow_metrics,
    )


def runtime_evaluate_intraday_levels_entry_quality(
    self,
    *,
    session: TradingSession,
    signal: Signal,
    current_price: float,
    current_bar_index: int,
) -> Dict[str, Any]:
    return _runtime_evaluate_intraday_levels_entry_quality_impl(
        self=self,
        session=session,
        signal=signal,
        current_price=current_price,
        current_bar_index=current_bar_index,
    )


def runtime_process_bar(
    self,
    run_id: str,
    ticker: str,
    timestamp: datetime,
    bar_data: Dict[str, Any],
    warmup_only: bool = False,
) -> Dict[str, Any]:
    return _runtime_process_bar_impl(
        self=self,
        run_id=run_id,
        ticker=ticker,
        timestamp=timestamp,
        bar_data=bar_data,
        warmup_only=warmup_only,
    )


def runtime_attach_intrabar_eval_trace(
    *,
    timestamp: datetime,
    bar_data: Dict[str, Any],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    return _attach_intrabar_eval_trace_impl(
        timestamp=timestamp,
        bar_data=bar_data,
        result=result,
    )


def runtime_process_trading_bar(
    self,
    session: TradingSession,
    bar: BarData,
    timestamp: datetime,
    warmup_only: bool = False,
) -> Dict[str, Any]:
    return _runtime_process_trading_bar_impl(
        self=self,
        session=session,
        bar=bar,
        timestamp=timestamp,
        warmup_only=warmup_only,
    )


def runtime_generate_signal(
    self,
    session: TradingSession,
    bar: BarData,
    timestamp: datetime,
) -> Optional[Signal]:
    return _runtime_generate_signal_impl(
        self=self,
        session=session,
        bar=bar,
        timestamp=timestamp,
    )


def runtime_calculate_indicators(
    self,
    bars: List[BarData],
    session: Optional[TradingSession] = None,
) -> Dict[str, Any]:
    return _runtime_calculate_indicators_impl(
        self=self,
        bars=bars,
        session=session,
    )


def runtime_evaluate_intrabar_slice(
    self,
    session: TradingSession,
    bar: BarData,
    timestamp: datetime,
) -> Dict[str, Any]:
    return _runtime_evaluate_intrabar_slice_impl(
        self=self,
        session=session,
        bar=bar,
        timestamp=timestamp,
    )
