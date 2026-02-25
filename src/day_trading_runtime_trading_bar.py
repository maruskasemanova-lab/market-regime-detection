"""Runtime method implementations extracted from DayTradingManager."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
import logging

from .day_trading_models import BarData, TradingSession
from .day_trading_runtime_trading_bar_helpers.session_guards import (
    _apply_max_daily_loss_circuit_breaker,
    _apply_regime_update_payload,
    _build_default_liquidity_sweep_payload,
    _handle_portfolio_drawdown_halt,
    _handle_warmup_only_mode,
)
from .day_trading_runtime_trading_bar_helpers.intrabar_trace import (
    _attach_active_position_intrabar_trace,
    _evaluate_intrabar_checkpoint_trace,
    _resolve_intrabar_checkpoint_meta,
)
from .day_trading_runtime_trading_bar_helpers.entry_actions import (
    # imported for backward-compatible monkey patches in tests
    _apply_threshold_rejection_payload,
)
from .day_trading_runtime_trading_bar_helpers.decision_context import (
    _apply_liquidity_sweep_detection_and_confirmation,  # imported for test patch compat
)
from .day_trading_runtime_trading_bar_helpers.entry_pipeline import (
    _process_entry_signal_generation_pipeline,
)
from .day_trading_runtime_intrabar import (
    calculate_intrabar_1s_snapshot as _calculate_intrabar_1s_snapshot_impl,  # compat test patch paths
)
from .day_trading_runtime_portfolio import (
    portfolio_drawdown_snapshot as _portfolio_drawdown_snapshot_impl,
)
from .day_trading_runtime_sweep import to_optional_float as _to_optional_float_impl
from .day_trading_runtime_pending import (
    process_pending_signal_entry as _process_pending_signal_entry_impl,
)
from .day_trading_runtime_position import (
    manage_active_position_lifecycle as _manage_active_position_lifecycle_impl,
)
from .day_trading_runtime.intrabar_slice import (
    runtime_evaluate_intrabar_slice as _runtime_evaluate_intrabar_slice_impl,
)

logger = logging.getLogger(__name__)

def runtime_process_trading_bar(
    self, 
    session: TradingSession, 
    bar: BarData, 
    timestamp: datetime,
    warmup_only: bool = False,
) -> Dict[str, Any]:
    """Process a trading bar - manage positions and generate signals."""
    logger.debug("runtime_process_trading_bar entered at %s", timestamp.isoformat())

    result = {
        'action': 'trading',
        'micro_regime': session.micro_regime,
        'liquidity_sweep': _build_default_liquidity_sweep_payload(session),
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

    if warmup_only and _handle_warmup_only_mode(
        self,
        session=session,
        current_bar_index=current_bar_index,
        timestamp=timestamp,
        result=result,
    ):
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

    if _handle_portfolio_drawdown_halt(
        self,
        session=session,
        bar=bar,
        timestamp=timestamp,
        current_bar_index=current_bar_index,
        current_price=current_price,
        portfolio_drawdown=portfolio_drawdown,
        result=result,
    ):
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
    if _apply_max_daily_loss_circuit_breaker(
        self,
        session=session,
        bar=bar,
        timestamp=timestamp,
        current_bar_index=current_bar_index,
        current_price=current_price,
        result=result,
    ):
        return result


    regime_update = self._maybe_refresh_regime(session, current_bar_index, timestamp)
    if regime_update:
        _apply_regime_update_payload(result, regime_update)

    if session.active_position:
        _attach_active_position_intrabar_trace(
            self,
            session=session,
            bar=bar,
            timestamp=timestamp,
            result=result,
        )

    if _process_entry_signal_generation_pipeline(
        self,
        session=session,
        session_key=session_key,
        bar=bar,
        timestamp=timestamp,
        current_price=current_price,
        current_bar_index=current_bar_index,
        formula_indicators=_formula_indicators,
        result=result,
    ):
        return result

    logger.debug(
        "runtime_process_trading_bar completed: bar=%s has_liquidity_sweep=%s",
        current_bar_index,
        "liquidity_sweep" in result,
    )
    return result
