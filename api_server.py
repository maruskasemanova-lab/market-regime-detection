"""
FastAPI Server for Trading Strategy System.
Connects to the existing backtest API on localhost:8000.
"""
import os
import re

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse
import uvicorn

from src.api_models import (
    BarInput,
    SessionQuery,
    StrategyToggle,
    StrategyUpdate,
    TradingConfig as TradingConfigModel,
)
from src.api_utils import coerce_regimes, get_regime_description, normalize_strategy_name
from src.strategy_engine import StrategyEngine
from src.day_trading_manager import DayTradingManager
from src.trading_config import TradingConfig
from src.strategy_formula_engine import (
    validate_strategy_formula,
)
from src.api_server_helpers.bar_payload import (
    build_day_trading_bar_payload,
    parse_bar_timestamp,
    sanitize_non_finite_numbers,
)
from src.api_server_helpers.session_config_apply import (
    apply_session_config_resolution,
    resolve_session_config_request,
)
from src.api_server_helpers.orchestrator_config import (
    apply_orchestrator_config_updates,
    serialize_orchestrator_config,
)
from src.api_server_helpers.strategy_update import (
    apply_strategy_param_updates,
)
from src.api_server_helpers.trading_config import (
    apply_global_trading_config,
    apply_trading_limits,
    read_global_trading_config,
    read_trading_limits,
)



# FastAPI app
app = FastAPI(
    title="Trading Strategy API",
    description="Regime-based trading strategies with visualization",
    version="1.0.0"
)


_cors_allow_origins_raw = str(
    os.getenv(
        "STRATEGY_CORS_ALLOW_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8002,http://127.0.0.1:8002",
    )
    or ""
).strip()
if not _cors_allow_origins_raw:
    _cors_allow_origins: List[str] = []
else:
    _cors_allow_origins = list(
        dict.fromkeys(
            token.strip()
            for token in _cors_allow_origins_raw.split(",")
            if token.strip()
        )
    )
if "*" in _cors_allow_origins:
    _cors_allow_origins = ["*"]

_cors_allow_origin_regex = str(
    os.getenv("STRATEGY_CORS_ALLOW_ORIGIN_REGEX", "") or ""
).strip()
if not _cors_allow_origin_regex:
    _cors_patterns: List[str] = []
    _cors_seen: set[str] = set()
    for _origin in _cors_allow_origins:
        try:
            _parsed = urlparse(str(_origin))
        except Exception:
            continue
        _host = str(_parsed.hostname or "").strip().lower()
        if not _host.endswith(".netlify.app"):
            continue
        _pattern = rf"https://(?:[a-z0-9-]+--)?{re.escape(_host)}"
        if _pattern in _cors_seen:
            continue
        _cors_seen.add(_pattern)
        _cors_patterns.append(_pattern)
    _cors_allow_origin_regex = (
        rf"^(?:{'|'.join(_cors_patterns)})$" if _cors_patterns else None
    )

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins or ["http://localhost:5173"],
    allow_origin_regex=_cors_allow_origin_regex,
    allow_credentials=("*" not in _cors_allow_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)

_INTERNAL_API_TOKEN = str(os.getenv("STRATEGY_INTERNAL_API_TOKEN") or "").strip()
_INTERNAL_PROTECTED_PREFIXES = (
    "/api/session/",
    "/api/orchestrator/",
    "/api/strategies/update",
    "/api/strategies/toggle",
)
_INTERNAL_PUBLIC_ROUTES = {
    ("GET", "/api/orchestrator/checkpoints"),
}


def _apply_cors_headers_for_error(request: Request, response: JSONResponse) -> JSONResponse:
    origin = str(request.headers.get("origin") or "").strip()
    if not origin:
        return response

    allow_origin: Optional[str] = None
    if "*" in _cors_allow_origins:
        allow_origin = "*"
    elif origin in _cors_allow_origins:
        allow_origin = origin
    elif _cors_allow_origin_regex and re.match(_cors_allow_origin_regex, origin):
        allow_origin = origin

    if not allow_origin:
        return response

    response.headers["access-control-allow-origin"] = allow_origin
    if allow_origin != "*":
        response.headers["access-control-allow-credentials"] = "true"
    response.headers["vary"] = "Origin"
    return response


@app.middleware("http")
async def _internal_api_token_guard(request: Request, call_next):
    if not _INTERNAL_API_TOKEN:
        return await call_next(request)

    method = str(request.method or "").upper()
    path = str(request.url.path or "")
    if method == "OPTIONS":
        return await call_next(request)
    if (method, path) in _INTERNAL_PUBLIC_ROUTES:
        return await call_next(request)
    if not any(path.startswith(prefix) for prefix in _INTERNAL_PROTECTED_PREFIXES):
        return await call_next(request)

    supplied = str(request.headers.get("x-internal-token") or "").strip()
    if supplied != _INTERNAL_API_TOKEN:
        return _apply_cors_headers_for_error(
            request,
            JSONResponse(
                status_code=403,
                content={"detail": "Forbidden: internal API token required."},
            ),
        )
    return await call_next(request)

# Strategy engine instance
engine = StrategyEngine(backtest_api_url="http://localhost:8000")

# Day trading manager for session-based API
day_trading_manager = DayTradingManager(regime_detection_minutes=5)


# ============ API Endpoints ============

@app.get("/")
async def root():
    return {"message": "Trading Strategy API", "status": "running"}


@app.get("/api/state")
async def get_state():
    """Get current engine state including regime, strategies, positions."""
    return engine.get_state()


@app.get("/api/regime")
async def get_regime():
    """Get current market regime."""
    # Fetch fresh data to detect regime
    ohlcv = engine.fetch_history(100)
    indicators = engine.fetch_all_indicators()
    
    if ohlcv and 'close' in ohlcv:
        regime = engine.detect_regime(ohlcv, indicators)
        engine.current_regime = regime
        
        return {
            "regime": regime.value,
            "description": get_regime_description(regime.value),
            "active_strategies": [s.name for s in engine.get_active_strategies(regime)]
        }
    
    return {"regime": "MIXED", "error": "Could not fetch data"}


@app.get("/api/strategies")
async def get_strategies():
    """Get all strategies with their configuration."""
    # Normalize allowed_regimes to Regime enums to avoid serialization errors
    for strat in engine.strategies.values():
        if hasattr(strat, "allowed_regimes"):
            strat.allowed_regimes = coerce_regimes(getattr(strat, "allowed_regimes"))
    for strat in day_trading_manager.strategies.values():
        if hasattr(strat, "allowed_regimes"):
            strat.allowed_regimes = coerce_regimes(getattr(strat, "allowed_regimes"))
    return {
        name: strategy.to_dict() 
        for name, strategy in engine.strategies.items()
    }


@app.post("/api/strategies/toggle")
async def toggle_strategy(config: StrategyToggle):
    """Enable or disable a strategy."""
    strat_name = normalize_strategy_name(config.strategy_name)
    
    if strat_name not in engine.strategies:
        raise HTTPException(status_code=404, detail=f"Strategy {config.strategy_name} not found")
    
    engine.strategies[strat_name].enabled = config.enabled
    # Keep day-trading manager in sync for session-based trading
    if strat_name in day_trading_manager.strategies:
        day_trading_manager.strategies[strat_name].enabled = config.enabled
    
    return {
        "strategy": config.strategy_name,
        "enabled": config.enabled,
        "message": f"Strategy {config.strategy_name} {'enabled' if config.enabled else 'disabled'}"
    }


@app.post("/api/strategies/update")
async def update_strategy(config: StrategyUpdate):
    """Update editable parameters of a strategy (numeric fields + allowed_regimes)."""
    strat_name = normalize_strategy_name(config.strategy_name)
    
    if strat_name not in engine.strategies:
        raise HTTPException(status_code=404, detail=f"Strategy {config.strategy_name} not found")
    
    strat = engine.strategies[strat_name]
    dtm_strat = day_trading_manager.strategies.get(strat_name)
    try:
        updated_fields = apply_strategy_param_updates(
            strat=strat,
            dtm_strat=dtm_strat,
            params=config.params,
            validate_formula=validate_strategy_formula,
            coerce_regimes=coerce_regimes,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    
    return {
        "strategy": config.strategy_name,
        "updated": updated_fields,
        "current": strat.to_dict()
    }


@app.get("/api/signals")
async def get_signals(limit: int = 50):
    """Get recent signals."""
    signals = engine.all_signals[-limit:]
    return {
        "total": len(engine.all_signals),
        "signals": [s.to_dict() for s in signals]
    }


@app.get("/api/positions")
async def get_positions():
    """Get open positions."""
    return {
        "open_positions": {
            k: v.to_dict() for k, v in engine.open_positions.items()
        },
        "count": len(engine.open_positions)
    }


@app.get("/api/trades")
async def get_trades(limit: int = 100):
    """Get trade history."""
    trades = engine.all_trades[-limit:]
    return {
        "total": len(engine.all_trades),
        "trades": [t.to_dict() for t in trades]
    }


@app.get("/api/performance")
async def get_performance():
    """Get performance summary."""
    return engine.get_performance_summary()


@app.post("/api/step")
async def step_backtest():
    """Advance backtest by one bar and process strategies."""
    result = engine.step()
    return result


@app.post("/api/run")
async def run_backtest(bars: int = 10):
    """Run multiple backtest steps."""
    results = []
    for _ in range(bars):
        result = engine.step()
        results.append(result)
    
    return {
        "bars_processed": bars,
        "final_state": engine.get_state(),
        "performance": engine.get_performance_summary(),
        "results": results[-5:]  # Last 5 results
    }


@app.post("/api/reset")
async def reset_engine():
    """Reset the strategy engine state."""
    global engine
    engine = StrategyEngine(backtest_api_url="http://localhost:8000")
    return {"message": "Engine reset", "state": engine.get_state()}


@app.get("/api/current")
async def get_current_price():
    """Get current price from backtest API."""
    data = engine.fetch_current_data()
    return data


@app.get("/api/history")
async def get_history(bars: int = 100):
    """Get historical OHLCV data."""
    ohlcv = engine.fetch_history(bars)
    return ohlcv


@app.get("/api/indicators")
async def get_all_indicators():
    """Get all indicators."""
    return engine.fetch_all_indicators()


# ============ Session-Based Day Trading API ============

@app.post("/api/session/intrabar_eval")
async def evaluate_intrabar_slice(bar: BarInput):
    """
    Evaluate a 5s intrabar slice side-effect-free without modifying the session state.
    Returns the layer scores, signal decision, and thresholds.
    """
    try:
        timestamp = parse_bar_timestamp(bar.timestamp)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid timestamp format: {bar.timestamp}")
    bar_data = build_day_trading_bar_payload(
        bar,
        include_l2_quality_flags=False,
    )
    
    result = day_trading_manager.evaluate_intrabar_slice(
        run_id=bar.run_id,
        ticker=bar.ticker,
        timestamp=timestamp,
        bar_data=bar_data,
    )

    return sanitize_non_finite_numbers(result)


@app.post("/api/session/bar")
async def process_bar(bar: BarInput):
    """
    Process a new bar for a trading session.
    
    This endpoint receives data from your external engine/environment
    and tracks the state for each run_id + ticker + date combination.
    
    Flow:
    1. Pre-market: Stores bars for regime context
    2. First X minutes (default 15): Collects data for regime detection
    3. After regime detection: Selects strategy and trades
    4. End of day: Closes positions and returns summary
    """
    try:
        timestamp = parse_bar_timestamp(bar.timestamp)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid timestamp format: {bar.timestamp}")
    bar_data = build_day_trading_bar_payload(
        bar,
        include_l2_quality_flags=True,
    )
    
    if bar.tcbbo_has_data:
        print(f"DEBUG: api_server received TCBBO data: has_data={bar.tcbbo_has_data}, net_premium={bar.tcbbo_net_premium}")

    result = day_trading_manager.process_bar(
        run_id=bar.run_id,
        ticker=bar.ticker,
        timestamp=timestamp,
        bar_data=bar_data,
        warmup_only=bool(bar.warmup_only),
    )

    # Sanitize NaN/Inf values that crash JSON serialization
    result = sanitize_non_finite_numbers(result)

    # Feed cross-asset reference bar to orchestrator if provided
    if bar.ref_ticker and bar.ref_close:
        orch = day_trading_manager.orchestrator
        if orch and orch.config.use_cross_asset:
            ref_bar = {
                'open': bar.ref_open or bar.ref_close,
                'high': bar.ref_high or bar.ref_close,
                'low': bar.ref_low or bar.ref_close,
                'close': bar.ref_close,
                'volume': bar.ref_volume or 0,
            }
            orch.update_cross_asset(bar.ref_ticker, ref_bar)
            orch.update_target_cross_asset({
                'close': bar.close,
                'volume': bar.volume,
            })

    return result


@app.get("/api/session")
async def get_session(run_id: str, ticker: str, date: str):
    """Get current session state."""
    session = day_trading_manager.get_session(run_id, ticker, date)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session.to_dict()


@app.get("/api/session/signals")
async def get_session_signals(run_id: str, ticker: str, date: str):
    """Get signals for a session."""
    session = day_trading_manager.get_session(run_id, ticker, date)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "signals": [s.to_dict() for s in session.signals],
        "count": len(session.signals)
    }


@app.get("/api/session/trades")
async def get_session_trades(run_id: str, ticker: str, date: str):
    """Get trades for a session."""
    session = day_trading_manager.get_session(run_id, ticker, date)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "trades": [t.to_dict() for t in session.trades],
        "count": len(session.trades),
        "total_pnl": session.total_pnl
    }


@app.post("/api/session/end")
async def end_session(query: SessionQuery):
    """Manually end a session and get the full summary."""
    summary = day_trading_manager.end_session(
        run_id=query.run_id,
        ticker=query.ticker,
        date=query.date
    )
    
    if not summary:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return summary


@app.get("/api/sessions")
async def list_sessions():
    """List all active sessions."""
    return day_trading_manager.get_all_sessions()


@app.delete("/api/session")
async def clear_session(run_id: str, ticker: str, date: str):
    """Clear a session from memory."""
    success = day_trading_manager.clear_session(run_id, ticker, date)
    
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"message": "Session cleared", "run_id": run_id, "ticker": ticker, "date": date}


@app.delete("/api/session/run")
async def clear_run_sessions(run_id: str, ticker: Optional[str] = None):
    """Clear all sessions and run-scoped state for a run_id (optionally for one ticker)."""
    normalized_ticker = ticker.upper() if ticker else None
    removed = day_trading_manager.clear_sessions_for_run(run_id, normalized_ticker)
    return {
        "message": "Run sessions cleared",
        "run_id": run_id,
        "ticker": normalized_ticker,
        "cleared_sessions": removed,
    }


@app.get("/api/config/trading")
async def get_trading_config():
    """Get current global trading configuration."""
    payload = read_trading_limits(day_trading_manager)
    payload.update(read_global_trading_config(day_trading_manager))
    return payload


@app.post("/api/config/trading")
async def update_trading_config(config: TradingConfigModel):
    """Update global trading configuration."""
    canonical = TradingConfig.from_dict(config.model_dump())
    apply_trading_limits(day_trading_manager, config.model_dump())
    apply_global_trading_config(day_trading_manager, canonical)

    response_config = read_trading_limits(day_trading_manager)
    response_config.update(canonical.to_session_params())
    return {
        "message": "Trading configuration updated",
        "config": response_config,
    }


@app.post("/api/session/config")
async def configure_session(
    run_id: str,
    ticker: str,
    date: str,
    request: Request,
    regime_detection_minutes: int = 15,
    regime_refresh_bars: int = 12,
    account_size_usd: float = 10_000.0,
    risk_per_trade_pct: float = 1.0,
    max_position_notional_pct: float = 100.0,
    max_fill_participation_rate: float = 0.20,
    min_fill_ratio: float = 0.35,
    enable_partial_take_profit: bool = True,
    partial_take_profit_rr: float = 1.0,
    partial_take_profit_fraction: float = 0.5,
    partial_protect_min_mfe_r: float = 0.0,
    trailing_activation_pct: float = 0.15,
    break_even_buffer_pct: float = 0.03,
    break_even_min_hold_bars: int = 3,
    break_even_activation_min_mfe_pct: float = 0.25,
    break_even_activation_min_r: float = 0.60,
    break_even_activation_min_r_trending_5m: float = 0.90,
    break_even_activation_min_r_choppy_5m: float = 0.60,
    break_even_activation_use_levels: bool = True,
    break_even_activation_use_l2: bool = True,
    break_even_level_buffer_pct: float = 0.02,
    break_even_level_max_distance_pct: float = 0.60,
    break_even_level_min_confluence: int = 2,
    break_even_level_min_tests: int = 1,
    break_even_l2_signed_aggression_min: float = 0.12,
    break_even_l2_imbalance_min: float = 0.15,
    break_even_l2_book_pressure_min: float = 0.10,
    break_even_l2_spread_bps_max: float = 12.0,
    break_even_costs_pct: float = 0.03,
    break_even_min_buffer_pct: float = 0.05,
    break_even_atr_buffer_k: float = 0.10,
    break_even_5m_atr_buffer_k: float = 0.10,
    break_even_tick_size: float = 0.01,
    break_even_min_tick_buffer: int = 1,
    break_even_anti_spike_bars: int = 1,
    break_even_anti_spike_hits_required: int = 2,
    break_even_anti_spike_require_close_beyond: bool = True,
    break_even_5m_no_go_proximity_pct: float = 0.10,
    break_even_5m_mfe_atr_factor: float = 0.15,
    break_even_5m_l2_bias_threshold: float = 0.10,
    break_even_5m_l2_bias_tighten_factor: float = 0.85,
    break_even_movement_formula_enabled: bool = False,
    break_even_movement_formula: str = "",
    break_even_proof_formula_enabled: bool = False,
    break_even_proof_formula: str = "",
    break_even_activation_formula_enabled: bool = False,
    break_even_activation_formula: str = "",
    break_even_trailing_handoff_formula_enabled: bool = False,
    break_even_trailing_handoff_formula: str = "",
    trailing_enabled_in_choppy: bool = False,
    time_exit_bars: int = 40,
    time_exit_formula_enabled: bool = False,
    time_exit_formula: str = "",
    adverse_flow_exit_enabled: bool = True,
    adverse_flow_threshold: float = 0.12,
    adverse_flow_min_hold_bars: int = 3,
    adverse_flow_consistency_threshold: float = 0.45,
    adverse_book_pressure_threshold: float = 0.15,
    adverse_flow_exit_formula_enabled: bool = False,
    adverse_flow_exit_formula: str = "",
    stop_loss_mode: str = "strategy",
    fixed_stop_loss_pct: float = 0.0,
    l2_confirm_enabled: bool = False,
    l2_min_delta: float = 0.0,
    l2_min_imbalance: float = 0.0,
    l2_min_iceberg_bias: float = 0.0,
    l2_lookback_bars: int = 3,
    l2_min_participation_ratio: float = 0.0,
    l2_min_directional_consistency: float = 0.0,
    l2_min_signed_aggression: float = 0.0,
    tcbbo_gate_enabled: bool = False,
    tcbbo_min_net_premium: float = 0.0,
    tcbbo_sweep_boost: float = 5.0,
    tcbbo_lookback_bars: int = 5,
    intraday_levels_enabled: bool = True,
    intraday_levels_swing_left_bars: int = 2,
    intraday_levels_swing_right_bars: int = 2,
    intraday_levels_test_tolerance_pct: float = 0.08,
    intraday_levels_break_tolerance_pct: float = 0.05,
    intraday_levels_breakout_volume_lookback: int = 20,
    intraday_levels_breakout_volume_multiplier: float = 1.2,
    intraday_levels_volume_profile_bin_size_pct: float = 0.05,
    intraday_levels_value_area_pct: float = 0.70,
    intraday_levels_entry_quality_enabled: bool = True,
    intraday_levels_min_levels_for_context: int = 2,
    intraday_levels_entry_tolerance_pct: float = 0.10,
    intraday_levels_break_cooldown_bars: int = 6,
    intraday_levels_rotation_max_tests: int = 2,
    intraday_levels_rotation_volume_max_ratio: float = 0.95,
    intraday_levels_recent_bounce_lookback_bars: int = 6,
    intraday_levels_require_recent_bounce_for_mean_reversion: bool = True,
    intraday_levels_momentum_break_max_age_bars: int = 3,
    intraday_levels_momentum_min_room_pct: float = 0.30,
    intraday_levels_momentum_min_broken_ratio: float = 0.30,
    intraday_levels_min_confluence_score: int = 2,
    intraday_levels_memory_enabled: bool = True,
    intraday_levels_memory_min_tests: int = 2,
    intraday_levels_memory_max_age_days: int = 5,
    intraday_levels_memory_decay_after_days: int = 2,
    intraday_levels_memory_decay_weight: float = 0.50,
    intraday_levels_memory_max_levels: int = 12,
    intraday_levels_opening_range_enabled: bool = True,
    intraday_levels_opening_range_minutes: int = 30,
    intraday_levels_opening_range_break_tolerance_pct: float = 0.05,
    intraday_levels_poc_migration_enabled: bool = True,
    intraday_levels_poc_migration_interval_bars: int = 30,
    intraday_levels_poc_migration_trend_threshold_pct: float = 0.20,
    intraday_levels_poc_migration_range_threshold_pct: float = 0.10,
    intraday_levels_composite_profile_enabled: bool = True,
    intraday_levels_composite_profile_days: int = 3,
    intraday_levels_composite_profile_current_day_weight: float = 1.0,
    intraday_levels_spike_detection_enabled: bool = True,
    intraday_levels_spike_min_wick_ratio: float = 0.60,
    intraday_levels_prior_day_anchors_enabled: bool = True,
    intraday_levels_gap_analysis_enabled: bool = True,
    intraday_levels_gap_min_pct: float = 0.30,
    intraday_levels_gap_momentum_threshold_pct: float = 2.0,
    intraday_levels_rvol_filter_enabled: bool = True,
    intraday_levels_rvol_lookback_bars: int = 20,
    intraday_levels_rvol_min_threshold: float = 0.80,
    intraday_levels_rvol_strong_threshold: float = 1.50,
    intraday_levels_adaptive_window_enabled: bool = True,
    intraday_levels_adaptive_window_min_bars: int = 6,
    intraday_levels_adaptive_window_rvol_threshold: float = 1.0,
    intraday_levels_adaptive_window_atr_ratio_max: float = 1.5,
    intraday_levels_micro_confirmation_enabled: bool = False,
    intraday_levels_micro_confirmation_bars: int = 2,
    intraday_levels_micro_confirmation_disable_for_sweep: bool = False,
    intraday_levels_micro_confirmation_sweep_bars: int = 0,
    intraday_levels_micro_confirmation_require_intrabar: bool = False,
    intraday_levels_micro_confirmation_intrabar_window_seconds: int = 5,
    intraday_levels_micro_confirmation_intrabar_min_coverage_points: int = 3,
    intraday_levels_micro_confirmation_intrabar_min_move_pct: float = 0.02,
    intraday_levels_micro_confirmation_intrabar_min_push_ratio: float = 0.10,
    intraday_levels_micro_confirmation_intrabar_max_spread_bps: float = 12.0,
    intraday_levels_confluence_sizing_enabled: bool = False,
    liquidity_sweep_detection_enabled: bool = False,
    sweep_min_aggression_z: float = -2.0,
    sweep_min_book_pressure_z: float = 1.5,
    sweep_max_price_change_pct: float = 0.05,
    sweep_atr_buffer_multiplier: float = 0.5,
    context_aware_risk_enabled: bool = False,
    context_risk_sl_buffer_pct: float = 0.03,
    context_risk_min_sl_pct: float = 0.30,
    context_risk_min_room_pct: float = 0.15,
    context_risk_min_effective_rr: float = 0.80,
    context_risk_trailing_tighten_zone: float = 0.20,
    context_risk_trailing_tighten_factor: float = 0.50,
    context_risk_level_trail_enabled: bool = True,
    context_risk_max_anchor_search_pct: float = 1.5,
    context_risk_min_level_tests_for_sl: int = 1,
    cold_start_each_day: bool = False,
    strategy_selection_mode: str = "adaptive_top_n",
    max_active_strategies: int = 3,
    momentum_diversification_json: str = "",
    max_daily_trades: Optional[int] = None,
    mu_choppy_hard_block_enabled: Optional[bool] = None,
    regime_filter_json: str = "",
    micro_confirmation_mode: str = "consecutive_close",
    micro_confirmation_volume_delta_min_pct: float = 0.60,
    weak_l2_fast_break_even_enabled: bool = False,
    weak_l2_aggression_threshold: float = 0.05,
    weak_l2_break_even_min_hold_bars: int = 2,
    ev_relaxation_enabled: bool = False,
    ev_relaxation_threshold: float = 10.0,
    ev_relaxation_factor: float = 0.50,
    intraday_levels_bounce_conflict_buffer_bars: int = 0,
    orchestrator_strategy_weight: float = 0.6,
    orchestrator_strategy_only_threshold: float = 0.0,
):
    """Configure session parameters before processing."""
    session = day_trading_manager.get_or_create_session(
        run_id=run_id,
        ticker=ticker,
        date=date,
        regime_detection_minutes=regime_detection_minutes
    )
    local_values = dict(locals())
    query_param_keys = {str(key) for key in request.query_params.keys()}
    body_payload: Dict[str, Any] = {}
    try:
        parsed_body = await request.json()
    except Exception:
        parsed_body = None
    if isinstance(parsed_body, dict):
        body_payload = parsed_body

    try:
        resolution = resolve_session_config_request(
            local_values=local_values,
            body_payload=body_payload,
            query_param_keys=query_param_keys,
            momentum_diversification_json=momentum_diversification_json,
            regime_filter_json=regime_filter_json,
            max_daily_trades=max_daily_trades,
            mu_choppy_hard_block_enabled=mu_choppy_hard_block_enabled,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        return apply_session_config_resolution(
            manager=day_trading_manager,
            session=session,
            run_id=run_id,
            ticker=ticker,
            resolution=resolution,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ============ Orchestrator Health & Config ============


@app.post("/api/orchestrator/reset")
async def reset_orchestrator_state(scope: str = "all", clear_sessions: bool = True):
    """
    Reset orchestrator/manager state for deterministic backtests.

    scope:
      - session: feature/regime/cross-asset runtime only
      - learning: calibrator/combiner/edge monitor only
      - all: both runtime + learning
    """
    try:
        summary = day_trading_manager.reset_backtest_state(
            scope=scope,
            clear_sessions=clear_sessions,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "message": "Orchestrator state reset",
        **summary,
    }


@app.get("/api/system/health")
async def get_system_health():
    """Get orchestrator system health for observability."""
    orch = day_trading_manager.orchestrator
    if not orch:
        return {"status": "orchestrator_not_initialized"}
    return {"status": "ok", "health": orch.get_system_health()}


@app.get("/api/orchestrator/config")
async def get_orchestrator_config():
    """Get current orchestrator configuration."""
    orch = day_trading_manager.orchestrator
    if not orch:
        return {"status": "not_initialized"}
    return serialize_orchestrator_config(orch)


@app.post("/api/orchestrator/config")
async def update_orchestrator_config(body: Dict[str, Any]):
    """Toggle orchestrator feature flags."""
    orch = day_trading_manager.orchestrator
    if not orch:
        raise HTTPException(400, "Orchestrator not initialized")
    updated = apply_orchestrator_config_updates(orch=orch, body=body)
    return {"message": "Orchestrator config updated", "updated": updated}


# ============ Checkpoint Persistence ============


@app.post("/api/orchestrator/checkpoint/save")
async def save_checkpoint(
    run_id: Optional[str] = None,
    ticker: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    path: Optional[str] = None,
):
    """Save orchestrator learning state to a checkpoint file."""
    orch = day_trading_manager.orchestrator
    if not orch:
        raise HTTPException(400, "Orchestrator not initialized")
    metadata = {
        k: v for k, v in {
            "run_id": run_id, "ticker": ticker,
            "date_from": date_from, "date_to": date_to,
        }.items() if v is not None
    }
    saved_path = orch.save_checkpoint(path=path, metadata=metadata)
    return {"status": "saved", "path": saved_path}


@app.post("/api/orchestrator/checkpoint/load")
async def load_checkpoint_endpoint(path: str):
    """Load orchestrator learning state from a checkpoint file."""
    orch = day_trading_manager.orchestrator
    if not orch:
        raise HTTPException(400, "Orchestrator not initialized")
    try:
        info = orch.load_checkpoint(path)
        return {"status": "loaded", "checkpoint": info}
    except FileNotFoundError:
        raise HTTPException(404, f"Checkpoint not found: {path}")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/orchestrator/checkpoints")
async def list_checkpoints():
    """List available checkpoint files."""
    import json as _json
    cp_dir = Path("data/checkpoints")
    if not cp_dir.exists():
        return []
    files = sorted(cp_dir.glob("checkpoint_*.json"), reverse=True)
    results = []
    for f in files:
        try:
            data = _json.loads(f.read_text())
            results.append({
                "path": str(f),
                "created_at": data.get("created_at"),
                "source": data.get("source", {}),
                "version": data.get("version"),
            })
        except Exception:
            pass
    return results


@app.post("/api/orchestrator/warmup")
async def warmup_feature_store(bars: List[Dict[str, Any]]):
    """Feed historical bars to warm up feature store z-score windows."""
    orch = day_trading_manager.orchestrator
    if not orch:
        raise HTTPException(400, "Orchestrator not initialized")
    count = orch.warmup_feature_store(bars)
    return {"status": "warmed_up", "bars_processed": count}


# ============ WebSocket for real-time updates ============


from fastapi import WebSocket, WebSocketDisconnect
import asyncio

connected_clients: List[WebSocket] = []
MAX_WS_CLIENTS = max(1, int(os.getenv("STRATEGY_MAX_WS_CLIENTS", "120")))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if len(connected_clients) >= MAX_WS_CLIENTS:
        await websocket.close(code=1013, reason="Too many websocket clients")
        return
    await websocket.accept()
    connected_clients.append(websocket)
    
    try:
        while True:
            # Listen for commands
            data = await websocket.receive_json()
            
            if data.get("command") == "step":
                result = engine.step()
                await websocket.send_json({
                    "type": "step_result",
                    "data": result
                })
            
            elif data.get("command") == "state":
                await websocket.send_json({
                    "type": "state",
                    "data": engine.get_state()
                })
            
            elif data.get("command") == "run":
                bars = data.get("bars", 10)
                for i in range(bars):
                    result = engine.step()
                    await websocket.send_json({
                        "type": "step_result",
                        "bar": i + 1,
                        "total": bars,
                        "data": result
                    })
                    await asyncio.sleep(0.1)  # Small delay for visualization
                    
    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


# ============ Main ============

if __name__ == "__main__":
    print("=" * 60)
    print("TRADING STRATEGY API SERVER")
    print("=" * 60)
    print("\nMake sure the backtest API is running on localhost:8000")
    print("\nEndpoints:")
    print("  GET  /api/state       - Current state")
    print("  GET  /api/regime      - Current regime")
    print("  GET  /api/strategies  - All strategies")
    print("  GET  /api/signals     - Recent signals")
    print("  GET  /api/positions   - Open positions")
    print("  GET  /api/trades      - Trade history")
    print("  GET  /api/performance - Performance summary")
    print("  POST /api/step        - Advance one bar")
    print("  POST /api/run         - Run multiple bars")
    print("  WS   /ws              - WebSocket for real-time")
    print("\nSession-Based Day Trading API:")
    print("  POST /api/session/bar - Process a bar for session")
    print("  GET  /api/session     - Get session state")
    print("  POST /api/session/end - End session and get summary")
    print("  GET  /api/sessions    - List all sessions")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
