"""
FastAPI Server for Trading Strategy System.
Connects to the existing backtest API on localhost:8000.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import uvicorn
from datetime import datetime

from src.api_models import (
    BarInput,
    MultiLayerConfig,
    SessionQuery,
    StrategyToggle,
    StrategyUpdate,
    TradingConfig,
)
from src.api_utils import coerce_regimes, get_regime_description, normalize_strategy_name
from src.strategy_engine import StrategyEngine
from src.day_trading_manager import DayTradingManager



# FastAPI app
app = FastAPI(
    title="Trading Strategy API",
    description="Regime-based trading strategies with visualization",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    
    updated_fields = {}
    for key, val in config.params.items():
        if key == "allowed_regimes" and isinstance(val, list):
            regimes = coerce_regimes(val)
            setattr(strat, "allowed_regimes", regimes)
            updated_fields[key] = [r.value for r in regimes]
            if dtm_strat:
                setattr(dtm_strat, "allowed_regimes", regimes)
            continue
        # Only update existing attributes and basic numeric/bool/str types
        if hasattr(strat, key) and isinstance(val, (int, float, bool, str, type(None))):
            setattr(strat, key, val)
            updated_fields[key] = val
            if dtm_strat and hasattr(dtm_strat, key):
                setattr(dtm_strat, key, val)
    
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
        timestamp = datetime.fromisoformat(bar.timestamp.replace('Z', '+00:00'))
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid timestamp format: {bar.timestamp}")
    
    bar_data = {
        'open': bar.open,
        'high': bar.high,
        'low': bar.low,
        'close': bar.close,
        'volume': bar.volume,
        'vwap': bar.vwap,
        'l2_delta': bar.l2_delta,
        'l2_buy_volume': bar.l2_buy_volume,
        'l2_sell_volume': bar.l2_sell_volume,
        'l2_volume': bar.l2_volume,
        'l2_imbalance': bar.l2_imbalance,
        'l2_bid_depth_total': bar.l2_bid_depth_total,
        'l2_ask_depth_total': bar.l2_ask_depth_total,
        'l2_book_pressure': bar.l2_book_pressure,
        'l2_book_pressure_change': bar.l2_book_pressure_change,
        'l2_iceberg_buy_count': bar.l2_iceberg_buy_count,
        'l2_iceberg_sell_count': bar.l2_iceberg_sell_count,
        'l2_iceberg_bias': bar.l2_iceberg_bias,
    }
    
    result = day_trading_manager.process_bar(
        run_id=bar.run_id,
        ticker=bar.ticker,
        timestamp=timestamp,
        bar_data=bar_data
    )
    
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


@app.get("/api/config/trading")
async def get_trading_config():
    """Get current global trading configuration."""
    return {
        "regime_detection_minutes": day_trading_manager.regime_detection_minutes,
        "regime_refresh_bars": day_trading_manager.regime_refresh_bars,
        "max_daily_loss": day_trading_manager.max_daily_loss,
        "max_trades_per_day": day_trading_manager.max_trades_per_day,
        "trade_cooldown_bars": day_trading_manager.trade_cooldown_bars,
        "risk_per_trade_pct": day_trading_manager.risk_per_trade_pct,
        "max_position_notional_pct": day_trading_manager.max_position_notional_pct,
        "max_fill_participation_rate": day_trading_manager.max_fill_participation_rate,
        "min_fill_ratio": day_trading_manager.min_fill_ratio,
        "time_exit_bars": day_trading_manager.time_exit_bars,
        "enable_partial_take_profit": day_trading_manager.enable_partial_take_profit,
        "partial_take_profit_rr": day_trading_manager.partial_take_profit_rr,
        "partial_take_profit_fraction": day_trading_manager.partial_take_profit_fraction,
        "adverse_flow_exit_enabled": day_trading_manager.adverse_flow_exit_enabled,
        "adverse_flow_threshold": day_trading_manager.adverse_flow_exit_threshold,
        "adverse_flow_min_hold_bars": day_trading_manager.adverse_flow_min_hold_bars,
    }


@app.post("/api/config/trading")
async def update_trading_config(config: TradingConfig):
    """Update global trading configuration."""
    day_trading_manager.regime_detection_minutes = config.regime_detection_minutes
    day_trading_manager.regime_refresh_bars = max(3, int(config.regime_refresh_bars))
    day_trading_manager.max_daily_loss = config.max_daily_loss
    day_trading_manager.max_trades_per_day = config.max_trades_per_day
    day_trading_manager.trade_cooldown_bars = config.trade_cooldown_bars
    day_trading_manager.risk_per_trade_pct = max(0.1, float(config.risk_per_trade_pct))
    day_trading_manager.max_position_notional_pct = max(1.0, float(config.max_position_notional_pct))
    day_trading_manager.max_fill_participation_rate = min(
        1.0, max(0.01, float(config.max_fill_participation_rate))
    )
    day_trading_manager.min_fill_ratio = min(1.0, max(0.01, float(config.min_fill_ratio)))
    day_trading_manager.time_exit_bars = max(1, int(config.time_exit_bars))
    day_trading_manager.enable_partial_take_profit = bool(config.enable_partial_take_profit)
    day_trading_manager.partial_take_profit_rr = max(0.25, float(config.partial_take_profit_rr))
    day_trading_manager.partial_take_profit_fraction = min(
        0.95, max(0.05, float(config.partial_take_profit_fraction))
    )
    day_trading_manager.adverse_flow_exit_enabled = bool(config.adverse_flow_exit_enabled)
    day_trading_manager.adverse_flow_exit_threshold = max(0.02, float(config.adverse_flow_threshold))
    day_trading_manager.adverse_flow_min_hold_bars = max(1, int(config.adverse_flow_min_hold_bars))
    
    return {
        "message": "Trading configuration updated",
        "config": {
            "regime_detection_minutes": config.regime_detection_minutes,
            "regime_refresh_bars": day_trading_manager.regime_refresh_bars,
            "max_daily_loss": config.max_daily_loss,
            "max_trades_per_day": config.max_trades_per_day,
            "trade_cooldown_bars": config.trade_cooldown_bars,
            "risk_per_trade_pct": day_trading_manager.risk_per_trade_pct,
            "max_position_notional_pct": day_trading_manager.max_position_notional_pct,
            "max_fill_participation_rate": day_trading_manager.max_fill_participation_rate,
            "min_fill_ratio": day_trading_manager.min_fill_ratio,
            "time_exit_bars": day_trading_manager.time_exit_bars,
            "enable_partial_take_profit": day_trading_manager.enable_partial_take_profit,
            "partial_take_profit_rr": day_trading_manager.partial_take_profit_rr,
            "partial_take_profit_fraction": day_trading_manager.partial_take_profit_fraction,
            "adverse_flow_exit_enabled": day_trading_manager.adverse_flow_exit_enabled,
            "adverse_flow_threshold": day_trading_manager.adverse_flow_exit_threshold,
            "adverse_flow_min_hold_bars": day_trading_manager.adverse_flow_min_hold_bars,
        }
    }


@app.post("/api/session/config")
async def configure_session(
    run_id: str,
    ticker: str,
    date: str,
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
    time_exit_bars: int = 40,
    adverse_flow_exit_enabled: bool = True,
    adverse_flow_threshold: float = 0.12,
    adverse_flow_min_hold_bars: int = 3,
    l2_confirm_enabled: bool = False,
    l2_min_delta: float = 0.0,
    l2_min_imbalance: float = 0.0,
    l2_min_iceberg_bias: float = 0.0,
    l2_lookback_bars: int = 3,
    l2_min_participation_ratio: float = 0.0,
    l2_min_directional_consistency: float = 0.0,
    l2_min_signed_aggression: float = 0.0,
):
    """Configure session parameters before processing."""
    session = day_trading_manager.get_or_create_session(
        run_id=run_id,
        ticker=ticker,
        date=date,
        regime_detection_minutes=regime_detection_minutes
    )
    session.regime_detection_minutes = regime_detection_minutes
    session.regime_refresh_bars = max(3, int(regime_refresh_bars))
    day_trading_manager.regime_refresh_bars = max(3, int(regime_refresh_bars))
    session.account_size_usd = account_size_usd
    session.risk_per_trade_pct = max(0.1, float(risk_per_trade_pct))
    session.max_position_notional_pct = max(1.0, float(max_position_notional_pct))
    session.max_fill_participation_rate = min(1.0, max(0.01, float(max_fill_participation_rate)))
    session.min_fill_ratio = min(1.0, max(0.01, float(min_fill_ratio)))
    day_trading_manager.max_fill_participation_rate = session.max_fill_participation_rate
    day_trading_manager.min_fill_ratio = session.min_fill_ratio
    session.enable_partial_take_profit = bool(enable_partial_take_profit)
    session.partial_take_profit_rr = max(0.25, float(partial_take_profit_rr))
    session.partial_take_profit_fraction = min(0.95, max(0.05, float(partial_take_profit_fraction)))
    session.time_exit_bars = max(1, int(time_exit_bars))
    session.adverse_flow_exit_enabled = bool(adverse_flow_exit_enabled)
    session.adverse_flow_threshold = max(0.02, float(adverse_flow_threshold))
    session.adverse_flow_min_hold_bars = max(1, int(adverse_flow_min_hold_bars))
    day_trading_manager.set_run_defaults(
        run_id=run_id,
        ticker=ticker,
        regime_detection_minutes=regime_detection_minutes,
        regime_refresh_bars=regime_refresh_bars,
        account_size_usd=account_size_usd,
        risk_per_trade_pct=risk_per_trade_pct,
        max_position_notional_pct=max_position_notional_pct,
        max_fill_participation_rate=max_fill_participation_rate,
        min_fill_ratio=min_fill_ratio,
        enable_partial_take_profit=enable_partial_take_profit,
        partial_take_profit_rr=partial_take_profit_rr,
        partial_take_profit_fraction=partial_take_profit_fraction,
        time_exit_bars=time_exit_bars,
        adverse_flow_exit_enabled=adverse_flow_exit_enabled,
        adverse_flow_threshold=adverse_flow_threshold,
        adverse_flow_min_hold_bars=adverse_flow_min_hold_bars,
        l2_confirm_enabled=l2_confirm_enabled,
        l2_min_delta=l2_min_delta,
        l2_min_imbalance=l2_min_imbalance,
        l2_min_iceberg_bias=l2_min_iceberg_bias,
        l2_lookback_bars=l2_lookback_bars,
        l2_min_participation_ratio=l2_min_participation_ratio,
        l2_min_directional_consistency=l2_min_directional_consistency,
        l2_min_signed_aggression=l2_min_signed_aggression,
    )
    
    return {
        "message": "Session configured",
        "session": session.to_dict()
    }


# ============ Multi-Layer Decision Engine Config ============

@app.get("/api/multilayer/config")
async def get_multilayer_config():
    """Get current multi-layer decision engine configuration."""
    ml = day_trading_manager.multi_layer
    pd = ml.pattern_detector
    return {
        "pattern_weight": ml.pattern_weight,
        "strategy_weight": ml.strategy_weight,
        "threshold": ml.threshold,
        "require_pattern": ml.require_pattern,
        "detector": {
            "body_doji_pct": pd.body_doji_pct,
            "wick_ratio_hammer": pd.wick_ratio_hammer,
            "engulfing_min_body_pct": pd.engulfing_min_body_pct,
            "volume_confirm_ratio": pd.volume_confirm_ratio,
            "vwap_proximity_pct": pd.vwap_proximity_pct,
        },
    }


@app.post("/api/multilayer/config")
async def update_multilayer_config(config: MultiLayerConfig):
    """Update multi-layer decision engine configuration."""
    ml = day_trading_manager.multi_layer
    pd = ml.pattern_detector
    updated = {}

    if config.pattern_weight is not None:
        ml.pattern_weight = config.pattern_weight
        updated["pattern_weight"] = config.pattern_weight
    if config.strategy_weight is not None:
        ml.strategy_weight = config.strategy_weight
        updated["strategy_weight"] = config.strategy_weight
    if config.threshold is not None:
        ml.threshold = config.threshold
        updated["threshold"] = config.threshold
    if config.require_pattern is not None:
        ml.require_pattern = config.require_pattern
        updated["require_pattern"] = config.require_pattern

    # Detector settings
    if config.body_doji_pct is not None:
        pd.body_doji_pct = config.body_doji_pct
        updated["body_doji_pct"] = config.body_doji_pct
    if config.wick_ratio_hammer is not None:
        pd.wick_ratio_hammer = config.wick_ratio_hammer
        updated["wick_ratio_hammer"] = config.wick_ratio_hammer
    if config.engulfing_min_body_pct is not None:
        pd.engulfing_min_body_pct = config.engulfing_min_body_pct
        updated["engulfing_min_body_pct"] = config.engulfing_min_body_pct
    if config.volume_confirm_ratio is not None:
        pd.volume_confirm_ratio = config.volume_confirm_ratio
        updated["volume_confirm_ratio"] = config.volume_confirm_ratio
    if config.vwap_proximity_pct is not None:
        pd.vwap_proximity_pct = config.vwap_proximity_pct
        updated["vwap_proximity_pct"] = config.vwap_proximity_pct

    return {
        "message": "Multi-layer config updated",
        "updated": updated,
        "current": (await get_multilayer_config()),
    }


# ============ WebSocket for real-time updates ============


from fastapi import WebSocket, WebSocketDisconnect
import asyncio

connected_clients: List[WebSocket] = []


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
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
