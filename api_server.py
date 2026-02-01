"""
FastAPI Server for Trading Strategy System.
Connects to the existing backtest API on localhost:8000.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
from datetime import datetime

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
day_trading_manager = DayTradingManager(regime_detection_minutes=15)


# ============ Pydantic Models ============

class StrategyToggle(BaseModel):
    strategy_name: str
    enabled: bool


class TrailingStopConfig(BaseModel):
    stop_type: str = "PERCENT"
    initial_stop_pct: float = 2.0
    trailing_pct: float = 0.8
    atr_multiplier: float = 2.0


class BarInput(BaseModel):
    """Input model for processing a single bar."""
    run_id: str
    ticker: str
    timestamp: str  # ISO format datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None


class SessionQuery(BaseModel):
    """Query model for session operations."""
    run_id: str
    ticker: str
    date: str  # YYYY-MM-DD format


class StrategyToggle(BaseModel):
    strategy_name: str
    enabled: bool


class TrailingStopConfig(BaseModel):
    stop_type: str = "PERCENT"
    initial_stop_pct: float = 2.0
    trailing_pct: float = 0.8
    atr_multiplier: float = 2.0


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
            "description": _get_regime_description(regime.value),
            "active_strategies": [s.name for s in engine.get_active_strategies(regime)]
        }
    
    return {"regime": "MIXED", "error": "Could not fetch data"}


def _get_regime_description(regime: str) -> str:
    descriptions = {
        "TRENDING": "Strong directional movement. Pullback and Momentum strategies preferred.",
        "CHOPPY": "Range-bound, low trend efficiency. Mean Reversion and VWAP Magnet strategies preferred.",
        "MIXED": "Uncertain direction. Rotation and conservative strategies preferred."
    }
    return descriptions.get(regime, "Unknown regime")


@app.get("/api/strategies")
async def get_strategies():
    """Get all strategies with their configuration."""
    return {
        name: strategy.to_dict() 
        for name, strategy in engine.strategies.items()
    }


@app.post("/api/strategies/toggle")
async def toggle_strategy(config: StrategyToggle):
    """Enable or disable a strategy."""
    strat_name = config.strategy_name.lower().replace(' ', '_')
    
    if strat_name not in engine.strategies:
        raise HTTPException(status_code=404, detail=f"Strategy {config.strategy_name} not found")
    
    engine.strategies[strat_name].enabled = config.enabled
    
    return {
        "strategy": config.strategy_name,
        "enabled": config.enabled,
        "message": f"Strategy {config.strategy_name} {'enabled' if config.enabled else 'disabled'}"
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
        'vwap': bar.vwap
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


@app.post("/api/session/config")
async def configure_session(run_id: str, ticker: str, date: str, regime_detection_minutes: int = 15):
    """Configure session parameters before processing."""
    session = day_trading_manager.get_or_create_session(
        run_id=run_id,
        ticker=ticker,
        date=date,
        regime_detection_minutes=regime_detection_minutes
    )
    
    return {
        "message": "Session configured",
        "session": session.to_dict()
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

